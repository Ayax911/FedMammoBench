"""Supervisor desatendido del grid de 7 celdas Fed-ISIC2019 (noche sin usuario despierto).

Corre las 7 celdas (`configs/federated/fedisic2019_*/`) secuencialmente vía
`run_cell.py`, con una escalera de reintentos automática por celda cuando algo
falla, para que una celda mala no bloquee el resto de la noche. Nunca toca
hiperparámetros de estrategia (lr/mu/eta/E/rounds) -- el único dial que mueve
es `batch_size` (Plan B/C), que es un ajuste de hardware, no del paper, y
queda documentado en el resumen final.

Escalera por celda:
  intento 1: batch_size del YAML (64, el del paper), timeout `ROUND_TIMEOUT_S`
             con vigilancia de atasco (sin crecimiento del log en
             `STALL_TIMEOUT_S` -> se mata y cuenta como fallo de ese intento).
  intento 2: si el intento 1 falló por OutOfMemoryError -> batch_size=32.
  intento 3: si el intento 2 también fue OOM -> batch_size=16.
  Cualquier fallo NO-OOM (excepción, handshake, etc.) se reintenta UNA vez al
  mismo batch_size (podría ser transitorio) y si repite, se abandona esa celda.
  Si los 3 intentos (o el reintento no-OOM) fallan, la celda queda FALLIDA y
  se sigue con la próxima -- nunca se detiene el supervisor por una celda.

Uso:
    .venv/bin/python scripts/fed_isic2019/supervise_cells.py
"""

import re
import subprocess
import sys
import time
from pathlib import Path

CELLS = [
    "fedisic2019_fedavg_r10",
    "fedisic2019_fedprox_r10",
    "fedisic2019_fedadagrad_r10",
    "fedisic2019_fedadam_r10",
    "fedisic2019_fedyogi_r10",
    "fedisic2019_cyclic_r60",
    "fedisic2019_scaffold_r10",
]

ROUND_TIMEOUT_S = 3 * 3600  # tope duro por intento (una celda nunca se come la noche entera)
STALL_TIMEOUT_S = 25 * 60  # sin crecimiento del log en 25min -> se considera atascada
POLL_S = 20
BATCH_SIZE_LADDER = [None, 32, 16]  # None = no tocar (64, el del paper)

REPO = Path(__file__).resolve().parents[2]
SUMMARY_PATH = REPO / "runs" / "federado" / "SUPERVISOR_SUMMARY.md"


def _run_cell_once(exp: str, batch_size: int | None, attempt_log: Path) -> tuple[int | None, bool]:
    """Lanza `run_cell.py` para `exp`, vigilando atascos por tamaño del log.

    Devuelve (returncode, stalled). returncode None si se mató por atasco o
    por exceder ROUND_TIMEOUT_S (ambos cuentan como fallo del intento).
    """
    cmd = [
        sys.executable,
        "scripts/fed_isic2019/run_cell.py",
        "--exp",
        exp,
        "--timeout",
        str(ROUND_TIMEOUT_S),
    ]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]

    attempt_log.parent.mkdir(parents=True, exist_ok=True)
    with open(attempt_log, "w") as f:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)

    start = time.time()
    last_size = -1
    last_growth = start
    while True:
        ret = proc.poll()
        if ret is not None:
            return ret, False
        now = time.time()
        if now - start > ROUND_TIMEOUT_S:
            proc.kill()
            return None, False
        size = attempt_log.stat().st_size if attempt_log.exists() else 0
        if size != last_size:
            last_size = size
            last_growth = now
        elif now - last_growth > STALL_TIMEOUT_S:
            print(f"[supervisor] {exp}: atascada (sin progreso en {STALL_TIMEOUT_S}s) -- matando", flush=True)
            proc.kill()
            # Limpieza extra: procesos server/cliente huérfanos de esta celda.
            subprocess.run(["pkill", "-9", "-f", f"federated.server --config configs/federated/{exp}/"])
            subprocess.run(["pkill", "-9", "-f", f"federated.client --config configs/federated/{exp}/"])
            return None, True
        time.sleep(POLL_S)


def _was_oom(attempt_log: Path) -> bool:
    if not attempt_log.exists():
        return False
    return "OutOfMemoryError" in attempt_log.read_text(errors="ignore")


def run_cell_with_ladder(exp: str) -> dict:
    """Corre una celda con la escalera de reintentos. Devuelve un resumen dict."""
    base_dir = REPO / "runs" / "federado" / exp
    attempts: list[dict] = []
    batch_idx = 0
    non_oom_retried = False

    while True:
        batch_size = BATCH_SIZE_LADDER[batch_idx]
        n = len(attempts) + 1
        attempt_log = base_dir / f"supervisor_attempt{n}.log"
        label = f"batch_size={batch_size or 64}"
        print(f"[supervisor] {exp} intento {n} ({label}) ...", flush=True)
        t0 = time.time()
        ret, stalled = _run_cell_once(exp, batch_size, attempt_log)
        dt = time.time() - t0
        oom = _was_oom(attempt_log)
        attempts.append(
            {"n": n, "batch_size": batch_size or 64, "returncode": ret, "stalled": stalled, "oom": oom, "seconds": dt}
        )
        print(
            f"[supervisor] {exp} intento {n}: returncode={ret} stalled={stalled} oom={oom} ({dt / 60:.1f} min)",
            flush=True,
        )

        if ret == 0:
            return {"exp": exp, "status": "OK", "attempts": attempts}

        if oom and batch_idx + 1 < len(BATCH_SIZE_LADDER):
            batch_idx += 1
            continue  # siguiente escalón de batch_size

        if not oom and not stalled and not non_oom_retried:
            # Fallo no-OOM, no-atasco: podría ser transitorio -- un reintento
            # al mismo batch_size antes de rendirse.
            non_oom_retried = True
            continue

        return {"exp": exp, "status": "FAILED", "attempts": attempts}


def write_summary(results: list[dict]) -> None:
    lines = ["# Resumen supervisor Fed-ISIC2019 (overnight)\n"]
    lines.append(f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for r in results:
        lines.append(f"\n## {r['exp']} -- {r['status']}\n")
        for a in r["attempts"]:
            flag = " <- OOM, escaló batch_size" if a["oom"] else (" <- atascada" if a["stalled"] else "")
            lines.append(
                f"- intento {a['n']}: batch_size={a['batch_size']}, "
                f"returncode={a['returncode']}, {a['seconds'] / 60:.1f} min{flag}\n"
            )
        if r["status"] == "OK":
            last = r["attempts"][-1]
            if last["batch_size"] != 64:
                lines.append(
                    f"  **OJO**: completó con batch_size={last['batch_size']}, no el 64 del paper "
                    "(desviación por OOM de hardware, no del paper) -- ver docs/FIXES_FEDISIC2019_FLAMBY_REPRO.md.\n"
                )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("".join(lines))
    print(f"[supervisor] resumen escrito en {SUMMARY_PATH}", flush=True)


def main() -> None:
    results = []
    for exp in CELLS:
        try:
            result = run_cell_with_ladder(exp)
        except Exception as e:  # nunca tumbar el supervisor por una celda
            result = {"exp": exp, "status": f"EXCEPTION: {e!r}", "attempts": []}
        results.append(result)
        write_summary(results)  # se re-escribe tras cada celda, no solo al final
    print("[supervisor] TODAS LAS CELDAS PROCESADAS.", flush=True)


if __name__ == "__main__":
    main()
