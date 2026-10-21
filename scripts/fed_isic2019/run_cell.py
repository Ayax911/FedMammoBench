"""Lanza una celda federada (servidor + 6 clientes) sobre datos reales.

Uso:
    .venv/bin/python scripts/fed_isic2019/run_cell.py --exp fedisic2019_fedavg_r10 --rounds 2
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import yaml

CLIENTS = ["bcn", "vidir_molemax", "vidir_modern", "rosendahl", "msk", "vienna_dias"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="Nombre de la celda (ej. fedisic2019_fedavg_r10)")
    parser.add_argument("--rounds", type=int, default=None, help="Sobrescribir número de rondas")
    parser.add_argument("--local-updates", type=int, default=None, help="Sobrescribir local_updates")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Sobrescribir data.batch_size en los 6 node_*.yaml (escalón de "
            "contingencia por OOM, ver docs/FIXES_FEDISIC2019_FLAMBY_REPRO.md "
            "-- NO es el default del paper, documentar cuando se use)."
        ),
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout en segundos")
    args = parser.parse_args()

    cfg_dir = Path("configs/federated") / args.exp
    server_yaml = cfg_dir / "server.yaml"
    if not server_yaml.exists():
        sys.exit(f"No existe {server_yaml}")

    # Guardar backup de server.yaml si modificamos rondas o local_updates
    orig_content = server_yaml.read_text()
    cfg_data = yaml.safe_load(orig_content)
    modified = False

    if args.rounds is not None and cfg_data["federation"]["rounds"] != args.rounds:
        cfg_data["federation"]["rounds"] = args.rounds
        modified = True
    if args.local_updates is not None and cfg_data["federation"].get("local_updates") != args.local_updates:
        cfg_data["federation"]["local_updates"] = args.local_updates
        modified = True

    # Mismo patrón (backup + restore en finally) para batch_size en los 6
    # node_*.yaml -- por defecto no se toca nada (args.batch_size is None).
    node_backups: dict[Path, str] = {}
    if args.batch_size is not None:
        for client in CLIENTS:
            node_yaml = cfg_dir / f"node_{client}.yaml"
            orig = node_yaml.read_text()
            node_data = yaml.safe_load(orig)
            if node_data["data"]["batch_size"] != args.batch_size:
                node_backups[node_yaml] = orig
                node_data["data"]["batch_size"] = args.batch_size
                node_yaml.write_text(yaml.dump(node_data, sort_keys=False))

    try:
        if modified:
            server_yaml.write_text(yaml.dump(cfg_data, sort_keys=False))

        print(f"=== Iniciando celda {args.exp} (rounds={cfg_data['federation']['rounds']}, updates={cfg_data['federation'].get('local_updates')}) ===")
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        # Debe fijarse ANTES de que el proceso importe torch/inicialice CUDA --
        # por eso va acá (env del subprocess), no dentro de client.py. Reduce
        # fragmentación del allocator de CUDA cuando varios procesos comparten
        # una GPU chica (confirmado necesario: con 6 nodos + gpu_lock_path
        # serializando el cómputo, un solo proceso activo igual mostraba ~5GB
        # "en uso" por fragmentación, no por el modelo en sí -- ver
        # docs/FIXES_FEDISIC2019_FLAMBY_REPRO.md).
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Lanzar servidor
        srv_cmd = [sys.executable, "-m", "src.federated.server", "--config", str(server_yaml)]
        srv_proc = subprocess.Popen(srv_cmd, env=env)
        time.sleep(2.5)

        # Lanzar los 6 nodos
        client_procs = []
        for client in CLIENTS:
            node_yaml = cfg_dir / f"node_{client}.yaml"
            cmd = [sys.executable, "-m", "src.federated.client", "--config", str(node_yaml)]
            proc = subprocess.Popen(cmd, env=env)
            client_procs.append((client, proc))

        # Esperar servidor
        t0 = time.time()
        srv_ret = None
        while time.time() - t0 < args.timeout:
            srv_ret = srv_proc.poll()
            if srv_ret is not None:
                break
            time.sleep(2.0)

        if srv_ret is None:
            print("TIMEOUT alcanzado en servidor. Matando procesos...")
            srv_proc.kill()
            for _, cp in client_procs:
                cp.kill()
            sys.exit(1)

        print(f"Servidor terminó con returncode: {srv_ret}")
        if srv_ret != 0:
            print("El servidor falló.")
            for _, cp in client_procs:
                cp.kill()
            sys.exit(srv_ret)

        # Esperar que los clientes terminen (suelen terminar inmediatamente tras el servidor)
        for name, cp in client_procs:
            c_ret = cp.wait(timeout=60)
            print(f"Cliente {name} terminó con returncode: {c_ret}")

        # Mostrar métricas del servidor
        server_log = Path("runs/federado") / args.exp / "server" / "server.log"
        if server_log.exists():
            print("\n--- server.log ---")
            print(server_log.read_text())

    finally:
        if modified:
            server_yaml.write_text(orig_content)
        for node_yaml, orig in node_backups.items():
            node_yaml.write_text(orig)


if __name__ == "__main__":
    main()
