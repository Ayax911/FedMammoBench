"""Registro consolidado config+resultados de TODOS los experimentos, en un `.xlsx`.

Un solo archivo (`experiment_registry.xlsx` en la raíz del repo, por
default) con una fila por experimento -- toda la sección del YAML
aplanada en columnas `cfg.*`, más las métricas de `val/metrics.json` y
`test/metrics.json` de su `run_dir` -- para poder ordenar/filtrar en una
hoja de cálculo sin abrir 40 YAML ni 40 JSON a mano.

**Se regenera ENTERO cada vez, nunca fila por fila.** No hay lógica de
"buscar la fila de expNN y actualizarla": este módulo relee `configs/*.yaml`
y `runs/<experiment_id>/{val,test}/metrics.json` desde cero en cada llamada.
Es deliberado, no la opción perezosa -- una actualización incremental podría
dejar una fila vieja si un experimento cambia de `run_dir` o si dos configs
alguna vez comparten `experiment_id` a mitad de una migración; releer todo
cuesta menos de un segundo (~50 YAML/JSON chicos) y garantiza que la fila
sea siempre exactamente lo que hay en disco ahora, igual que
`evaluate_split()` relee el mejor checkpoint en vez de confiar en el estado
del modelo en memoria (ver su nota en `src/eval_pipeline.py`).

Cubre las dos familias de experimentos, en hojas separadas porque su config
y sus resultados tienen formas distintas:

  - `centralizados` / `centralizados_por_bd`: un YAML por experimento
    (`configs/*.yaml`), resultados en `run_dir/{val,test}/metrics.json` +
    desglose opcional en `run_dir/test/metrics_by_database.json`.
  - `federados` / `federados_por_nodo`: un `server.yaml` por experimento
    (`configs/federated/<exp>/server.yaml`), resultado agregado en
    `run_dir/best.json` (métrica ponderada federada -- NO el AUC pooled,
    ver CUIDADO en `.claude/context/experiments/FEDERATED.md`) + métricas
    locales por nodo en `run_dir/../nodes/<db>/{val,test}/metrics.json`.

Uso programático (lo que llama `src/cli.py:run()` al final de cada corrida
centralizada -- ver su comentario):

    >>> from src.registry import write_registry
    >>> write_registry()

Uso manual, para refrescar también las hojas federadas (nada las dispara
automáticamente hoy -- ver docstring de `scripts/build_experiment_registry.py`
para el porqué) o para apuntar a otra ruta:

    .venv/bin/python -m scripts.build_experiment_registry
    .venv/bin/python -m scripts.build_experiment_registry --output otro_registro.xlsx
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Raíz del repo -- este archivo vive en src/, así que un nivel arriba.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
FEDERATED_CONFIGS_DIR = CONFIGS_DIR / "federated"
DEFAULT_REGISTRY_PATH = REPO_ROOT / "experiment_registry.xlsx"

# Mismas 8 claves "planas" que trae tanto test/metrics.json como cada
# entrada de test/metrics_by_database.json (que además trae las cm_*
# derivadas de la matriz de confusión -- se dejan fuera acá para no duplicar
# columnas: accuracy/auc/f1/... y cm_accuracy/cm_auc/... miden lo mismo por
# dos caminos distintos, ver evaluate_split() en src/eval_pipeline.py).
METRIC_KEYS: list[str] = [
    "accuracy",
    "auc",
    "f1",
    "f1_macro",
    "loss",
    "precision",
    "sensitivity",
    "specificity",
]

# Bloque narrativo de cada experimento -- a qué pregunta/barrido pertenece,
# NO un dato que viva en el YAML (los encabezados son prosa libre, no un
# campo `block:`). Fuente: los agrupamientos ya documentados en
# `.claude/context/experiments/{CENTRALIZED,FEDERATED}.md` ("Ablación de
# pretraining (exp17–23, 32, 33)", "Anti-sobreajuste (exp28–32)", "Grid
# principal: estrategia × rondas (exp41–52)", etc.), NO un heurístico
# inventado acá.
#
# Match por SUBSTRING sobre `experiment_id` en minúsculas -- no por rango
# numérico -- porque la convención real del repo es nombrar cada YAML
# `expNN_<slug-descriptivo>` y ese slug ya lleva el nombre del bloque (ver
# CONFIG.md); así un experimento nuevo que seed la convención (`exp61_
# antioverfit_algo.yaml`) cae solo en su bloque sin tocar esta tabla, algo
# que un rango `exp28<=n<=32` no puede hacer. Los pocos casos que MUEREN sin
# heurística (exp01/exp02, pre-pipeline; `exp_example_01`, plantilla) llevan
# regla explícita.
#
# Orden IMPORTA -- de más a menos específico, primer match gana (ver el
# "exp32_antioverfit_radimagenet" doblemente narrado en CENTRALIZED.md como
# antioverfit Y como ablación de pretraining -- se prioriza "antioverfit"
# por ser el nombre real del archivo/experiment_id).
#
# Mantenimiento: un experimento nuevo que no calce ningún patrón cae en
# "otros" (visible en la columna, no un bloque existente equivocado) --
# agregar su regla acá junto con la entrada nueva en CENTRALIZED.md/
# FEDERATED.md, no antes.
_BLOCK_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"fedgrid.*_mu\d"), "federado_grid_proximal_mu"),       # exp53-56
    (re.compile(r"fedgrid"), "federado_grid_estrategia_rondas"),        # exp41-52
    (re.compile(r"fedavg_full"), "federado_baseline"),                  # exp40
    (re.compile(r"antioverfit"), "antioverfit"),                        # exp28-32, 34-36
    (re.compile(r"bydatabase"), "bydatabase_individual"),                # exp24-27
    (re.compile(r"hpsearch"), "hpsearch_v1"),                            # exp37-39
    (re.compile(r"pretrain_ablation|pretrain_imagenet|pretrain_radimagenet"), "pretrain_ablation"),  # exp17-23,33,58-60
    (re.compile(r"fedmammobench_full"), "full_dataset_baseline"),        # exp05-06
    (re.compile(r"camilo"), "inc_replica"),                              # exp57
    (re.compile(r"inc_strict_replica|frozen_backbone"), "piloto_manifest_inc"),  # exp03-04
    (re.compile(r"resnet_layer4_test|classification_images_hl"), "piloto_humo"),  # exp01-02
    (re.compile(r"^exp0[7-9]_|^exp1[0-6]_|fullfreeze_bce|layer4_bce"), "head_freeze_sweep"),  # exp07-16
    (re.compile(r"^exp_example"), "plantilla"),
]


def classify_block(experiment_id: str) -> str:
    """Bloque narrativo de un experimento a partir de su `experiment_id` -- ver `_BLOCK_RULES`."""
    lowered = experiment_id.lower()
    for pattern, block in _BLOCK_RULES:
        if pattern.search(lowered):
            return block
    return "otros"


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplana un dict anidado (un YAML ya parseado) a columnas `a.b.c`.

    Listas se serializan a JSON (`[1024]` -> `"[1024]"`) en vez de
    explotarlas en columnas `a.b.0`/`a.b.1` -- `head.hparams.hidden_layers`
    varía en longitud entre experimentos (`[1024]` vs `[1024, 512]` vs
    `[1024, 512, 256]`), y una columna por índice dejaría de alinear el
    mismo hiperparámetro en la misma columna entre filas.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, prefix=f"{flat_key}."))
        elif isinstance(value, list):
            flat[flat_key] = json.dumps(value) if value else None
        else:
            flat[flat_key] = value
    return flat


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metric_row(metrics: dict[str, Any] | None) -> dict[str, float | None]:
    """Subconjunto de METRIC_KEYS, redondeado a 4 decimales para lectura en la hoja.

    Los YAML de config guardan sus números tal cual (esos SÍ importan al
    dígito, ej. `weight: [0.7593, 1.4642]`); los resultados solo se
    redondean para mostrar, nunca se usan para recalcular nada acá.
    """
    if metrics is None:
        return {key: None for key in METRIC_KEYS}
    return {key: (round(metrics[key], 4) if key in metrics else None) for key in METRIC_KEYS}


def _status(run_dir: Path, val_metrics: dict | None, test_metrics: dict | None) -> str:
    if not run_dir.exists():
        return "sin_correr"
    if val_metrics is None or test_metrics is None:
        return "incompleto"
    return "completo"


def build_centralized_tables(configs_dir: Path = CONFIGS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Escanea `configs/*.yaml` (planos, no `configs/federated/`) y arma las dos tablas centralizadas.

    Returns:
        (tabla principal -- una fila por experiment_id,
         tabla larga por base de datos -- una fila por (experiment_id, database),
         a partir de `test/metrics_by_database.json`, vacía si ningún
         experimento tiene desglose).
    """
    rows: list[dict[str, Any]] = []
    by_database_rows: list[dict[str, Any]] = []

    for config_path in sorted(configs_dir.glob("*.yaml")):
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not raw or "experiment_id" not in raw:
            continue  # no es un YAML de experimento (ej. un template sin experiment_id)

        experiment_id = raw["experiment_id"]
        run_dir = REPO_ROOT / raw.get("train", {}).get("run_dir", f"runs/{experiment_id}")

        val_metrics = _read_json(run_dir / "val" / "metrics.json")
        test_metrics = _read_json(run_dir / "test" / "metrics.json")
        by_database = _read_json(run_dir / "test" / "metrics_by_database.json") or {}

        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "bloque": classify_block(experiment_id),
            "config_file": str(config_path.relative_to(REPO_ROOT)),
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "status": _status(run_dir, val_metrics, test_metrics),
        }
        row.update({f"val_{k}": v for k, v in _metric_row(val_metrics).items()})
        row.update({f"test_{k}": v for k, v in _metric_row(test_metrics).items()})
        row.update({f"cfg.{k}": v for k, v in _flatten(raw).items() if k != "experiment_id"})
        rows.append(row)

        for database, metrics in by_database.items():
            by_database_rows.append({
                "experiment_id": experiment_id,
                "bloque": classify_block(experiment_id),
                "database": database,
                **_metric_row(metrics),
            })

    identity_cols = ["experiment_id", "bloque", "config_file", "run_dir", "status"]
    val_cols = [f"val_{k}" for k in METRIC_KEYS]
    test_cols = [f"test_{k}" for k in METRIC_KEYS]
    cfg_cols = sorted({k for row in rows for k in row if k.startswith("cfg.")})
    main_df = pd.DataFrame(rows).reindex(columns=identity_cols + val_cols + test_cols + cfg_cols)

    by_database_cols = ["experiment_id", "bloque", "database"] + METRIC_KEYS
    by_database_df = pd.DataFrame(by_database_rows).reindex(columns=by_database_cols)

    return main_df, by_database_df


def build_federated_tables(
    federated_configs_dir: Path = FEDERATED_CONFIGS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Escanea `configs/federated/<exp>/server.yaml` y arma las dos tablas federadas.

    Cada nodo se descubre a partir de sus `node_<db>.yaml` en la misma
    carpeta del experimento (ej. `node_cmmd.yaml` -> nodo `cmmd`), que es la
    misma convención de nombres que usan los `run_dir` de los nodos
    (`<run_dir_servidor>/../nodes/<db>/`, ver `docker-compose.federated.yaml`).

    Returns:
        (tabla principal -- una fila por experiment_id, con la métrica
         agregada de `best.json` y el PROMEDIO simple entre nodos de sus
         métricas locales; tabla larga -- una fila por (experiment_id, node,
         split), sin promediar, para ver la dispersión entre nodos que el
         promedio esconde).
    """
    rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []

    if not federated_configs_dir.exists():
        empty_main = pd.DataFrame(columns=["experiment_id", "bloque", "config_file", "run_dir", "status"])
        empty_nodes = pd.DataFrame(columns=["experiment_id", "bloque", "node", "split", *METRIC_KEYS])
        return empty_main, empty_nodes

    for exp_dir in sorted(federated_configs_dir.iterdir()):
        server_yaml = exp_dir / "server.yaml"
        if not server_yaml.exists():
            continue

        with open(server_yaml, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        experiment_id = raw["experiment_id"]
        run_dir = REPO_ROOT / raw.get("tracking", {}).get("run_dir", f"runs/{experiment_id}/server")
        nodes_dir = run_dir.parent / "nodes"

        best = _read_json(run_dir / "best.json")

        node_ids = sorted(p.stem.removeprefix("node_") for p in exp_dir.glob("node_*.yaml"))
        node_val_metrics: list[dict] = []
        node_test_metrics: list[dict] = []
        for node_id in node_ids:
            for split, bucket in (("val", node_val_metrics), ("test", node_test_metrics)):
                metrics = _read_json(nodes_dir / node_id / split / "metrics.json")
                if metrics is not None:
                    bucket.append(metrics)
                node_rows.append({
                    "experiment_id": experiment_id,
                    "bloque": classify_block(experiment_id),
                    "node": node_id,
                    "split": split,
                    **_metric_row(metrics),
                })

        row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "bloque": classify_block(experiment_id),
            "config_file": str(server_yaml.relative_to(REPO_ROOT)),
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "status": "sin_correr" if best is None else "completo",
            "best_round": best.get("best_round") if best else None,
            "best_metric_name": best.get("metric_name") if best else None,
            "best_metric_value": (round(best["metric_value"], 4) if best else None),
        }
        for split, bucket in (("val", node_val_metrics), ("test", node_test_metrics)):
            for key in METRIC_KEYS:
                values = [m[key] for m in bucket if key in m]
                row[f"mean_{split}_{key}"] = round(sum(values) / len(values), 4) if values else None
        row.update({f"cfg.{k}": v for k, v in _flatten(raw).items() if k != "experiment_id"})
        rows.append(row)

    identity_cols = [
        "experiment_id", "bloque", "config_file", "run_dir", "status",
        "best_round", "best_metric_name", "best_metric_value",
    ]
    mean_cols = [f"mean_{split}_{k}" for split in ("val", "test") for k in METRIC_KEYS]
    cfg_cols = sorted({k for row in rows for k in row if k.startswith("cfg.")})
    main_df = pd.DataFrame(rows).reindex(columns=identity_cols + mean_cols + cfg_cols)

    node_cols = ["experiment_id", "bloque", "node", "split"] + METRIC_KEYS
    node_df = pd.DataFrame(node_rows).reindex(columns=node_cols)

    return main_df, node_df


def write_registry(
    output_path: Path = DEFAULT_REGISTRY_PATH,
    configs_dir: Path = CONFIGS_DIR,
    federated_configs_dir: Path = FEDERATED_CONFIGS_DIR,
) -> Path:
    """Regenera `output_path` entero con las 4 hojas. Devuelve la ruta escrita.

    Sin argumentos hace exactamente lo que necesita `src/cli.py:run()` al
    final de cada corrida centralizada: relee TODO `configs/` (no solo el
    experimento que acaba de correr) para que la hoja quede consistente con
    disco sin importar qué otro experimento haya cambiado de estado
    mientras tanto.
    """
    centralized_main, centralized_by_database = build_centralized_tables(configs_dir)
    federated_main, federated_by_node = build_federated_tables(federated_configs_dir)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        centralized_main.to_excel(writer, sheet_name="centralizados", index=False)
        centralized_by_database.to_excel(writer, sheet_name="centralizados_por_bd", index=False)
        federated_main.to_excel(writer, sheet_name="federados", index=False)
        federated_by_node.to_excel(writer, sheet_name="federados_por_nodo", index=False)

    _format_workbook(output_path)
    return output_path


def _format_workbook(path: Path) -> None:
    """Encabezado en negrita, panel congelado bajo la fila 1 y columnas autoajustadas.

    Puramente cosmético -- si algo acá fallara no debería tirar abajo
    `write_registry()`, pero `openpyxl` sobre un archivo que este mismo
    proceso acaba de escribir no tiene forma realista de fallar, así que no
    se envuelve en try/except (ver CLAUDE.md sobre no manejar casos que no
    pueden pasar).
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Font

    workbook = load_workbook(path)
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for column_cells in sheet.columns:
            length = max((len(str(cell.value)) for cell in column_cells if cell.value is not None), default=0)
            sheet.column_dimensions[column_cells[0].column_letter].width = min(max(length + 2, 10), 60)
    workbook.save(path)
