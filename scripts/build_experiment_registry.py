"""CLI manual para regenerar `experiment_registry.xlsx` -- la lógica vive en `src/registry.py`.

`src/cli.py:run()` ya llama a `write_registry()` al final de cada corrida
CENTRALIZADA, así que para ese caso no hace falta correr este script a mano.

Corré esto a mano cuando:
  - Querés refrescar las hojas `federados`/`federados_por_nodo` -- nada las
    dispara automáticamente hoy. A propósito: el proceso servidor
    (`src/federated/server.py:run_server()`) termina antes de que exista el
    AUC pooled real (ver CUIDADO en `.claude/context/experiments/FEDERATED.md`
    -- el agregado ponderado de `best.json` NO es comparable contra un AUC
    centralizado sin ese paso posterior), así que enganchar la regeneración
    ahí congelaría una fila con el número que no hay que citar. Correr este
    script a mano, después de esa re-evaluación, evita esa trampa.
  - Editaste un YAML de `configs/` y querés ver el cambio reflejado sin
    volver a entrenar nada.
  - Querés un `.xlsx` en otra ruta (`--output`), o generarlo por primera vez
    antes de correr cualquier experimento (todas las filas salen con
    status "sin_correr").

Uso:
    .venv/bin/python -m scripts.build_experiment_registry
    .venv/bin/python -m scripts.build_experiment_registry --output /tmp/registro.xlsx
"""

import argparse
from pathlib import Path

from src.registry import DEFAULT_REGISTRY_PATH, write_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenera el registro consolidado config+resultados de todos los experimentos."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help=f"Ruta del .xlsx a escribir (default: {DEFAULT_REGISTRY_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = write_registry(output_path=args.output)
    print(f"Registro escrito en {written}")


if __name__ == "__main__":
    main()
