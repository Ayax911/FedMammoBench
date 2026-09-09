"""Shim entre un agente de W&B Sweeps y `src.cli` -- un trial del sweep por invocación.

`src/cli.py` acepta exactamente un argumento, `--config <yaml>`, y no tiene
forma de recibir overrides sueltos; un agente de W&B, al revés, entrega los
hiperparámetros muestreados como argumentos de línea de comandos
(`--lr=0.0007 --weight_decay=0.003 ...`, vía `${args}` en la definición del
sweep). Este script es el traductor entre ambos: toma un YAML base, le aplica
los overrides del trial, materializa el config resultante en disco, y lanza
`python -m src.cli --config <ese archivo>` como SUBPROCESO.

Por qué subproceso y no `from src.cli import run` (las tres razones, en orden
de importancia):

  1. CLAUDE.md fija como regla dura que NADA importa `cli.py` -- es la punta
     de la cadena de dependencias, no una biblioteca. Importarlo desde acá
     rompería esa invariante por conveniencia.
  2. Proceso limpio por trial: sin memoria de CUDA arrastrada entre corridas,
     sin `set_global_seed()` de un trial contaminando al siguiente, sin estado
     de W&B a medio cerrar. Con 80-100 trials seguidos eso deja de ser
     teórico.
  3. Un trial que crashea (OOM con batch_size grande, por ejemplo) mata al
     subproceso, no al agente -- el sweep sigue con el siguiente.

Las variables de entorno que el agente exporta (`WANDB_SWEEP_ID`,
`WANDB_RUN_ID`) se heredan al subproceso, así que el `wandb.init()` que
`MetricsLogger` hace ADENTRO de `src.cli` (ver `src/tracking.py`) adjunta la
corrida al sweep correctamente. Por eso este script no llama a `wandb.init()`
ni importa `wandb`: hacerlo abriría una segunda corrida en el mismo proceso y
competiría con la que abre `MetricsLogger`.

Artefactos -- deliberadamente FUERA de `configs/` y `runs/`:

    sweeps/<sweep_id>/configs/trial_<run_id>.yaml   config materializada del trial
    sweeps/<sweep_id>/runs/trial_<run_id>/          run_dir del trial

`configs/` son YAML escritos a mano que se leen de punta a punta, sin herencia
(decisión de proyecto, ver CLAUDE.md), y `runs/` es el registro de resultados
commiteado. Cien trials generados a máquina no pertenecen a ninguno de los dos;
`sweeps/` está en `.gitignore`. Cuando el sweep termina, el config ganador se
promueve A MANO a un `configs/expNN_*.yaml` permanente con su header
explicando de qué sweep salió.

Uso -- normalmente no se invoca a mano, lo llama el agente (ver
`sweeps/hpsearch_v1.yaml`):

    wandb sweep sweeps/hpsearch_v1.yaml
    wandb agent <entity>/fedmammobench2.0/<sweep_id>

Para inspeccionar qué config produciría un conjunto de overrides, sin entrenar
(no necesita ni GPU ni datos ni credenciales de W&B):

    .venv/bin/python -m scripts.sweep_train \
        --base_config configs/exp31_antioverfit_no_inputdrop.yaml --sweep_name hpsearch_v1 \
        --lr=0.0005 --weight_decay=0.01 --epochs=40 --dry_run
"""

import argparse
import os
import subprocess
import sys
import uuid
from pathlib import Path

from src.config import ExperimentConfig, load_config, save_config

SWEEPS_ROOT = Path("sweeps")


def parse_hidden_layers(raw: str) -> list[int]:
    """Traduce la codificación de `hidden_layers` que viaja por línea de comandos.

    `hidden_layers` es el único hiperparámetro del espacio de búsqueda cuyo
    valor es una lista, y `${args}` de W&B lo serializaría como `[512]` --
    una representación que hay que parsear y que se rompe distinto según la
    versión. Se codifica como string separado por comas para que el viaje por
    CLI sea inequívoco: `""` (vacío) -> `[]`, `"512"` -> `[512]`,
    `"1024,512"` -> `[1024, 512]`.

    Args:
        raw: el string tal como llegó del agente.

    Returns:
        list[int]: tamaños de las capas ocultas, en orden.

    Raises:
        ValueError: algún elemento no es un entero.

    Example:
        >>> parse_hidden_layers("1024,512")
        [1024, 512]
        >>> parse_hidden_layers("")
        []
    """
    stripped = raw.strip()
    if not stripped:
        return []
    return [int(part) for part in stripped.split(",")]


def build_trial_config(
    base_config_path: Path,
    args: argparse.Namespace,
    sweep_id: str,
    run_id: str,
) -> ExperimentConfig:
    """Aplica los overrides del trial sobre el YAML base y devuelve el config resultante.

    Solo se tocan los campos que el sweep muestrea; todo lo demás (arquitectura,
    manifest, normalización, augmentación, pesos de clase de la loss) queda
    exactamente como lo dejó el YAML base. Un override que llegue como `None`
    -- porque el sweep no muestrea esa dimensión -- no modifica nada, así que
    este mismo shim sirve para un sweep de 9 dimensiones o de 2.

    Args:
        base_config_path: YAML base, típicamente `configs/exp31_*.yaml`.
        args: namespace de `parse_args()`, con los overrides muestreados.
        sweep_id: id del sweep (de `WANDB_SWEEP_ID`), para la ruta de artefactos.
        run_id: id de esta corrida (de `WANDB_RUN_ID`), para la ruta y el
            `experiment_id`.

    Returns:
        ExperimentConfig: el config del trial, ya validado por Pydantic.
    """
    config = load_config(base_config_path)

    # --- Optimizer ---
    if args.lr is not None:
        config.optimizer.hparams["lr"] = args.lr
    if args.backbone_lr is not None:
        config.optimizer.hparams["backbone_lr"] = args.backbone_lr
    if args.weight_decay is not None:
        config.optimizer.hparams["weight_decay"] = args.weight_decay
    if args.beta1 is not None:
        # betas viaja como par; solo beta1 se busca -- beta2 en 0.999 es el
        # default de Adam/AdamW y no hay ninguna evidencia en el proyecto que
        # sugiera moverlo.
        config.optimizer.hparams["betas"] = [args.beta1, 0.999]

    # --- Loss ---
    if args.label_smoothing is not None:
        config.loss.hparams["label_smoothing"] = args.label_smoothing

    # --- Head ---
    if args.input_dropout is not None:
        # input_dropout y NO dropout: ConfigurableMLPHead.build() inserta el
        # Dropout de `dropout` DENTRO del bucle de capas ocultas, así que con
        # hidden_layers=[] es un no-op. Buscar sobre él desperdiciaría una
        # dimensión entera del espacio en los trials sin capas ocultas.
        config.head.hparams["input_dropout"] = args.input_dropout
    if args.hidden_layers is not None:
        config.head.hparams["hidden_layers"] = parse_hidden_layers(args.hidden_layers)

    # --- Data ---
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size

    # --- Train + scheduler (acoplados) ---
    if args.epochs is not None:
        config.train.epochs = args.epochs
        # T_max SIEMPRE se deriva de epochs, nunca se muestrea aparte. El
        # desajuste que arrastran los configs actuales (T_max=200/80 con la
        # mejor época alrededor de la 10, o sea el coseno todavía al ~96% de
        # su LR inicial cuando ya pasó lo importante) es exactamente el bug
        # que un sweep con ambas dimensiones sueltas reproduciría en la mitad
        # de los trials.
        if config.scheduler is not None:
            config.scheduler.hparams["T_max"] = args.epochs

    # --- Identidad y rutas del trial ---
    trial_name = f"trial_{run_id}"
    trial_root = SWEEPS_ROOT / sweep_id
    config.experiment_id = trial_name
    config.train.run_dir = trial_root / "runs" / trial_name
    config.train.checkpoint_dir = trial_root / "runs" / trial_name / "weights"
    # Agrupa los trials del mismo sweep en la UI de W&B (tabla de runs
    # agrupada por columna "Group", el mismo mecanismo que ya usan
    # exp05-36 con nombres como "antioverfit_regularized" -- ver
    # TrainConfig.wandb_group en src/config.py). La pertenencia real al
    # sweep de W&B (pestaña Sweeps, paralelo-coordenadas, importancia de
    # hiperparámetros) la resuelve wandb.init() solo, leyendo
    # WANDB_SWEEP_ID del entorno -- este campo es aparte, para la vista de
    # runs planos.
    #
    # `args.sweep_name` (obligatorio, ver parse_args()) en vez de solo
    # `sweep_id`: sweep_id es el hash opaco que asigna W&B (algo como
    # "a1b2c3d4") -- útil para que dos lanzamientos del mismo diseño de
    # sweep (mismo YAML relanzado) no se mezclen en un solo grupo, pero
    # ilegible por sí solo en la tabla de runs. Prefijarlo con el nombre
    # legible del sweep (ej. "hpsearch_v1", el mismo que su YAML) deja el
    # grupo identificable de un vistazo Y sigue siendo único por
    # lanzamiento.
    config.train.wandb_group = f"{args.sweep_name}_{sweep_id}"

    return config


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos, uno por dimensión del espacio de búsqueda.

    Los nombres llevan guion bajo, no guion medio, para que calcen exactamente
    con los nombres de parámetro de la definición del sweep -- `${args}` los
    expande como `--<nombre_del_parametro>=<valor>` sin traducir nada.
    """
    parser = argparse.ArgumentParser(description="Corre un trial de un sweep de W&B sobre src.cli.")
    parser.add_argument("--base_config", required=True, type=Path, help="YAML base sobre el que se aplican los overrides.")
    parser.add_argument(
        "--sweep_name",
        required=True,
        help=(
            "Nombre legible del sweep (ej. 'hpsearch_v1', el mismo que su archivo YAML) -- "
            "no es una dimensión del espacio de búsqueda, es un argumento fijo del "
            "`command:` de la definición del sweep, igual que --base_config. Se usa para "
            "armar el wandb_group de cada trial: '<sweep_name>_<sweep_id>'."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Materializa el config del trial y NO entrena.")

    parser.add_argument("--lr", type=float, default=None, help="LR de la cabeza.")
    parser.add_argument("--backbone_lr", type=float, default=None, help="LR del backbone (LR discriminativo, ver src/cli.py).")
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None, help="beta1 de AdamW; beta2 queda en 0.999.")
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--input_dropout", type=float, default=None, help="Dropout tras Flatten() en la cabeza.")
    parser.add_argument("--hidden_layers", type=str, default=None, help='Separado por comas: "" | "512" | "1024,512".')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="También fija scheduler.T_max al mismo valor.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Sin las variables del agente (invocación a mano, típicamente --dry_run)
    # se cae a valores locales, para que el script sea ejecutable y
    # verificable sin W&B ni red.
    sweep_id = os.environ.get("WANDB_SWEEP_ID", "local")
    run_id = os.environ.get("WANDB_RUN_ID", uuid.uuid4().hex[:8])

    config = build_trial_config(args.base_config, args, sweep_id, run_id)

    trial_config_path = SWEEPS_ROOT / sweep_id / "configs" / f"trial_{run_id}.yaml"
    save_config(config, trial_config_path)
    print(f"Config del trial: {trial_config_path}")

    if args.dry_run:
        print("--dry_run: no se entrena.")
        return

    # Ver el docstring del módulo para por qué esto es un subproceso y no un
    # import de src.cli.
    completed = subprocess.run(
        [sys.executable, "-m", "src.cli", "--config", str(trial_config_path)],
        check=False,
    )
    # Propagar el código de salida: un trial fallido tiene que verse como
    # fallido para el agente, no como uno que terminó bien sin métricas.
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
