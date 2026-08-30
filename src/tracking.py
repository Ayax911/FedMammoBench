"""Persistencia de métricas de entrenamiento — un registro en CSV y en
TensorBoard por cada época, para poder graficar/comparar corridas después de
que el proceso termine. Misma idea que runs/<RUN_NAME>/loss_history.csv en
la serie de notebooks, pero como código reusable en vez de celdas copiadas.

W&B es un tercer destino opcional, no un backend separado (Strategy) --
CSV y TensorBoard ya conviven como dos ramas dentro de esta misma clase, así
que W&B suma como una tercera rama, no como una jerarquía nueva. Se activa
solo si se pasa wandb_project; si no, `wandb` ni se importa. Reproduce el
mismo "degradar a offline sin bloquear nunca" que ya usan las celdas de W&B
de los notebooks (ver exp23), pero automático en vez de una verificación
manual antes de correr."""

import csv
from pathlib import Path
from types import TracebackType

from torch.utils.tensorboard import SummaryWriter  # pyright: ignore[reportMissingTypeStubs]


def _wandb_credentials_cached() -> bool:
    """Espeja el chequeo ya documentado en CLAUDE.md (`grep -q "api.wandb.ai"
    ~/.netrc`) pero como código: nunca abre el archivo con otro fin que
    confirmar que la entrada existe, nunca imprime ni devuelve su contenido.

    Returns:
        bool: True si hay una entrada de api.wandb.ai en ~/.netrc (sesión de
            `wandb login` ya cacheada), False si no hay ~/.netrc o no la tiene.
    """
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.is_file():
        return False
    return "api.wandb.ai" in netrc_path.read_text()


class MetricsLogger:
    """Escribe las métricas de cada época a metrics.csv, TensorBoard, y
    opcionalmente W&B.

    No calcula nada y no decide qué es "mejor" — eso ya lo hacen evaluate()
    y Trainer. Esta clase solo persiste lo que se le pasa.
    """

    def __init__(
        self,
        run_dir: str | Path,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        """Prepara el destino de metrics.csv, el writer de TensorBoard, y
        (si se pide) una corrida de W&B.

        Args:
            run_dir: carpeta destino — se crea si no existe. metrics.csv y
                los eventos de TensorBoard quedan ambos dentro de ella.
            wandb_project: nombre del proyecto de W&B. None (default) ->
                W&B queda completamente desactivado, `wandb` ni se importa.
            wandb_run_name: nombre de esta corrida en W&B. Ignorado si
                wandb_project es None.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("w", newline="")
        # None hasta el primer log() — recién ahí se conocen las columnas
        # (los nombres de métrica que Trainer decida pasar).
        self._csv_writer: csv.DictWriter[str] | None = None

        self._tb_writer = SummaryWriter(log_dir=str(self.run_dir))

        self._wandb_run = None
        if wandb_project is not None:
            import wandb  # import perezoso -- nadie que no pida W&B lo paga

            mode = "online" if _wandb_credentials_cached() else "offline"
            self._wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                dir=str(self.run_dir),
                mode=mode,
            )

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        """Escribe una fila de CSV y un scalar de TensorBoard por cada
        entrada de metrics, para esta época.

        Args:
            epoch: número de época — columna "epoch" del CSV y global_step
                de TensorBoard.
            metrics: nombre de métrica -> valor, ya combinando train y val
                en un solo dict (ej. {"train_loss": ..., "val_loss": ...,
                "val_accuracy": ...}). Esta clase no distingue splits, solo
                persiste lo que recibe — la combinación es responsabilidad
                de quien llama (Trainer.fit()).

        Raises:
            ValueError: si metrics no trae exactamente las mismas claves
                que la primera llamada a log() (csv.DictWriter exige
                columnas consistentes entre filas).
        """
        row: dict[str, float | int] = {"epoch": epoch, **metrics}

        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        for name, value in metrics.items():
            self._tb_writer.add_scalar(name, value, epoch)  # pyright: ignore[reportUnknownMemberType]

        if self._wandb_run is not None:
            self._wandb_run.log(metrics, step=epoch)

    def close(self) -> None:
        """Cierra el archivo CSV, el SummaryWriter, y la corrida de W&B.

        Llamar al final de fit(), incluso si algo falló a mitad de
        entrenamiento — por eso también existe __exit__.
        """
        self._csv_file.close()
        self._tb_writer.close()  # pyright: ignore[reportUnknownMemberType]
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
