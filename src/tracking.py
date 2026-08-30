"""Persistencia de métricas de entrenamiento — un registro en CSV y en
TensorBoard por cada época, para poder graficar/comparar corridas después de
que el proceso termine. Misma idea que runs/<RUN_NAME>/loss_history.csv en
la serie de notebooks, pero como código reusable en vez de celdas copiadas."""

import csv
from pathlib import Path
from types import TracebackType

from torch.utils.tensorboard import SummaryWriter  # pyright: ignore[reportMissingTypeStubs]


class MetricsLogger:
    """Escribe las métricas de cada época a metrics.csv y a TensorBoard.

    No calcula nada y no decide qué es "mejor" — eso ya lo hacen evaluate()
    y Trainer. Esta clase solo persiste lo que se le pasa.
    """

    def __init__(self, run_dir: str | Path) -> None:
        """Prepara el destino de metrics.csv y el writer de TensorBoard.

        Args:
            run_dir: carpeta destino — se crea si no existe. metrics.csv y
                los eventos de TensorBoard quedan ambos dentro de ella.
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("w", newline="")
        # None hasta el primer log() — recién ahí se conocen las columnas
        # (los nombres de métrica que Trainer decida pasar).
        self._csv_writer: csv.DictWriter[str] | None = None

        self._tb_writer = SummaryWriter(log_dir=str(self.run_dir))

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

    def close(self) -> None:
        """Cierra el archivo CSV y el SummaryWriter.

        Llamar al final de fit(), incluso si algo falló a mitad de
        entrenamiento — por eso también existe __exit__.
        """
        self._csv_file.close()
        self._tb_writer.close()  # pyright: ignore[reportUnknownMemberType]

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
