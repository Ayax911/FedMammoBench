"""Early stopping: decide si la métrica de validación de esta época es una
mejora real, y si hay que detener el entrenamiento por falta de progreso.

Deliberadamente sin I/O — a diferencia del `EarlyStopping` del proyecto INC
(`classification_images/early_stopping.py`), que guarda checkpoints
directamente dentro de `__call__`. Aquí la decisión ("¿mejoró?", "¿hay que
parar?") está separada de la persistencia: `Trainer.fit()` (`train/trainer.py`)
sigue siendo el único lugar que llama a `save_checkpoint()`.

También corrige, como efecto colateral, el bug documentado en REFACTOR.md/
PHASES.md de que `Trainer` "siempre maximiza" `metric_name`: `mode` decide
la dirección de comparación una sola vez, en un solo objeto, en vez de un
`>` hardcodeado en `Trainer.fit()`.

Ejemplo de uso:
    >>> from src.train.early_stopping import EarlyStopping
    >>> stopper = EarlyStopping(patience=50, min_delta=0.005, mode="max")
    >>> improved = stopper.step(val_f1)
    >>> if stopper.should_stop:
    ...     break
"""

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """Rastrea la mejor métrica de validación vista y cuenta épocas sin mejora.

    Attributes:
        patience: épocas consecutivas sin mejora real antes de activar
            `should_stop`. `None` desactiva la parada temprana — el objeto
            sigue rastreando `best_value`/`improved` (para el checkpoint),
            pero `should_stop` nunca pasa a `True`.
        min_delta: mejora mínima para contar como mejora real; evita que
            fluctuaciones de ruido reseteen el contador de paciencia. `0.0`
            (default) cuenta cualquier mejora, por pequeña que sea. El
            proyecto INC usa `0.005` sobre F1 de validación.
        mode: `"max"` para métricas donde más es mejor (f1, auc, accuracy,
            sensitivity, specificity, precision), `"min"` para donde menos
            es mejor (loss).

    Example:
        >>> stopper = EarlyStopping(patience=10, min_delta=0.005, mode="max")
        >>> for value in [0.70, 0.71, 0.705, 0.706]:
        ...     improved = stopper.step(value)
    """

    patience: int | None = None
    min_delta: float = 0.0
    mode: str = "max"

    best_value: float = field(init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Valida `mode` e inicializa `best_value` al peor valor posible para ese modo.

        Raises:
            ValueError: si `mode` no es `"max"` ni `"min"`.
        """
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode debe ser 'max' o 'min', recibido {self.mode!r}")
        self.best_value = float("-inf") if self.mode == "max" else float("inf")

    def step(self, value: float) -> bool:
        """Registra el valor de la métrica de validación de esta época.

        Args:
            value: valor de la métrica de validación de la época actual.

        Returns:
            bool: `True` si `value` es una mejora real (más allá de
                `min_delta`) sobre la mejor vista hasta ahora — la señal
                que `Trainer.fit()` usa para decidir si guardar checkpoint.
                `False` en caso contrario; incrementa `self.counter` y,
                si `self.patience` está fijado y se alcanza, activa
                `self.should_stop`.

        Example:
            >>> stopper = EarlyStopping(patience=2, mode="max")
            >>> stopper.step(0.80)   # True, primera vez
            >>> stopper.step(0.79)   # False, counter=1
            >>> stopper.step(0.79)   # False, counter=2 == patience -> should_stop=True
        """
        improved = (
            value > self.best_value + self.min_delta
            if self.mode == "max"
            else value < self.best_value - self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.patience is not None and self.counter >= self.patience:
                self.should_stop = True

        return improved
