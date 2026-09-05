"""Cabeza MLP multi-capa configurable, con activación elegible y BatchNorm opcional."""

import torch.nn as nn

from ..head_builder import HeadBuilder

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "linear": nn.Identity,
}


class ConfigurableMLPHead(HeadBuilder):
    """Cabeza MLP con número y tamaño de capas ocultas arbitrarios.

    Complementa a `StandardMLPHead` (`mlp_configs/standard_mlp.py`) — no la
    reemplaza. `StandardMLPHead` reproduce la cabeza fija de la serie de
    notebooks (una capa oculta, `BatchNorm1d` siempre, un único dropout);
    `ConfigurableMLPHead` reproduce la cabeza del proyecto INC
    (`classification_images/models/mlp_models.py`): N capas ocultas
    arbitrarias, activación elegible por nombre, sin `BatchNorm1d` por
    defecto. Ambas conviven registradas en `_HEAD_STRATEGIES`
    (`models/heads.py`).

    Arquitectura: `Flatten -> [Linear -> (BatchNorm1d?) -> activación -> Dropout] * N -> Linear`.

    Example:
        >>> builder = ConfigurableMLPHead(
        ...     in_features=2048, hidden_layers=[2048, 1024, 256],
        ...     activation="gelu", dropout=0.5, num_classes=2,
        ... )
        >>> head_module = builder.build()
    """

    def __init__(
        self,
        in_features: int = 2048,
        hidden_layers: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.5,
        num_classes: int = 2,
        use_batchnorm: bool = False,
        negative_slope: float = 0.01,
    ) -> None:
        """Inicializa los parámetros de la cabeza configurable.

        Args:
            in_features: dimensión de entrada (salida del backbone tras GAP,
                ej. 2048 para ResNet50).
            hidden_layers: tamaños de las capas ocultas, en orden. `[]` o
                `None` (default) produce un único `Linear(in_features,
                num_classes)`, sin capas ocultas.
            activation: nombre de la función de activación, case-insensitive.
                Opciones: `"relu"`, `"leakyrelu"`, `"sigmoid"`, `"tanh"`,
                `"gelu"`, `"linear"` (identidad, sin no-linealidad).
            dropout: probabilidad de dropout tras cada activación. `0`
                omite la capa `Dropout` por completo (no un `Dropout(p=0)`
                inerte).
            num_classes: logits de salida — 2 para `"cross_entropy"`/
                `"focal"`, 1 para `"bce"` (ver `LossSpec` en
                `train/build.py`).
            use_batchnorm: si `True`, inserta `BatchNorm1d` tras cada
                `Linear` oculto, como `StandardMLPHead`. El proyecto INC no
                lo usa — default `False`; activarlo es una variante propia,
                no parte de la reproducción.
            negative_slope: pendiente de la rama negativa, SOLO para
                `activation="leakyrelu"` (se ignora en las demás). Default
                `0.01`, el de `nn.LeakyReLU`. El proyecto INC usa `0.2`
                (`classification_images/models/mlp_models.py:get_activation`
                devuelve `nn.LeakyReLU(0.2)`), así que reproducirlo exige
                pasar `negative_slope: 0.2` explícitamente.

        Raises:
            ValueError: si `activation` no está en las opciones soportadas.
        """
        activation_key = activation.lower()
        if activation_key not in _ACTIVATIONS:
            raise ValueError(
                f"Activación desconocida: {activation!r}. Opciones: {sorted(_ACTIVATIONS)}"
            )
        self.in_features = in_features
        self.hidden_layers = hidden_layers or []
        self.activation = activation_key
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_batchnorm = use_batchnorm
        self.negative_slope = negative_slope

    def build(self) -> nn.Sequential:
        """Ensambla la cabeza según la configuración.

        Returns:
            nn.Sequential: capas listas para recibir la salida del backbone
                (un tensor `[B, in_features, 1, 1]` o `[B, in_features]`,
                el `Flatten()` inicial cubre ambos casos).

        Example:
            >>> head = ConfigurableMLPHead(in_features=2048, hidden_layers=[512]).build()
        """
        layers: list[nn.Module] = [nn.Flatten()]
        prev_size = self.in_features

        # `negative_slope` solo existe en LeakyReLU; el resto de activaciones
        # se construyen sin argumentos.
        activation_kwargs = (
            {"negative_slope": self.negative_slope} if self.activation == "leakyrelu" else {}
        )

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(_ACTIVATIONS[self.activation](**activation_kwargs))
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.num_classes))
        return nn.Sequential(*layers)
