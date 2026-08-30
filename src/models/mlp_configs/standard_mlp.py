import torch.nn as nn
from ..head_builder import HeadBuilder

class StandardMLPHead(HeadBuilder):
    """Standard multi-layer perceptron (MLP) classification head.

    Applies the following architecture sequence:
        Flatten -> Linear(in_features, hidden_dim) -> BatchNorm1d(hidden_dim) 
        -> ReLU(inplace=True) -> Dropout(p=dropout) -> Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        in_features: int = 2048,
        hidden_dim: int = 512,
        dropout: float = 0.5,
        num_classes: int = 1,
    ) -> None:
        """Initializes configuration parameters for the MLP head builder.

        Args:
            in_features: Input feature dimension output by the backbone encoder
                (e.g., 2048 for ResNet50 after global average pooling).
            hidden_dim: Number of channels in the intermediate hidden dense layer.
            dropout: Dropout probability applied after activation to prevent overfitting.
            num_classes: Number of output logits. Defaults to 1 — a single
                logit for binary BCEWithLogitsLoss classification (benign vs
                malignant). Must be paired with loss="bce" in build_loss()
                (train/build.py); num_classes=2 pairs with loss="cross_entropy"
                instead — nothing enforces this match automatically, see
                LossSpec's docstring in train/build.py.
        """
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_classes = num_classes

    def build(self) -> nn.Sequential:
        """Assembles the head components into a PyTorch nn.Sequential container.

        Returns:
            nn.Sequential: Executable sequence of linear, normalization, activation,
                dropout, and classification layers.
        """
        return nn.Sequential(
            # Flatten spatial dimensions [B, C, 1, 1] -> [B, C]
            nn.Flatten(),
            # Intermediate projection layer
            nn.Linear(self.in_features, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            # Regularization layer
            nn.Dropout(p=self.dropout),
            # Final output classification logits [B, num_classes]
            nn.Linear(self.hidden_dim, self.num_classes),
        )