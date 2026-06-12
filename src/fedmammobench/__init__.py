"""fedmammobench: federated learning for binary mammography classification.

Top-level package. Modules:

- ``fedmammobench.configs``    : dataclass schema + YAML loader
- ``fedmammobench.datasets``   : dataset implementations and federated partitioning
- ``fedmammobench.models``     : ResNet18, EfficientNet-B0, model factory
- ``fedmammobench.training``   : reusable Trainer and loss functions
- ``fedmammobench.evaluation`` : Evaluator and clinical-style metrics
- ``fedmammobench.federated``  : Flower client / server / strategies
- ``fedmammobench.utils``      : seeding, logging, checkpoints, TensorBoard, CSV
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
