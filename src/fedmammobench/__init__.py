"""fedmammo: federated learning for binary mammography classification.

Top-level package. Modules:

- ``fedmammo.configs``    : dataclass schema + YAML loader
- ``fedmammo.datasets``   : dataset implementations and federated partitioning
- ``fedmammo.models``     : ResNet18, EfficientNet-B0, model factory
- ``fedmammo.training``   : reusable Trainer and loss functions
- ``fedmammo.evaluation`` : Evaluator and clinical-style metrics
- ``fedmammo.federated``  : Flower client / server / strategies
- ``fedmammo.utils``      : seeding, logging, checkpoints, TensorBoard, CSV
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
