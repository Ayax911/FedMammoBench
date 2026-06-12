"""Model definitions and factory."""

from fedmammo.models.factory import build_model, list_models, register_model
from fedmammo.models.resnet import ResNet18Classifier, ResNet50Classifier
from fedmammo.models.efficientnet import EfficientNetB0Classifier
from fedmammo.models.densenet import DenseNet121Classifier
from fedmammo.models.inception import InceptionV3Classifier
from fedmammo.models.weight_loaders import (
    LoadReport,
    WeightLoader,
    apply_freeze_policy,
    load_weights,
    register_loader,
    resolve_source,
)

__all__ = [
    "LoadReport",
    "WeightLoader",
    "apply_freeze_policy",
    "build_model",
    "DenseNet121Classifier",
    "EfficientNetB0Classifier",
    "InceptionV3Classifier",
    "list_models",
    "load_weights",
    "register_loader",
    "register_model",
    "ResNet18Classifier",
    "ResNet50Classifier",
    "resolve_source",
]
