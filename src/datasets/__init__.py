"""Módulo de procesamiento de manifests, particionamiento de datos y data loaders.

Reexporta las clases y fábricas principales de dataset:
    - Manifest: Carga y validación de archivos CSV manifest.
    - Split: Particionamiento a nivel de paciente (anti-fuga de datos).
    - MammoBenchDataset: Dataset de PyTorch para imágenes de mamografía.
    - TransformBuilder: Fábrica de pipelines de transformaciones de torchvision.
    - builder_dataloader: Fábrica de DataLoaders de PyTorch para train/val/test.
"""

from .build import builder_dataloader
from .dataset import MammoBenchDataset
from .manifest import Manifest
from .split import Split
from .transform import TransformBuilder

__all__ = [
    "Manifest",
    "Split",
    "MammoBenchDataset",
    "TransformBuilder",
    "builder_dataloader",
]