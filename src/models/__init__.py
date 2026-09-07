"""Módulo de arquitecturas, pesos preentrenados, congelamiento y cabezas de clasificación.

Reexporta componentes clave de construcción y gestión de modelos:
    - build_model, ArchitectureSpec: Fábrica y especificación de modelos backbone.
    - load_weights, truncate_backbone: Carga/remapeo de tensores de checkpoints y truncado de encoder.
    - FreezeStrategy, ResNetFreezeStrategy: Estrategias de congelamiento por bloques.
    - HeadBuilder, get_head_strategy: Interfaz e inspección de cabezas de clasificación.
    - LoadReport: Reporte de coincidencia de tensores al cargar pesos.
"""

from .build import ArchitectureSpec, build_model
from .freeze import FreezeStrategy, ResNetFreezeStrategy
from .head_builder import HeadBuilder
from .heads import get_head_strategy
from .reports import LoadReport
from .weights import load_weights, truncate_backbone

__all__ = [
    "build_model",
    "ArchitectureSpec",
    "load_weights",
    "truncate_backbone",
    "FreezeStrategy",
    "ResNetFreezeStrategy",
    "HeadBuilder",
    "get_head_strategy",
    "LoadReport",
]