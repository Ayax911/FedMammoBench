"""Utilidades de reproducibilidad: semilla global de una sola vez al inicio
de la corrida, más la semilla por-worker y el generador determinista que el
DataLoader de train necesita porque `num_workers > 0` spawnea procesos aparte.

Ejemplo de uso:
    >>> from src.seed import set_global_seed, seed_worker, make_generator
    >>> set_global_seed(42)
    >>> gen = make_generator(42)
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fija la semilla en todas las fuentes de aleatoriedad del pipeline.

    Llamar UNA sola vez, al inicio de cada corrida, antes de construir
    datasets/modelos/dataloaders. `random`, `numpy` y `torch` mantienen estados
    de aleatoriedad independientes entre sí — y `torch` a su vez mantiene uno
    separado para CPU y otro para CUDA — por eso hacen falta las cuatro
    llamadas en vez de una sola.

    Args:
        seed: Valor entero de la semilla a aplicar globalmente.

    Example:
        >>> set_global_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed_all(seed)  # pyright: ignore[reportUnknownMemberType]


def seed_worker(worker_id: int) -> None:
    """Función de inicialización para pasar a `worker_init_fn=` en `DataLoader`.

    PyTorch la llama una vez por worker, justo después de spawnear el
    proceso. `set_global_seed()` ya corrió antes en el proceso principal, pero
    cada worker (`num_workers > 0`) es un proceso aparte con su propia copia del
    estado de random/numpy — sin esto, esa copia queda sin re-sembrar y el
    orden de augmentations por worker no es reproducible entre corridas.

    Args:
        worker_id: Índice del worker asignado por PyTorch DataLoader.

    Example:
        >>> loader = DataLoader(dataset, num_workers=2, worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Crea un generador determinista para pasar a `generator=` en `DataLoader`.

    Fija el orden del shuffle, independiente de en qué estado haya
    quedado el RNG global de PyTorch para cuando se construye el loader.

    Args:
        seed: Valor de semilla para inicializar el `torch.Generator`.

    Returns:
        torch.Generator: Objeto generador configurado con la semilla dada.

    Example:
        >>> gen = make_generator(42)
        >>> loader = DataLoader(dataset, shuffle=True, generator=gen)
    """
    g = torch.Generator()
    g.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    return g
