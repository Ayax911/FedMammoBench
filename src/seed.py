"""Utilidades de reproducibilidad: semilla global de una sola vez al inicio
de la corrida, más la semilla por-worker y el generador determinista que el
DataLoader de train necesita porque num_workers>0 spawnea procesos aparte."""

import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fija la semilla en todas las fuentes de aleatoriedad del pipeline.

    Llamar UNA sola vez, al inicio de cada corrida, antes de construir
    datasets/modelos/dataloaders. random, numpy y torch mantienen estados
    de aleatoriedad independientes entre sí — y torch a su vez mantiene uno
    separado para CPU y otro para CUDA — por eso hacen falta las cuatro
    llamadas en vez de una sola.

    Args:
        seed: semilla a aplicar.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.manual_seed_all(seed)  # pyright: ignore[reportUnknownMemberType]


def seed_worker(worker_id: int) -> None:
    """Pensado para pasar como worker_init_fn= al DataLoader de train.

    PyTorch la llama una vez por worker, justo después de spawnear el
    proceso. set_global_seed() ya corrió antes en el proceso principal, pero
    cada worker (num_workers>0) es un proceso aparte con su propia copia del
    estado de random/numpy — sin esto, esa copia queda sin re-sembrar y el
    orden de augmentations por worker no es reproducible entre corridas.

    Args:
        worker_id: índice del worker. No se usa directamente acá, pero
            DataLoader exige que worker_init_fn acepte este parámetro.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Generador determinista para pasar como generator= al DataLoader de
    train — fija el orden del shuffle, independiente de en qué estado haya
    quedado el RNG global de torch para cuando se construye el loader.

    Args:
        seed: semilla del generador.

    Returns:
        torch.Generator: listo para pasar a DataLoader(generator=...).
    """
    g = torch.Generator()
    g.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    return g
