"""Dataset implementations for fedmammo.

The package provides:

- ``MammographyDataset``        : abstract base class
- ``SyntheticMammographyDataset`` : noise-tensor stand-in for smoke tests / CI
- ``CBISDDSMDataset``           : CBIS-DDSM loader (PNG or DICOM)
- ``VinDrMammoDataset``         : VinDr-Mammo loader (DICOM)
- ``build_dataset``             : factory keyed by config
- ``partition_indices``         : IID / Dirichlet / quantity-skew partitioning
- ``build_transforms``          : Albumentations pipelines
"""

from fedmammo.datasets.base import MammographyDataset, Sample
from fedmammo.datasets.cbis_ddsm import CBISDDSMDataset
from fedmammo.datasets.factory import build_dataset
from fedmammo.datasets.loaders import build_dataloader
from fedmammo.datasets.mammo_bench import MammoBenchDataset
from fedmammo.datasets.partitioning import partition_indices
from fedmammo.datasets.synthetic import SyntheticMammographyDataset
from fedmammo.datasets.transforms import build_transforms
from fedmammo.datasets.vindr_mammo import VinDrMammoDataset

__all__ = [
    "MammographyDataset",
    "Sample",
    "SyntheticMammographyDataset",
    "CBISDDSMDataset",
    "VinDrMammoDataset",
    "MammoBenchDataset",
    "build_dataset",
    "build_dataloader",
    "partition_indices",
    "build_transforms",
]
