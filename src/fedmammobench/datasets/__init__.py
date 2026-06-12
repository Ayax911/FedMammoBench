"""Dataset implementations for fedmammobench.

The package provides:

- ``MammographyDataset``        : abstract base class
- ``CBISDDSMDataset``           : CBIS-DDSM loader (PNG or DICOM)
- ``VinDrMammoDataset``         : VinDr-Mammo loader (DICOM)
- ``MammoBenchDataset``         : Mammo-Bench loader (JPG)
- ``build_dataset``             : factory keyed by config
- ``register_dataset`` / ``list_datasets`` : registry to add new datasets
- ``partition_indices``         : IID / Dirichlet / quantity-skew partitioning
- ``build_transforms``          : Albumentations pipelines

Importing the loader modules below also populates the dataset registry via
their ``@register_dataset`` decorators. To add a new dataset, create a loader
module that registers its builder and import it here.
"""

from fedmammobench.datasets.base import MammographyDataset, Sample
from fedmammobench.datasets.cbis_ddsm import CBISDDSMDataset
from fedmammobench.datasets.factory import build_dataset
from fedmammobench.datasets.loaders import build_dataloader
from fedmammobench.datasets.mammo_bench import MammoBenchDataset
from fedmammobench.datasets.partitioning import partition_indices
from fedmammobench.datasets.registry import (
    list_datasets,
    register_dataset,
)
from fedmammobench.datasets.transforms import build_transforms
from fedmammobench.datasets.vindr_mammo import VinDrMammoDataset

__all__ = [
    "MammographyDataset",
    "Sample",
    "CBISDDSMDataset",
    "VinDrMammoDataset",
    "MammoBenchDataset",
    "build_dataset",
    "build_dataloader",
    "partition_indices",
    "register_dataset",
    "list_datasets",
    "build_transforms",
]
