"""Abstract base class for mammography datasets.

Subclasses are responsible for producing a list of :class:`Sample` instances;
the base class handles image IO (PNG/JPG/DICOM), transform application, and
provides a ``labels`` accessor used by federated partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# 0 = benign, 1 = malignant
BENIGN: int = 0
MALIGNANT: int = 1
LABEL_NAMES: tuple[str, str] = ("benign", "malignant")


@dataclass(frozen=True)
class Sample:
    """A single dataset record.

    Attributes:
        image_path: Path to the image file on disk.
        label: 0 (benign) or 1 (malignant).
        patient_id: Optional patient identifier; used for non-leaky splits.
        extra: Free-form per-sample metadata (study id, view, BI-RADS, ...).
    """

    image_path: str
    label: int
    patient_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class MammographyDataset(Dataset):
    """Base class.

    Subclasses must populate ``samples`` (a list of :class:`Sample`) before or
    during ``__init__``. The base ``__getitem__`` reads the referenced file,
    runs the (optional) Albumentations transform, and returns
    ``(tensor, label)``.

    Args:
        samples: List of samples to expose.
        transform: Optional Albumentations transform (or callable producing
            ``{"image": tensor}`` when called with ``image=np.ndarray``).
            If None, the raw uint8 numpy array is returned as a float tensor.
        image_format: One of ``png``, ``jpg``, ``dicom``. The loader uses the
            file extension when available and falls back to this hint.
        grayscale: If True, load images as single-channel.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        transform: Any | None = None,
        image_format: str = "png",
        grayscale: bool = True,
    ) -> None:
        self.samples: list[Sample] = list(samples)
        self.transform = transform
        self.image_format = image_format.lower()
        self.grayscale = grayscale

    # ------------------------------------------------------------------
    # Sequence API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = self.load_image(sample.image_path)
        image_t = self.apply_transform(image)
        return image_t, int(sample.label)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """Per-sample labels as an int64 numpy array (cheap to access)."""
        return np.asarray([s.label for s in self.samples], dtype=np.int64)

    @property
    def patient_ids(self) -> list[str | None]:
        return [s.patient_id for s in self.samples]

    def class_counts(self) -> dict[int, int]:
        """Return ``{class_index: count}`` for the contained samples."""
        labels = self.labels
        unique, counts = np.unique(labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts, strict=True)}

    def subset(self, indices: Sequence[int]) -> "MammographyDataset":
        """Return a shallow copy containing only ``indices``.

        Used by partitioning to materialize client-local datasets that share
        the parent's transform and IO settings.
        """
        sub = MammographyDataset(
            samples=[self.samples[i] for i in indices],
            transform=self.transform,
            image_format=self.image_format,
            grayscale=self.grayscale,
        )
        # Allow subclasses to attach overridden read methods.
        sub._read_array = self._read_array  # type: ignore[attr-defined]
        sub._read_dicom = self._read_dicom  # type: ignore[attr-defined]
        return sub

    # ------------------------------------------------------------------
    # IO + transform
    # ------------------------------------------------------------------

    def load_image(self, path: str) -> np.ndarray:
        """Return a uint8 HxW (grayscale) or HxWxC numpy array."""
        ext = Path(path).suffix.lower()
        if ext in {".dcm", ".dicom"} or self.image_format == "dicom":
            arr = self._read_dicom(path)
        else:
            arr = self._read_array(path)
        if self.grayscale and arr.ndim == 3:
            # Convert RGB/BGR to grayscale by luminosity.
            arr = (
                0.299 * arr[..., 0].astype(np.float32)
                + 0.587 * arr[..., 1].astype(np.float32)
                + 0.114 * arr[..., 2].astype(np.float32)
            ).astype(np.uint8)
        if not self.grayscale and arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr

    def apply_transform(self, image: np.ndarray) -> torch.Tensor:
        if self.transform is None:
            tensor = torch.from_numpy(image).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.permute(2, 0, 1).contiguous()
            return tensor / 255.0
        out = self.transform(image=image)
        return out["image"]

    # ------------------------------------------------------------------
    # IO primitives (overridable)
    # ------------------------------------------------------------------

    def _read_array(self, path: str) -> np.ndarray:
        """Read a PNG/JPG/TIFF using OpenCV. Returns uint8 array."""
        import cv2  # local import keeps optional dependency at module top

        flags = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        arr = cv2.imread(path, flags)
        if arr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if not self.grayscale:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return arr

    def _read_dicom(self, path: str) -> np.ndarray:
        """Read a DICOM file and return a uint8 array.

        Applies VOI LUT when present and normalizes to [0, 255]. Inverts
        photometric MONOCHROME1 (white-on-black) to MONOCHROME2.
        """
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        ds = pydicom.dcmread(path)
        try:
            arr = apply_voi_lut(ds.pixel_array, ds)
        except Exception:  # noqa: BLE001 — VOI LUT often missing on converted DICOMs
            arr = ds.pixel_array
        arr = arr.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr_min = float(arr.min())
        arr_max = float(arr.max())
        if arr_max - arr_min > 0:
            arr = (arr - arr_min) / (arr_max - arr_min)
        arr = (arr * 255.0).astype(np.uint8)
        return arr


__all__ = ["BENIGN", "MALIGNANT", "LABEL_NAMES", "Sample", "MammographyDataset"]
