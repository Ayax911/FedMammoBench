"""VinDr-Mammo dataset loader.

Expects the official PhysioNet release. The breast-level annotations CSV
``breast-level_annotations.csv`` is the source of labels and splits.

Assumed CSV columns (override in YAML if the release version differs):

- ``study_id``     : study identifier
- ``series_id``    : series identifier (usually present)
- ``image_id``     : image identifier (DICOM file stem)
- ``laterality``   : ``L`` or ``R``
- ``view_position``: ``CC`` or ``MLO``
- ``breast_birads``: 1..5 (sometimes written as ``BI-RADS 1`` etc.)
- ``split``        : ``training`` or ``test``

Image files live at ``<image_root>/<study_id>/<image_id>.dicom``.

BI-RADS to binary label mapping:

| BI-RADS | label                              |
|---------|------------------------------------|
| 1, 2    | benign                             |
| 4, 5    | malignant                          |
| 3       | configurable (drop / benign / malignant) |
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fedmammo.datasets.base import BENIGN, MALIGNANT, MammographyDataset, Sample
from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)

_BIRADS_RE = re.compile(r"(\d)")


def _parse_birads(value: object) -> int | None:
    """Extract an integer 1..5 from values like ``2``, ``'2'``, ``'BI-RADS 4'``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    m = _BIRADS_RE.search(s)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    if 1 <= n <= 5:
        return n
    return None


def _label_from_birads(birads: int, birads_3_policy: str) -> int | None:
    if birads in (1, 2):
        return BENIGN
    if birads in (4, 5):
        return MALIGNANT
    if birads == 3:
        if birads_3_policy == "benign":
            return BENIGN
        if birads_3_policy == "malignant":
            return MALIGNANT
        if birads_3_policy == "drop":
            return None
        raise ValueError(f"Unknown birads_3_policy: {birads_3_policy!r}")
    return None


class VinDrMammoDataset(MammographyDataset):
    """VinDr-Mammo loader (DICOM)."""

    @classmethod
    def from_annotations(
        cls,
        annotations_path: str | Path,
        image_root: str | Path,
        *,
        val_fraction: float = 0.1,
        birads_3_policy: Literal["drop", "benign", "malignant"] = "drop",
        seed: int = 42,
        verify_paths: bool = True,
        grayscale: bool = True,
        transform_train=None,  # noqa: ANN001
        transform_eval=None,  # noqa: ANN001
    ) -> dict[str, "VinDrMammoDataset"]:
        """Build train/val/test splits from the breast-level annotations CSV."""
        annotations_path = Path(annotations_path).expanduser().resolve()
        image_root = Path(image_root).expanduser().resolve()
        if not annotations_path.is_file():
            raise FileNotFoundError(f"VinDr-Mammo annotations not found: {annotations_path}")
        df = pd.read_csv(annotations_path)

        # Resolve column names defensively — the public release has mixed conventions.
        col_study = _pick_column(df, ["study_id", "studyInstanceUID", "study_instance_uid"])
        col_image = _pick_column(df, ["image_id", "sopInstanceUID", "sop_instance_uid"])
        col_birads = _pick_column(df, ["breast_birads", "birads", "breast_BIRADS"])
        col_split = _pick_column(df, ["split", "subset"])

        if col_birads is None:
            raise KeyError(
                f"Could not find a BI-RADS column in {annotations_path}. "
                f"Columns present: {list(df.columns)}"
            )

        df["_birads"] = df[col_birads].map(_parse_birads)
        df["_label"] = df["_birads"].map(
            lambda b: _label_from_birads(int(b), birads_3_policy) if b is not None else None
        )
        dropped = int(df["_label"].isna().sum())
        if dropped:
            _logger.warning(
                "Dropping %d VinDr-Mammo rows with unusable BI-RADS / labels (policy=%s)",
                dropped,
                birads_3_policy,
            )
        df = df.dropna(subset=["_label"]).copy()
        df["_label"] = df["_label"].astype(int)

        def _resolve_image(study_id: object, image_id: object) -> Path:
            stem = str(image_id)
            # VinDr-Mammo ships .dicom files; tolerate .dcm too.
            for ext in (".dicom", ".dcm"):
                candidate = image_root / str(study_id) / f"{stem}{ext}"
                if candidate.is_file():
                    return candidate
            # Fallback path even if not yet verified.
            return image_root / str(study_id) / f"{stem}.dicom"

        if col_study is None or col_image is None:
            raise KeyError(
                "VinDr-Mammo annotations are missing a study/image identifier column. "
                f"Found columns: {list(df.columns)}"
            )

        df["_abs_path"] = [
            _resolve_image(s, i)
            for s, i in zip(df[col_study].tolist(), df[col_image].tolist(), strict=True)
        ]
        if verify_paths:
            mask = df["_abs_path"].map(lambda p: p.is_file())
            missing = int((~mask).sum())
            if missing:
                _logger.warning(
                    "Dropping %d VinDr-Mammo rows whose DICOM files were missing under %s",
                    missing,
                    image_root,
                )
            df = df[mask].copy()

        if df.empty:
            raise RuntimeError(
                f"VinDr-Mammo annotations {annotations_path} yielded zero usable rows."
            )

        samples = [
            Sample(
                image_path=str(row["_abs_path"]),
                label=int(row["_label"]),
                patient_id=str(row[col_study]),  # patient id is study-level in VinDr
                extra={"birads": int(row["_birads"])},
            )
            for _, row in df.iterrows()
        ]

        # Determine splits.
        splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
        if col_split is not None:
            split_col = df[col_split].astype(str).str.lower()
            for i, sv in enumerate(split_col.tolist()):
                if "test" in sv:
                    splits["test"].append(i)
                else:
                    splits["train"].append(i)
            # Carve a validation chunk out of train.
            if val_fraction > 0 and splits["train"]:
                rng = np.random.default_rng(seed)
                train_idx = np.array(splits["train"])
                rng.shuffle(train_idx)
                n_val = int(len(train_idx) * val_fraction)
                splits["val"] = train_idx[:n_val].tolist()
                splits["train"] = train_idx[n_val:].tolist()
        else:
            # No split column — patient-disjoint stratified split.
            from fedmammo.datasets.cbis_ddsm import _stratified_patient_split

            splits = _stratified_patient_split(samples, val_fraction, 0.1, seed)

        out: dict[str, VinDrMammoDataset] = {}
        for split_name, idxs in splits.items():
            transform = transform_train if split_name == "train" else transform_eval
            out[split_name] = cls(
                samples=[samples[i] for i in idxs],
                transform=transform,
                image_format="dicom",
                grayscale=grayscale,
            )
            _logger.info(
                "VinDr-Mammo %s split: %d samples (counts=%s)",
                split_name,
                len(out[split_name]),
                out[split_name].class_counts(),
            )
        return out


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column from ``candidates`` present in ``df``, case-insensitively."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


__all__ = ["VinDrMammoDataset"]
