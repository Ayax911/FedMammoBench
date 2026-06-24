"""Mammo-Bench dataset loader.

Mammo-Bench (https://github.com/...) consolidates ~19,700 mammography images
from six public sources (INbreast, DDSM, KAU-BCMD, CMMD, CDD-CESM, DMID) into
a single CSV manifest plus a tree of preprocessed JPG files. This loader
mirrors :class:`fedmammobench.datasets.CBISDDSMDataset` so its usage is consistent.

Expected CSV columns (defaults; not currently overridable via YAML):

- ``preprocessed_image_path`` : path to the JPG (relative to ``image_root``,
  or absolute)
- ``classification``          : one of ``Normal``, ``Benign``, ``Malignant``
- ``source_subjectID``        : patient identifier — used for non-leaky splits
- ``split`` (optional)        : ``train``/``val``/``test``. If absent, the
  loader produces stratified, patient-disjoint splits.

Binary label mapping:

- ``Malignant`` -> 1 (malignant)
- ``Benign``    -> 0 (benign)
- ``Normal``    -> 0 by default; drop entirely if ``normal_policy="drop"``.

Rows whose resolved image path does not exist on disk are logged and skipped
(same defensive behaviour as the other loaders).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fedmammobench.datasets.base import BENIGN, MALIGNANT, MammographyDataset, Sample
from fedmammobench.datasets.cbis_ddsm import _stratified_patient_split
from fedmammobench.datasets.registry import register_dataset
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# Column names used by the canonical Mammo-Bench CSV.
_COL_IMAGE = "preprocessed_image_path"
_COL_LABEL = "classification"
_COL_PATIENT = "source_subjectID"
_COL_SPLIT = "split"


def _normalize_label(raw: object, *, normal_policy: str) -> int | None:
    """Map a Mammo-Bench classification string to a binary label.

    Known labels in the dataset:
      - ``Normal``               -> 0 or drop depending on ``normal_policy``
      - ``Benign``               -> 0
      - ``Malignant``            -> 1
      - ``Suspicious Malignant`` -> dropped (ambiguous; matches official code behavior)

    Returns ``None`` when the row should be dropped.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    token = str(raw).strip().lower()
    if token == "suspicious malignant":
        # Dropped explicitly: ambiguous label present in both CSVs (235 cases).
        # The official Mammo-Bench classification code also excludes these rows.
        return None
    if token.startswith("malignant"):
        return MALIGNANT
    if token.startswith("benign"):
        return BENIGN
    if token.startswith("normal"):
        if normal_policy == "benign":
            return BENIGN
        if normal_policy == "drop":
            return None
        raise ValueError(
            f"Unknown normal_policy={normal_policy!r}; expected 'benign' or 'drop'."
        )
    return None


class MammoBenchDataset(MammographyDataset):
    """Mammo-Bench loader.

    Use :meth:`from_manifest` to build ``train``/``val``/``test`` splits.
    """

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        image_root: str | Path,
        *,
        normal_policy: Literal["benign", "drop"] = "benign",
        image_format: Literal["png", "jpg", "dicom"] = "jpg",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        verify_paths: bool = True,
        grayscale: bool = True,
        transform_train=None,  # noqa: ANN001
        transform_eval=None,  # noqa: ANN001
    ) -> dict[str, "MammoBenchDataset"]:
        """Build train/val/test splits from a Mammo-Bench CSV manifest.

        Args:
            manifest_path: Path to ``mammo-bench.csv`` (or any equivalent
                CSV with the columns documented at module level).
            image_root: Directory ``preprocessed_image_path`` is resolved
                against when the column value is relative.
            normal_policy: How to treat rows with ``classification == "Normal"``.
                ``benign`` maps them to class 0; ``drop`` removes them.
            image_format: Forwarded to :class:`MammographyDataset` for IO
                dispatch. Mammo-Bench is JPG; this is here for API symmetry.
            val_fraction: Validation share when the CSV has no ``split``
                column.
            test_fraction: Test share when the CSV has no ``split`` column.
            seed: Shuffling seed for stratified splits.
            verify_paths: If True, drop rows whose image file is missing.
            grayscale: Single-channel loading flag.
            transform_train: Train-time transform (Albumentations Compose).
            transform_eval: Val/test-time transform.

        Returns:
            Dict with keys ``train``, ``val``, ``test``.
        """
        manifest_path = Path(manifest_path).expanduser().resolve()
        image_root = Path(image_root).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Mammo-Bench manifest not found: {manifest_path}")
        df = pd.read_csv(manifest_path)

        for col in (_COL_IMAGE, _COL_LABEL):
            if col not in df.columns:
                raise KeyError(
                    f"Mammo-Bench manifest {manifest_path} is missing required column "
                    f"{col!r}. Available columns: {list(df.columns)}"
                )

        # Normalize labels (drops Normal rows when policy=drop).
        df["_label"] = df[_COL_LABEL].map(
            lambda v: _normalize_label(v, normal_policy=normal_policy)
        )
        dropped_label = int(df["_label"].isna().sum())
        if dropped_label:
            _logger.warning(
                "Dropping %d Mammo-Bench rows with unrecognized labels "
                "(normal_policy=%s)",
                dropped_label,
                normal_policy,
            )
        df = df.dropna(subset=["_label"]).copy()
        df["_label"] = df["_label"].astype(int)

        # Resolve image paths.
        def _resolve(p: object) -> Path:
            pp = Path(str(p))
            return pp if pp.is_absolute() else (image_root / pp)

        df["_abs_path"] = df[_COL_IMAGE].astype(str).map(_resolve)
        if verify_paths:
            mask = df["_abs_path"].map(lambda p: p.is_file())
            missing = int((~mask).sum())
            if missing:
                _logger.warning(
                    "Dropping %d Mammo-Bench rows whose image files were not "
                    "found under %s",
                    missing,
                    image_root,
                )
            df = df[mask].copy()

        if df.empty:
            raise RuntimeError(
                f"Mammo-Bench manifest {manifest_path} produced zero usable rows. "
                "Check image_root and label column."
            )

        # Build Sample objects.
        has_patient = _COL_PATIENT in df.columns
        samples = [
            Sample(
                image_path=str(row["_abs_path"]),
                label=int(row["_label"]),
                patient_id=(str(row[_COL_PATIENT]) if has_patient else None),
                # Keep the manifest's original image path so downstream
                # evaluation can merge per-sample predictions back onto the
                # manifest by an exact key (no path-resolution duplication).
                extra={_COL_IMAGE: str(row[_COL_IMAGE])},
            )
            for _, row in df.iterrows()
        ]

        # Splits.
        if _COL_SPLIT in df.columns:
            split_col = df[_COL_SPLIT].astype(str).str.lower()
            splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
            for i, sv in enumerate(split_col.tolist()):
                if sv in {"train", "training"}:
                    splits["train"].append(i)
                elif sv in {"val", "valid", "validation"}:
                    splits["val"].append(i)
                elif sv in {"test", "testing"}:
                    splits["test"].append(i)
            if not splits["val"] and val_fraction > 0 and splits["train"]:
                # Manifest has train/test only — carve val from train at patient level
                # to prevent the same patient appearing in both train and val.
                train_samples = [samples[i] for i in splits["train"]]
                sub_splits = _stratified_patient_split(
                    train_samples, val_fraction, 0.0, seed
                )
                orig = splits["train"]
                splits["train"] = [orig[i] for i in sub_splits["train"]]
                splits["val"] = [orig[i] for i in sub_splits["val"]]
        else:
            # Patient-disjoint stratified split.
            splits = _stratified_patient_split(samples, val_fraction, test_fraction, seed)

        out: dict[str, MammoBenchDataset] = {}
        for split_name, idxs in splits.items():
            transform = transform_train if split_name == "train" else transform_eval
            out[split_name] = cls(
                samples=[samples[i] for i in idxs],
                transform=transform,
                image_format=image_format,
                grayscale=grayscale,
            )
            _logger.info(
                "Mammo-Bench %s split: %d samples (class counts=%s)",
                split_name,
                len(out[split_name]),
                out[split_name].class_counts(),
            )
        return out


@register_dataset("mammo_bench")
def _build_mammo_bench(cfg, train_tx, eval_tx):  # noqa: ANN001, ANN201
    """Registered builder for the Mammo-Bench dataset."""
    if not cfg.data.manifest_path or not cfg.data.image_root:
        raise ValueError("data.name=mammo_bench requires both `manifest_path` and `image_root`.")
    return MammoBenchDataset.from_manifest(
        manifest_path=cfg.data.manifest_path,
        image_root=cfg.data.image_root,
        normal_policy=cfg.data.normal_policy,
        image_format=cfg.data.image_format,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.seed,
        grayscale=cfg.data.grayscale,
        transform_train=train_tx,
        transform_eval=eval_tx,
    )


__all__ = ["MammoBenchDataset"]
