"""CBIS-DDSM dataset loader.

Assumes the user has downloaded the CBIS-DDSM release (TCIA) and that a CSV
manifest exists with at least these columns (defaults shown — override in
YAML if your manifest uses different names):

- ``image_path``  : path to the image file (PNG, JPG, or DICOM)
- ``pathology``   : one of ``BENIGN``, ``BENIGN_WITHOUT_CALLBACK``, ``MALIGNANT``
- ``patient_id``  : patient identifier (used for non-leaky train/val/test splits)
- ``split``       : ``train`` / ``val`` / ``test`` — *optional*; if absent we
                    generate stratified, patient-disjoint splits.

The loader normalizes labels: any value containing ``MALIGNANT`` (case-insensitive)
becomes 1; ``BENIGN`` or ``BENIGN_WITHOUT_CALLBACK`` becomes 0. Other labels are
dropped with a logged warning.

If ``image_path`` is relative, it is resolved against ``image_root``.
Rows whose resolved path does not exist on disk are dropped (with a warning).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from fedmammobench.configs.schema import DataColumnMapping
from fedmammobench.datasets.base import BENIGN, MALIGNANT, MammographyDataset, Sample
from fedmammobench.datasets.registry import register_dataset
from fedmammobench.utils.logging_utils import get_logger

_logger = get_logger(__name__)

_BENIGN_TOKENS = ("benign", "benign_without_callback", "benign-without-callback")


def _normalize_label(raw: str) -> int | None:
    """Map a raw pathology string to a binary label or None to drop."""
    if raw is None:
        return None
    token = str(raw).strip().lower().replace(" ", "_")
    if "malignant" in token:
        return MALIGNANT
    if token in _BENIGN_TOKENS or token.startswith("benign"):
        return BENIGN
    return None


def _stratified_patient_split(
    samples: list[Sample], val_fraction: float, test_fraction: float, seed: int
) -> dict[str, list[int]]:
    """Split ``samples`` by ``patient_id`` so a patient never appears in two splits.

    Positive and negative patients are split independently at the requested
    fractions and then recombined, guaranteeing that each split contains at
    least one patient from the rare (malignant) class whenever possible.
    The previous approach of sorting all patients by malignancy rate and then
    taking consecutive slices placed all rare-class patients in val/test,
    leaving the train set with zero positives.
    """
    if not 0.0 <= val_fraction + test_fraction < 1.0:
        raise ValueError(
            "val_fraction + test_fraction must lie in [0, 1), got "
            f"{val_fraction} + {test_fraction}"
        )

    by_patient: dict[str, list[int]] = {}
    for idx, s in enumerate(samples):
        pid = s.patient_id or f"_anon_{idx}"
        by_patient.setdefault(pid, []).append(idx)

    rng = np.random.default_rng(seed)
    patients = list(by_patient.keys())

    # Separate patients by whether they have at least one positive sample.
    pos_patients = [p for p in patients if any(samples[i].label == 1 for i in by_patient[p])]
    neg_patients = [p for p in patients if p not in set(pos_patients)]
    rng.shuffle(pos_patients)
    rng.shuffle(neg_patients)

    def _split_group(group: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(group)
        if n == 0:
            return [], [], []
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        n_train = n - n_val - n_test
        if n_train <= 0:
            # Too few patients to populate all splits — keep at least 1 in train.
            n_train = 1
            n_val = max(n - n_train - n_test, 0)
        return group[:n_train], group[n_train : n_train + n_val], group[n_train + n_val :]

    pos_train, pos_val, pos_test = _split_group(pos_patients)
    neg_train, neg_val, neg_test = _split_group(neg_patients)

    buckets: dict[str, list[str]] = {
        "train": pos_train + neg_train,
        "val": pos_val + neg_val,
        "test": pos_test + neg_test,
    }

    if len(buckets["train"]) == 0:
        raise ValueError(f"Splits leave no train patients (n_pat={len(patients)})")

    out: dict[str, list[int]] = {}
    for split, pats in buckets.items():
        idx = [i for p in pats for i in by_patient[p]]
        rng.shuffle(idx)
        out[split] = idx
    return out


class CBISDDSMDataset(MammographyDataset):
    """CBIS-DDSM loader.

    Use :meth:`from_manifest` to construct ``train``/``val``/``test`` splits.
    """

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        image_root: str | Path,
        *,
        columns: DataColumnMapping,
        image_format: Literal["png", "jpg", "dicom"] = "png",
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 42,
        verify_paths: bool = True,
        grayscale: bool = True,
        transform_train=None,  # noqa: ANN001
        transform_eval=None,  # noqa: ANN001
    ) -> dict[str, "CBISDDSMDataset"]:
        """Build train/val/test splits from a CSV manifest.

        Returns a dict ``{"train": ..., "val": ..., "test": ...}`` of
        :class:`CBISDDSMDataset` instances. The eval transform is used for
        ``val`` and ``test``.
        """
        manifest_path = Path(manifest_path).expanduser().resolve()
        image_root = Path(image_root).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"CBIS-DDSM manifest not found: {manifest_path}")
        df = pd.read_csv(manifest_path)

        required = [columns.image_path, columns.label]
        for col in required:
            if col not in df.columns:
                raise KeyError(
                    f"Manifest {manifest_path} is missing required column {col!r}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Map labels, drop rows we can't classify.
        df["_label"] = df[columns.label].map(_normalize_label)
        dropped_label = int(df["_label"].isna().sum())
        if dropped_label:
            _logger.warning("Dropping %d rows with unrecognized labels", dropped_label)
        df = df.dropna(subset=["_label"]).copy()
        df["_label"] = df["_label"].astype(int)

        # Resolve image paths and (optionally) verify existence.
        def _resolve(p: str) -> Path:
            pp = Path(p)
            return pp if pp.is_absolute() else (image_root / pp)

        df["_abs_path"] = df[columns.image_path].astype(str).map(_resolve)
        if verify_paths:
            exists_mask = df["_abs_path"].map(lambda p: p.is_file())
            missing = int((~exists_mask).sum())
            if missing:
                _logger.warning(
                    "Dropping %d rows whose image files were not found under %s",
                    missing,
                    image_root,
                )
            df = df[exists_mask].copy()

        if df.empty:
            raise RuntimeError(
                f"CBIS-DDSM manifest {manifest_path} produced zero usable rows. "
                "Check image_root and label column."
            )

        samples = [
            Sample(
                image_path=str(row["_abs_path"]),
                label=int(row["_label"]),
                patient_id=(
                    str(row[columns.patient_id]) if columns.patient_id in df.columns else None
                ),
                extra={},
            )
            for _, row in df.iterrows()
        ]

        # Determine splits.
        if columns.split in df.columns:
            split_col = df[columns.split].astype(str).str.lower()
            splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
            for i, sv in enumerate(split_col.tolist()):
                if sv in {"train", "training"}:
                    splits["train"].append(i)
                elif sv in {"val", "valid", "validation"}:
                    splits["val"].append(i)
                elif sv in {"test", "testing"}:
                    splits["test"].append(i)
            if not splits["val"] and val_fraction > 0:
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
            splits = _stratified_patient_split(samples, val_fraction, test_fraction, seed)

        out: dict[str, CBISDDSMDataset] = {}
        for split_name, idxs in splits.items():
            transform = transform_train if split_name == "train" else transform_eval
            out[split_name] = cls(
                samples=[samples[i] for i in idxs],
                transform=transform,
                image_format=image_format,
                grayscale=grayscale,
            )
            counts = out[split_name].class_counts()
            _logger.info(
                "CBIS-DDSM %s split: %d samples (counts=%s)",
                split_name,
                len(out[split_name]),
                counts,
            )
        return out


@register_dataset("cbis_ddsm")
def _build_cbis_ddsm(cfg, train_tx, eval_tx):  # noqa: ANN001, ANN201
    """Registered builder for the CBIS-DDSM dataset."""
    if not cfg.data.manifest_path or not cfg.data.image_root:
        raise ValueError("data.name=cbis_ddsm requires both `manifest_path` and `image_root`.")
    return CBISDDSMDataset.from_manifest(
        manifest_path=cfg.data.manifest_path,
        image_root=cfg.data.image_root,
        columns=cfg.data.columns,
        image_format=cfg.data.image_format,
        val_fraction=cfg.data.val_fraction,
        test_fraction=cfg.data.test_fraction,
        seed=cfg.seed,
        grayscale=cfg.data.grayscale,
        transform_train=train_tx,
        transform_eval=eval_tx,
    )


__all__ = ["CBISDDSMDataset"]
