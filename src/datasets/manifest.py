"""CSV manifest parsing, validation, column normalization, and path resolution module."""

from pathlib import Path
import pandas as pd


class Manifest:
    """Parses, validates, normalizes, and resolves image paths for dataset manifest CSVs.

    Ensures structural integrity during instantiation: verifies mandatory columns,
    checks patient ID presence, normalizes classification strings to binary labels,
    and resolves absolute file paths against an image root directory.

    Attributes:
        manifest_path: Path to the input manifest CSV file.
        image_root: Directory path used to resolve relative image paths.
        df: Processed pandas DataFrame augmented with `label_norm` and `abs_image_path`.

    Example:
        >>> from src.datasets.manifest import Manifest
        >>> manifest = Manifest("manifests/fedmammobench.csv", "data/images")
        >>> print(len(manifest.df))
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_root: str | Path,
        label_map: dict[str, int] | None = None,
    ) -> None:
        """Loads and validates a dataset manifest CSV.

        Args:
            manifest_path: File path to the manifest CSV.
            image_root: Directory against which relative image paths are resolved.
            label_map: optional `classification -> int` map for non-binary tasks
                (e.g. the 8 ISIC2019 classes). `None` (default) keeps the legacy
                `benign -> 0 / malignant -> 1` mapping. See `normalize_labels()`.

        Raises:
            FileNotFoundError: If `manifest_path` does not exist.
            NotADirectoryError: If `image_root` is not a valid directory.
            ValueError: If mandatory columns are missing, patient IDs are null,
                or classification values are unrecognized.

        Example:
            >>> manifest = Manifest("data/manifest.csv", "data/raw_images")
        """
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(manifest_path)
        self.image_root = Path(image_root)

        # Execute validation pipeline in fixed order
        self.verify_columns()
        self.check_patients_id()
        self.normalize_labels(label_map)
        self.resolve_image_paths()

    def verify_columns(self) -> None:
        """Verifies that all mandatory manifest columns exist in the DataFrame.

        Raises:
            ValueError: If one or more required columns (`preprocessed_image_path`,
                `classification`, `split`, `patient_id`) are missing.
        """
        required = ["preprocessed_image_path", "classification", "split", "patient_id"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest is missing required column(s): {missing}")

    def check_patients_id(self) -> None:
        """Verifies that no rows contain missing or null patient_id values.

        Raises:
            ValueError: If one or more rows contain null or blank patient IDs (necessary to
                prevent patient data leakage across train/val/test splits).
        """
        stripped = self.df["patient_id"].astype(str).str.strip()
        invalid_mask = self.df["patient_id"].isna() | (stripped == "") | (stripped.str.lower() == "nan")
        if invalid_mask.any():
            n_invalid = int(invalid_mask.sum())
            raise ValueError(
                f"{n_invalid} row(s) have a missing or blank patient_id. "
                "Every row must have a non-empty patient id — this is what prevents "
                "the same patient from landing in both train and test."
            )

    def normalize_labels(self, label_map: dict[str, int] | None = None) -> None:
        """Normalizes raw 'classification' strings into integer labels in self.df['label_norm'].

        Default (`label_map=None`): maps case-insensitive 'benign' -> 0 and 'malignant' -> 1.
        With `label_map` (keys compared case-insensitively, after strip), that map is used
        instead of the binary one (multiclass, e.g. ISIC2019).

        Args:
            label_map: optional `classification -> int` map. `None` = binary legacy map.

        Raises:
            ValueError: If any classification value is not recognized by the active map.
        """
        mapping = (
            {"benign": 0, "malignant": 1}
            if label_map is None
            else {str(k).strip().lower(): int(v) for k, v in label_map.items()}
        )
        cleaned = self.df["classification"].str.strip().str.lower()
        self.df["label_norm"] = cleaned.map(mapping)

        unmapped = self.df["label_norm"].isna()
        if unmapped.any():
            bad_values = self.df.loc[unmapped, "classification"].unique()
            raise ValueError(f"Unrecognized classification value(s): {list(bad_values)}")

        self.df["label_norm"] = self.df["label_norm"].astype(int)

    def resolve_image_paths(self) -> None:
        """Constructs self.df['abs_image_path'] by joining relative paths to self.image_root.

        Raises:
            NotADirectoryError: If self.image_root directory does not exist.
            ValueError: If preprocessed_image_path values are missing.
        """
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"image_root does not exist: {self.image_root}")

        if self.df["preprocessed_image_path"].isna().any():
            n_missing = int(self.df["preprocessed_image_path"].isna().sum())
            raise ValueError(f"{n_missing} row(s) have a missing preprocessed_image_path.")

        self.df["abs_image_path"] = self.df["preprocessed_image_path"].astype(str).map(
            lambda p: str(self.image_root / str(p))
        )
