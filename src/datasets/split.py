"""Patient-aware dataset split validation and DataFrame extraction module."""

import pandas as pd
from .manifest import Manifest


class Split:
    """Derives and validates patient-disjoint train, validation, and test splits from a Manifest.

    Guarantees strict patient anti-leakage: validates that no single `patient_id` has images
    assigned across multiple split partitions.

    Args:
        manifest: Validated `Manifest` instance containing patient_id and split columns.

    Attributes:
        manifest: Reference to the underlying `Manifest`.
        patient_col: Column name identifying patient IDs (`"patient_id"`).
        split_col: Column name identifying split partition names (`"split"`).
        splits: Dictionary mapping split names (`"train"`, `"val"`, `"test"`) to unique patient ID lists.

    Example:
        >>> from src.datasets.manifest import Manifest
        >>> from src.datasets.split import Split
        >>> manifest = Manifest("manifests/fedmammobench.csv", "data/images")
        >>> split = Split(manifest=manifest)
        >>> df_train = split.train_df()
    """

    def __init__(self, *, manifest: Manifest) -> None:
        """Initializes Split object and validates anti-leakage patient consistency.

        Args:
            manifest: Validated `Manifest` object.

        Raises:
            ValueError: If any patient_id appears in multiple split partitions.
        """
        self.manifest = manifest
        self.patient_col = "patient_id"
        self.split_col = "split"

        # Validate anti-leakage consistency across patients
        self.verify_patient_consistency()
        self.splits: dict[str, list[str]] = self.group_by_split()

    def verify_patient_consistency(self) -> None:
        """Verifies that no patient ID has records assigned to multiple split values.

        Raises:
            ValueError: If one or more patients have images split across train, val, or test.
        """
        split_counts = self.manifest.df.groupby(self.patient_col)[self.split_col].nunique()
        invalid_patients = split_counts[split_counts > 1]

        if not invalid_patients.empty:
            raise ValueError(
                f"{len(invalid_patients)} patient(s) have rows in more than one "
                f"split: {invalid_patients.index.tolist()}"
            )

    def group_by_split(self) -> dict[str, list[str]]:
        """Groups patient IDs by their designated split partition name.

        Returns:
            dict[str, list[str]]: Dictionary mapping split names (e.g. `"train"`, `"val"`, `"test"`)
                to lists of unique patient ID strings.
        """
        grouped = self.manifest.df.groupby(self.split_col)[self.patient_col].unique()
        return {str(split_name): patients.tolist() for split_name, patients in grouped.items()}

    def train_df(self) -> pd.DataFrame:
        """Extracts manifest rows belonging to the training split patients.

        Returns:
            pd.DataFrame: Filtered DataFrame containing training samples.
        """
        train_patients = self.splits["train"]
        return self.manifest.df[self.manifest.df[self.patient_col].isin(train_patients)]

    def val_df(self) -> pd.DataFrame:
        """Extracts manifest rows belonging to the validation split patients.

        Returns:
            pd.DataFrame: Filtered DataFrame containing validation samples.
        """
        val_patients = self.splits["val"]
        return self.manifest.df[self.manifest.df[self.patient_col].isin(val_patients)]

    def test_df(self) -> pd.DataFrame:
        """Extracts manifest rows belonging to the test split patients.

        Returns:
            pd.DataFrame: Filtered DataFrame containing test samples.
        """
        test_patients = self.splits["test"]
        return self.manifest.df[self.manifest.df[self.patient_col].isin(test_patients)]