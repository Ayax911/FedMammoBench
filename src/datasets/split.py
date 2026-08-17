from .manifest import Manifest
import pandas as pd

class Split():
    def __init__(self, * ,manifest: Manifest) -> None:

        self.manifest = manifest
        self.patient_col = "patient_id"
        self.split_col = "split"

        self.verify_patient_consistency()
        self.splits: dict[str, list[str]] = self.group_by_split()

    def verify_patient_consistency(self) -> None:

        """Raise if any patient_id has rows spanning more than one split value.

        Raises:
            ValueError: one or more patients have inconsistent split assignment.
        """

        split_counts = self.manifest.df.groupby(self.patient_col)[self.split_col].nunique()
        invalid_patients = split_counts[split_counts > 1]

        if not invalid_patients.empty:
            raise ValueError(
                f"{len(invalid_patients)} patient(s) have rows in more than one "
                f"split: {invalid_patients.index.tolist()}"
            )

    def group_by_split(self) -> dict[str, list[str]]:
        """Return {'train': [...patient_ids...], 'val': [...], 'test': [...]},
        read directly off manifest.df — no computation, just grouping."""
        grouped = self.manifest.df.groupby(self.split_col)[self.patient_col].unique()
        return {str(split_name): patients.tolist() for split_name, patients in grouped.items()}

    def train_df(self) -> pd.DataFrame:
        """Return manifest.df filtered to only rows belonging to train patients."""
        train_patients = self.splits["train"]
        return self.manifest.df[self.manifest.df[self.patient_col].isin(train_patients)]

    def val_df(self) -> pd.DataFrame:
            """Return manifest.df filtered to only rows belonging to val patients."""
            val_patients = self.splits["val"]
            return self.manifest.df[self.manifest.df[self.patient_col].isin(val_patients)]

    def test_df(self) -> pd.DataFrame:
            """Return manifest.df filtered to only rows belonging to test patients."""
            test_patients = self.splits["test"]
            return self.manifest.df[self.manifest.df[self.patient_col].isin(test_patients)]