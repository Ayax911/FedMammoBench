import pandas as pd
from pathlib import Path

class Manifest:
    def __init__(self, manifest_path: str | Path, image_root: str | Path)-> None:
        """Load and validate a manifest CSV is structurally trustworthy.

        Args:
            manifest_path: Path to the manifest CSV file.
            image_root: Directory that relative image paths in the manifest
            are resolved against.

        Raises:
            FileNotFoundError: If the manifest file does not exist.
            NotADirectoryError: image_root doesn't exist.
            ValueError: If a required column is missing.
        """
        self.manifest_path = Path(manifest_path)
        self.df = pd.read_csv(manifest_path)
        self.image_root = Path(image_root)

        self.verify_columns()
        self.check_patients_id()
        self.normalize_labels()
        self.resolve_image_paths()

    def verify_columns(self) -> None:
        """Raise if any required column is missing.

        Raises:
            ValueError: one or more required columns are missing.
        """
        required = ["preprocessed_image_path", "classification", "split", "patient_id"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Manifest is missing required column(s): {missing}")

    def check_patients_id(self) -> None:
        """Raise if any row has a missing patient_id.

        Raises:
            ValueError: one or more rows have a null patient_id.
        """
        null_mask = self.df['patient_id'].isna()
        if null_mask.any():
            n_null = int(null_mask.sum())
            raise ValueError(
                f"{n_null} row(s) have a missing patient_id. "
                "Every row must have a patient id — this is what prevents "
                "the same patient from landing in both train and test."
            )
        
    def normalize_labels(self) -> None:
        """Add self.df['label_norm'] (0/1) from the raw 'classification' column.
        Leaves 'classification' untouched.

        Raises:
            ValueError: a classification value isn't recognized.
        """
        cleaned = self.df['classification'].str.strip().str.lower()
        self.df['label_norm'] = cleaned.map({'benign': 0, 'malignant': 1})

        unmapped = self.df['label_norm'].isna()
        if unmapped.any():
            bad_values = self.df.loc[unmapped, 'classification'].unique()
            raise ValueError(f"Unrecognized classification value(s): {list(bad_values)}")

        self.df['label_norm'] = self.df['label_norm'].astype(int)

    def resolve_image_paths(self) -> None:
        """Add self.df['abs_image_path'] by joining preprocessed_image_path
        onto self.image_root.

        Raises:
            NotADirectoryError: self.image_root doesn't exist.
        """
        if not self.image_root.is_dir():
            raise NotADirectoryError(f"image_root does not exist: {self.image_root}")

        if self.df["preprocessed_image_path"].isna().any():
            n_missing = int(self.df["preprocessed_image_path"].isna().sum())
            raise ValueError(f"{n_missing} row(s) have a missing preprocessed_image_path.")

        self.df["abs_image_path"] = self.df["preprocessed_image_path"].astype(str).map(
            lambda p: str(self.image_root / str(p))
        )
          
          
    

  

        
        

        


    
