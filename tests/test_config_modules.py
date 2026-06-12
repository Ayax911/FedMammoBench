"""Tests for per-section config modules (R1) and methodology corrections (C1, C2, C5).

Run::

    pytest tests/test_config_modules.py -v
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ---------------------------------------------------------------------------
# DataConfig validation
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_valid_fractions(self):
        from fedmammobench.configs.data_config import DataConfig
        cfg = DataConfig(val_fraction=0.1, test_fraction=0.1)
        cfg.validate()  # must not raise

    def test_fractions_exceed_one_raises(self):
        from fedmammobench.configs.data_config import DataConfig
        cfg = DataConfig(val_fraction=0.6, test_fraction=0.6)
        with pytest.raises(ValueError, match="val_fraction"):
            cfg.validate()

    def test_fractions_sum_to_one_allowed_for_server(self):
        # val=0.0 + test=1.0 is valid for gRPC server holdout configs
        from fedmammobench.configs.data_config import DataConfig
        cfg = DataConfig(name="mammo_bench", val_fraction=0.0, test_fraction=1.0)
        cfg.validate()  # must not raise

    def test_zero_fractions_valid(self):
        from fedmammobench.configs.data_config import DataConfig
        cfg = DataConfig(val_fraction=0.0, test_fraction=0.0)
        cfg.validate()

    def test_batch_size_zero_raises(self):
        from fedmammobench.configs.data_config import DataConfig
        cfg = DataConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size"):
            cfg.validate()


# ---------------------------------------------------------------------------
# C5: check_patient_ids_for_nan
# ---------------------------------------------------------------------------

class TestCheckPatientIdsForNan:
    def test_all_valid_returns_false(self):
        from fedmammobench.configs.data_config import check_patient_ids_for_nan
        assert check_patient_ids_for_nan(["p001", "p002", "p003"]) is False

    def test_none_returns_true(self):
        from fedmammobench.configs.data_config import check_patient_ids_for_nan
        assert check_patient_ids_for_nan(["p001", None, "p003"]) is True

    def test_float_nan_returns_true(self):
        from fedmammobench.configs.data_config import check_patient_ids_for_nan
        assert check_patient_ids_for_nan(["p001", float("nan"), "p003"]) is True

    def test_math_nan_returns_true(self):
        from fedmammobench.configs.data_config import check_patient_ids_for_nan
        assert check_patient_ids_for_nan([math.nan]) is True

    def test_empty_list_returns_false(self):
        from fedmammobench.configs.data_config import check_patient_ids_for_nan
        assert check_patient_ids_for_nan([]) is False

    def test_integer_ids_valid(self):
        from fedmammobench.configs.data_config import check_patient_ids_for_nan
        assert check_patient_ids_for_nan([1, 2, 3, 4]) is False


# ---------------------------------------------------------------------------
# ModelConfig validation
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_valid_config_no_raise(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(name="resnet50", weight_source="radimagenet", in_channels=1)
        cfg.validate(normalize_preset="radimagenet_gray")

    def test_radimagenet_unsupported_arch_raises(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(name="efficientnet_b0", weight_source="radimagenet", in_channels=1)
        with pytest.raises(ValueError, match="radimagenet"):
            cfg.validate()

    def test_custom_without_path_raises(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(name="resnet50", weight_source="custom", checkpoint_path=None)
        with pytest.raises(ValueError, match="checkpoint_path"):
            cfg.validate()

    def test_gray_preset_wrong_channels_raises(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(name="resnet50", in_channels=3)
        with pytest.raises(ValueError, match="in_channels"):
            cfg.validate(normalize_preset="radimagenet_gray")

    def test_rgb_preset_wrong_channels_raises(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(name="resnet50", in_channels=1)
        with pytest.raises(ValueError, match="in_channels"):
            cfg.validate(normalize_preset="imagenet_rgb")

    def test_gray_preset_correct_channels_ok(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(name="resnet18", in_channels=1)
        cfg.validate(normalize_preset="imagenet_gray")

    def test_effective_weight_source_auto_pretrained(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(pretrained=True, weight_source="auto")
        assert cfg._effective_weight_source() == "imagenet"

    def test_effective_weight_source_auto_not_pretrained(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(pretrained=False, weight_source="auto")
        assert cfg._effective_weight_source() == "none"

    def test_config_hash_fields_contains_name(self):
        from fedmammobench.configs.model_config import ModelConfig
        cfg = ModelConfig(name="resnet50", num_classes=2, in_channels=1)
        fields = cfg.config_hash_fields()
        assert fields["name"] == "resnet50"
        assert fields["num_classes"] == 2


# ---------------------------------------------------------------------------
# PartitioningConfig validation
# ---------------------------------------------------------------------------

class TestPartitioningConfig:
    def test_valid_config(self):
        from fedmammobench.configs.data_config import PartitioningConfig
        cfg = PartitioningConfig(alpha=0.5, min_per_client=16)
        cfg.validate()

    def test_alpha_zero_raises(self):
        from fedmammobench.configs.data_config import PartitioningConfig
        cfg = PartitioningConfig(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            cfg.validate()

    def test_min_per_client_zero_raises(self):
        from fedmammobench.configs.data_config import PartitioningConfig
        cfg = PartitioningConfig(min_per_client=0)
        with pytest.raises(ValueError, match="min_per_client"):
            cfg.validate()


# ---------------------------------------------------------------------------
# FederatedConfig validation
# ---------------------------------------------------------------------------

class TestFederatedConfig:
    def test_valid_config(self):
        from fedmammobench.configs.federated_config import FederatedConfig
        cfg = FederatedConfig(num_clients=4, rounds=10, min_fit_clients=2)
        cfg.validate()

    def test_min_fit_clients_exceeds_num_clients_raises(self):
        from fedmammobench.configs.federated_config import FederatedConfig
        cfg = FederatedConfig(num_clients=2, min_fit_clients=4)
        with pytest.raises(ValueError, match="min_fit_clients"):
            cfg.validate()

    def test_rounds_zero_raises(self):
        from fedmammobench.configs.federated_config import FederatedConfig
        cfg = FederatedConfig(rounds=0)
        with pytest.raises(ValueError, match="rounds"):
            cfg.validate()

    def test_model_config_hash_deterministic(self):
        from fedmammobench.configs.federated_config import FederatedConfig
        fed = FederatedConfig()
        fields = {"name": "resnet50", "num_classes": 2, "in_channels": 1, "dropout": 0.2}
        h1 = fed.model_config_hash(fields)
        h2 = fed.model_config_hash(fields)
        assert h1 == h2
        assert len(h1) == 16

    def test_model_config_hash_differs_on_change(self):
        from fedmammobench.configs.federated_config import FederatedConfig
        fed = FederatedConfig()
        fields_a = {"name": "resnet50", "num_classes": 2, "in_channels": 1, "dropout": 0.2}
        fields_b = {"name": "resnet18", "num_classes": 2, "in_channels": 1, "dropout": 0.2}
        assert fed.model_config_hash(fields_a) != fed.model_config_hash(fields_b)


# ---------------------------------------------------------------------------
# TrainingConfig validation
# ---------------------------------------------------------------------------

class TestTrainingConfig:
    def test_valid_config(self):
        from fedmammobench.configs.training_config import TrainingConfig
        cfg = TrainingConfig(epochs=10, local_epochs=1)
        cfg.validate()

    def test_epochs_zero_raises(self):
        from fedmammobench.configs.training_config import TrainingConfig
        cfg = TrainingConfig(epochs=0)
        with pytest.raises(ValueError, match="epochs"):
            cfg.validate()

    def test_fedprox_amp_warns(self):
        from fedmammobench.configs.training_config import TrainingConfig
        cfg = TrainingConfig(epochs=5, mixed_precision=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg.validate(strategy_name="fedprox", proximal_mu=0.01)
        assert any("FedProx" in str(warning.message) for warning in w)

    def test_fedprox_no_amp_no_warn(self):
        from fedmammobench.configs.training_config import TrainingConfig
        cfg = TrainingConfig(epochs=5, mixed_precision=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg.validate(strategy_name="fedprox", proximal_mu=0.01)
        assert not any("FedProx" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# ExperimentConfig.validate() cross-section checks
# ---------------------------------------------------------------------------

class TestExperimentConfigValidate:
    def test_full_valid_config_no_raise(self):
        from fedmammobench.configs.experiment import ExperimentConfig
        cfg = ExperimentConfig()
        cfg.validate()

    def test_channel_preset_mismatch_raises(self):
        from fedmammobench.configs.experiment import ExperimentConfig
        from fedmammobench.configs.model_config import ModelConfig
        from fedmammobench.configs.training_config import TrainingConfig, AugmentationConfig
        cfg = ExperimentConfig()
        cfg.model = ModelConfig(name="resnet50", in_channels=3)
        cfg.training = TrainingConfig()
        cfg.training.augmentation = AugmentationConfig(normalize_preset="radimagenet_gray")
        with pytest.raises(ValueError, match="in_channels"):
            cfg.validate()

    def test_unfreeze_at_epoch_beyond_rounds_warns(self):
        from fedmammobench.configs.experiment import ExperimentConfig
        from fedmammobench.configs.model_config import ModelConfig
        from fedmammobench.configs.federated_config import FederatedConfig
        cfg = ExperimentConfig(mode="federated")
        cfg.model = ModelConfig(freeze_backbone=True, unfreeze_at_epoch=50)
        cfg.federated = FederatedConfig(rounds=10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cfg.validate()
        assert any("unfreeze_at_epoch" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# C1: Patient leakage fix — validate train/val patient disjointness
# ---------------------------------------------------------------------------

class TestC1PatientSplitNoLeakage:
    """Verify that when manifest has train/test-only labels, val is carved
    at patient level (no patient appears in both train and val)."""

    def _make_manifest_csv(self, tmp_path: Path) -> Path:
        """Create a minimal CSV with train/test split only."""
        content = (
            "image_path,pathology,patient_id,split\n"
            "img001.png,MALIGNANT,P001,train\n"
            "img002.png,BENIGN,P001,train\n"
            "img003.png,MALIGNANT,P002,train\n"
            "img004.png,BENIGN,P002,train\n"
            "img005.png,MALIGNANT,P003,train\n"
            "img006.png,BENIGN,P003,train\n"
            "img007.png,MALIGNANT,P004,train\n"
            "img008.png,BENIGN,P004,train\n"
            "img009.png,MALIGNANT,P005,test\n"
            "img010.png,BENIGN,P005,test\n"
        )
        csv_path = tmp_path / "manifest.csv"
        csv_path.write_text(content)
        return csv_path

    def test_no_patient_overlap_train_val(self, tmp_path):
        """Train and val must be patient-disjoint after C1 fix."""
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")

        from fedmammobench.datasets.cbis_ddsm import CBISDDSMDataset
        from fedmammobench.configs.schema import DataColumnMapping

        csv_path = self._make_manifest_csv(tmp_path)
        # Create minimal image stubs so the loader doesn't drop them
        for i in range(1, 11):
            (tmp_path / f"img{i:03d}.png").write_bytes(b"")

        # We test at the split level — the actual image loading is irrelevant
        # for the split logic, so we call the internal function directly.
        import numpy as np
        from fedmammobench.datasets.cbis_ddsm import _stratified_patient_split
        from fedmammobench.datasets.base import Sample

        samples = [
            Sample(image_path=f"img{i:03d}.png", label=i % 2, patient_id=f"P{(i-1)//2 + 1:03d}")
            for i in range(1, 9)  # 8 train samples, 4 patients
        ]
        splits = _stratified_patient_split(samples, val_fraction=0.25, test_fraction=0.0, seed=42)
        train_pids = {samples[i].patient_id for i in splits["train"]}
        val_pids = {samples[i].patient_id for i in splits["val"]}
        assert train_pids.isdisjoint(val_pids), (
            f"Patient leakage detected: {train_pids & val_pids} appear in both train and val"
        )


# ---------------------------------------------------------------------------
# Backward compatibility: schema.py still exports everything
# ---------------------------------------------------------------------------

class TestSchemaBackwardCompat:
    def test_all_classes_importable_from_schema(self):
        from fedmammobench.configs.schema import (
            AugmentationConfig,
            DataColumnMapping,
            DataConfig,
            EvaluationConfig,
            ExperimentConfig,
            FederatedConfig,
            LossConfig,
            ModelConfig,
            NORMALIZE_PRESETS,
            OptimizerConfig,
            PartitioningConfig,
            SchedulerConfig,
            StrategyConfig,
            TrainingConfig,
        )
        assert ExperimentConfig is not None
        assert isinstance(NORMALIZE_PRESETS, dict)

    def test_experiment_config_from_schema(self):
        from fedmammobench.configs.schema import ExperimentConfig
        cfg = ExperimentConfig(name="test", seed=0)
        assert cfg.name == "test"
