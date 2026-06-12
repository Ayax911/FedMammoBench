"""Tests for hybrid server-side training (federated.server_training).

The central node owns a local manifest and continues training from the
aggregated weights each round. These tests run CPU-only on a tiny on-disk
fixture and skip when torch / albumentations are unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _write_tiny_manifest(root: Path, *, n_patients: int = 8, per_patient: int = 4) -> tuple[str, str]:
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import csv

    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for p in range(n_patients):
        pathology = "BENIGN" if p % 2 == 0 else "MALIGNANT"
        for k in range(per_patient):
            arr = rng.integers(0, 256, size=(32, 32), dtype="uint8")
            fname = f"p{p:02d}_{k}.png"
            Image.fromarray(arr).save(img_dir / fname)
            rows.append(
                {"image_path": f"images/{fname}", "pathology": pathology, "patient_id": f"P{p:02d}"}
            )
    manifest = root / "server.csv"
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_path", "pathology", "patient_id"])
        w.writeheader()
        w.writerows(rows)
    return str(manifest), str(img_dir.parent)


def _cfg_with_server_training(manifest: str, image_root: str):
    from fedmammobench.configs.schema import (
        DataConfig,
        ExperimentConfig,
        FederatedConfig,
        ModelConfig,
        ServerTrainingConfig,
        StrategyConfig,
        TrainingConfig,
    )

    return ExperimentConfig(
        name="server_train_smoke",
        mode="federated",
        seed=42,
        device="cpu",
        data=DataConfig(name="cbis_ddsm", image_size=32, batch_size=8, num_workers=0),
        model=ModelConfig(name="resnet18", pretrained=False, in_channels=1),
        training=TrainingConfig(local_epochs=1),
        federated=FederatedConfig(
            num_clients=2,
            rounds=1,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            strategy=StrategyConfig(name="fedavg"),
            server_training=ServerTrainingConfig(
                enabled=True,
                manifest_path=manifest,
                image_root=image_root,
                local_epochs=1,
                server_weight=1.0,
            ),
        ),
    )


def test_server_training_config_validation() -> None:
    from fedmammobench.configs.schema import ServerTrainingConfig

    # enabled but missing paths → error
    with pytest.raises(ValueError, match="manifest_path"):
        ServerTrainingConfig(enabled=True).validate()
    # bad weight
    with pytest.raises(ValueError, match="server_weight"):
        ServerTrainingConfig(
            enabled=True, manifest_path="m", image_root="i", server_weight=1.5
        ).validate()
    # disabled → no-op even with nothing set
    ServerTrainingConfig(enabled=False).validate()


def test_server_trainer_updates_weights(tmp_path) -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    import numpy as np

    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.federated.server_training import ServerTrainer
    from fedmammobench.models import build_model

    manifest, image_root = _write_tiny_manifest(tmp_path)
    cfg = _cfg_with_server_training(manifest, image_root)
    cfg.validate()

    template = build_model(cfg.model)
    init = state_dict_to_ndarrays(template)

    trainer = ServerTrainer(cfg)
    out = trainer.train(init, server_round=1)

    assert len(out) == len(init)
    diffs = [np.linalg.norm(a - b) for a, b in zip(init, out, strict=True)]
    assert any(d > 1e-8 for d in diffs), "server training must change the weights"
    assert trainer.last_loss == trainer.last_loss, "last_loss should be a real number"


def test_attach_server_training_wraps_aggregate_fit(tmp_path) -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.federated.server_training import (
        ServerTrainer,
        attach_server_training,
    )
    from fedmammobench.models import build_model

    manifest, image_root = _write_tiny_manifest(tmp_path)
    cfg = _cfg_with_server_training(manifest, image_root)

    template = build_model(cfg.model)
    init_params = ndarrays_to_parameters(state_dict_to_ndarrays(template))

    class DummyStrategy:
        def aggregate_fit(self, server_round, results, failures):
            # Pretend aggregation returned the initial params unchanged.
            return init_params, {"agg_metric": 1.0}

    strategy = DummyStrategy()
    trainer = ServerTrainer(cfg)
    attach_server_training(strategy, trainer, server_weight=1.0)

    new_params, metrics = strategy.aggregate_fit(1, [], [])
    assert "server_train_loss" in metrics
    assert metrics["agg_metric"] == 1.0  # original metrics preserved
    # Parameters should differ after the server training step.
    import numpy as np

    before = parameters_to_ndarrays(init_params)
    after = parameters_to_ndarrays(new_params)
    diffs = [np.linalg.norm(a - b) for a, b in zip(before, after, strict=True)]
    assert any(d > 1e-8 for d in diffs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
