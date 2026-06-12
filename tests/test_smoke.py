"""End-to-end smoke tests on a tiny on-disk image fixture.

These tests cover the full graph (data -> model -> Trainer -> Evaluator) and
the federated client interface (without launching a Flower simulation, which
adds Ray bring-up cost). A handful of grayscale PNGs plus a CSV manifest are
written to a temp directory and loaded through the CBIS-DDSM loader, so the
suite needs no real datasets and no synthetic stand-in. Runs in well under a
minute on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow running ``pytest tests/`` without ``pip install -e .``.
SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ---------------------------------------------------------------------------
# Tiny on-disk fixture: grayscale PNGs + CBIS-DDSM-style CSV manifest
# ---------------------------------------------------------------------------

def _write_tiny_dataset(
    root: Path,
    *,
    n_patients: int = 12,
    per_patient: int = 4,
    image_size: int = 32,
    all_benign: bool = False,
) -> tuple[str, str]:
    """Write PNGs + a manifest under ``root``; return (manifest_path, image_root).

    Patients are split benign/malignant so labels are balanced unless
    ``all_benign`` is set (used to force a single-class set for NaN-metric tests).
    """
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import csv

    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    for p in range(n_patients):
        pathology = "BENIGN" if (all_benign or p % 2 == 0) else "MALIGNANT"
        for k in range(per_patient):
            arr = rng.integers(0, 256, size=(image_size, image_size), dtype="uint8")
            fname = f"p{p:02d}_{k}.png"
            Image.fromarray(arr).save(img_dir / fname)  # 2D uint8 → mode "L"
            rows.append(
                {"image_path": f"images/{fname}", "pathology": pathology, "patient_id": f"P{p:02d}"}
            )

    manifest = root / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image_path", "pathology", "patient_id"])
        writer.writeheader()
        writer.writerows(rows)
    return str(manifest), str(img_dir.parent)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> tuple[str, str]:
    """A balanced, multi-patient on-disk fixture shared across the module."""
    root = tmp_path_factory.mktemp("tiny_ds")
    return _write_tiny_dataset(root)


def _make_cfg(
    manifest_path: str,
    image_root: str,
    *,
    image_size: int = 32,
    num_clients: int = 2,
):
    from fedmammobench.configs.schema import (
        AugmentationConfig,
        DataConfig,
        ExperimentConfig,
        FederatedConfig,
        LossConfig,
        ModelConfig,
        OptimizerConfig,
        PartitioningConfig,
        SchedulerConfig,
        StrategyConfig,
        TrainingConfig,
    )

    cfg = ExperimentConfig(
        name="smoke",
        mode="federated",
        seed=42,
        device="cpu",
        data=DataConfig(
            name="cbis_ddsm",
            manifest_path=manifest_path,
            image_root=image_root,
            image_size=image_size,
            grayscale=True,
            num_classes=2,
            batch_size=8,
            num_workers=0,
            val_fraction=0.2,
            test_fraction=0.2,
            balance_classes=True,
        ),
        partitioning=PartitioningConfig(
            scheme="dirichlet", alpha=0.5, min_per_client=2, max_retries=50
        ),
        model=ModelConfig(name="resnet18", pretrained=False, in_channels=1),
        training=TrainingConfig(
            epochs=1,
            local_epochs=1,
            optimizer=OptimizerConfig(name="adamw", lr=1e-3),
            scheduler=SchedulerConfig(name="none"),
            augmentation=AugmentationConfig(),
            loss=LossConfig(name="ce", auto_class_weights=True),
        ),
        federated=FederatedConfig(
            num_clients=num_clients,
            rounds=1,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            strategy=StrategyConfig(name="fedavg"),
        ),
    )
    return cfg


def test_imports() -> None:
    """Importing the package and submodules must not raise."""
    import fedmammobench  # noqa: F401
    import fedmammobench.configs  # noqa: F401
    import fedmammobench.datasets  # noqa: F401
    import fedmammobench.evaluation  # noqa: F401
    import fedmammobench.federated  # noqa: F401
    import fedmammobench.models  # noqa: F401
    import fedmammobench.training  # noqa: F401
    import fedmammobench.utils  # noqa: F401


def test_dataset_registry_populated() -> None:
    """The built-in loaders must self-register via @register_dataset."""
    from fedmammobench.datasets import list_datasets

    names = list_datasets()
    for expected in ("cbis_ddsm", "mammo_bench", "vindr_mammo"):
        assert expected in names, f"{expected} missing from registry: {names}"


def test_dataset_fixture_shape(tiny_dataset) -> None:
    pytest.importorskip("albumentations")
    pytest.importorskip("torch")
    from fedmammobench.datasets import build_dataset

    manifest, image_root = tiny_dataset
    cfg = _make_cfg(manifest, image_root, image_size=32)
    datasets = build_dataset(cfg)
    assert set(datasets) == {"train", "val", "test"}
    x, y = datasets["train"][0]
    assert x.shape == (1, 32, 32), f"unexpected shape {x.shape}"
    assert int(y) in (0, 1)


def test_partitioning_iid_and_dirichlet() -> None:
    import numpy as np

    from fedmammobench.datasets.partitioning import partition_indices

    labels = np.repeat([0, 1], 32)
    iid = partition_indices(labels, num_clients=4, scheme="iid", seed=1)
    assert len(iid) == 4
    assert sum(len(c) for c in iid) == labels.size

    dir_parts = partition_indices(
        labels, num_clients=4, scheme="dirichlet", alpha=0.5, min_per_client=4, seed=1
    )
    assert len(dir_parts) == 4
    assert min(len(c) for c in dir_parts) >= 4


def test_centralized_one_epoch_cpu(tiny_dataset) -> None:
    """End-to-end: build dataset, model, trainer, run one epoch on CPU."""
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")

    from fedmammobench.datasets import build_dataloader, build_dataset
    from fedmammobench.evaluation import Evaluator
    from fedmammobench.models import build_model
    from fedmammobench.training import Trainer, build_loss, build_optimizer

    manifest, image_root = tiny_dataset
    cfg = _make_cfg(manifest, image_root, image_size=32)
    datasets = build_dataset(cfg)
    train_loader = build_dataloader(datasets["train"], batch_size=8, num_workers=0)
    val_loader = build_dataloader(
        datasets["val"], batch_size=8, num_workers=0, shuffle=False
    )

    device = torch.device("cpu")
    model = build_model(cfg.model).to(device)
    criterion = build_loss(cfg.training.loss, train_labels=datasets["train"].labels).to(device)
    optimizer = build_optimizer(model, cfg.training.optimizer)
    trainer = Trainer(model, optimizer, criterion, device, log_tag="smoke")
    evaluator = Evaluator(model, device=device)

    stats = trainer.train_one_epoch(train_loader, epoch=0)
    assert stats["samples"] > 0
    metrics = evaluator.evaluate(val_loader, criterion=criterion)
    for key in ("accuracy", "precision", "recall", "f1", "sensitivity", "specificity"):
        assert key in metrics


def test_federated_client_fit_evaluate(tiny_dataset) -> None:
    """Exercise FedMammoBenchClient.fit / .evaluate without spinning up Flower."""
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")

    from fedmammobench.datasets import build_dataset, partition_indices
    from fedmammobench.federated.client import FedMammoBenchClient
    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.models import build_model

    manifest, image_root = tiny_dataset
    cfg = _make_cfg(manifest, image_root, image_size=32)
    datasets = build_dataset(cfg)
    parts = partition_indices(
        datasets["train"].labels, num_clients=2, scheme="iid", seed=0
    )
    sub_train = datasets["train"].subset(parts[0])
    device = torch.device("cpu")

    client = FedMammoBenchClient(
        client_id=0,
        cfg=cfg,
        train_dataset=sub_train,
        val_dataset=datasets["val"],
        device=device,
    )

    # Pull initial params from a fresh model — server pushes them in real FL.
    template = build_model(cfg.model).to(device)
    init_params = state_dict_to_ndarrays(template)

    new_params, n_samples, fit_metrics = client.fit(init_params, {"local_epochs": 1})
    assert len(new_params) == len(init_params)
    assert n_samples > 0
    assert "train_loss" in fit_metrics

    loss, n_eval, eval_metrics = client.evaluate(new_params, {})
    assert n_eval > 0
    assert "roc_auc" in eval_metrics or "accuracy" in eval_metrics
    assert isinstance(loss, float)


def test_partitioning_patient_disjoint() -> None:
    """All images of a patient must land on exactly one client."""
    import numpy as np
    from fedmammobench.datasets.partitioning import partition_indices

    # 8 patients × 3 images = 24 samples; binary labels, balanced.
    n_patients = 8
    per_patient = 3
    patient_ids = np.repeat([f"P{i}" for i in range(n_patients)], per_patient)
    labels = np.tile([0, 1], (n_patients * per_patient) // 2 + 1)[: n_patients * per_patient]

    for scheme in ("iid", "dirichlet", "quantity_skew"):
        parts = partition_indices(
            labels,
            num_clients=2,
            patient_ids=patient_ids,
            scheme=scheme,
            alpha=1.0,
            min_per_client=1,
            max_retries=50,
            seed=42,
        )
        assert len(parts) == 2
        assert sum(len(p) for p in parts) == len(labels), f"{scheme}: total index mismatch"

        client_patients = [set(patient_ids[i] for i in p) for p in parts]
        for ci in range(len(client_patients)):
            for cj in range(ci + 1, len(client_patients)):
                overlap = client_patients[ci] & client_patients[cj]
                assert not overlap, (
                    f"scheme={scheme}: patient overlap between client {ci} and {cj}: {overlap}"
                )


def test_fedprox_proximal_term_applied(tiny_dataset) -> None:
    """FedProx with mu>0 must produce different updates than FedAvg (mu=0)."""
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")
    import numpy as np

    from fedmammobench.datasets import build_dataset, partition_indices
    from fedmammobench.federated.client import FedMammoBenchClient
    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.models import build_model

    manifest, image_root = tiny_dataset
    cfg = _make_cfg(manifest, image_root, image_size=32)
    datasets = build_dataset(cfg)
    parts = partition_indices(datasets["train"].labels, num_clients=2, scheme="iid", seed=0)
    sub_train = datasets["train"].subset(parts[0])
    device = torch.device("cpu")

    template = build_model(cfg.model).to(device)
    init_params = state_dict_to_ndarrays(template)

    # FedAvg: mu=0
    client_avg = FedMammoBenchClient(0, cfg, sub_train, None, device)
    params_avg, _, _ = client_avg.fit(init_params, {"local_epochs": 1, "proximal_mu": 0.0})

    # FedProx: strong mu to guarantee visible difference
    client_prox = FedMammoBenchClient(0, cfg, sub_train, None, device)
    params_prox, _, _ = client_prox.fit(init_params, {"local_epochs": 1, "proximal_mu": 10.0})

    diffs = [np.linalg.norm(a - b) for a, b in zip(params_avg, params_prox, strict=True)]
    assert any(d > 1e-6 for d in diffs), (
        "FedProx with mu=10.0 must produce different parameters than FedAvg"
    )


def test_evaluate_nan_metric_omitted_not_zero(tiny_dataset) -> None:
    """NaN metrics (e.g. roc_auc on single-class val set) must be omitted, not 0.0."""
    pytest.importorskip("albumentations")
    torch = pytest.importorskip("torch")

    from fedmammobench.datasets import build_dataset, partition_indices
    from fedmammobench.federated.client import FedMammoBenchClient
    from fedmammobench.federated.param_utils import state_dict_to_ndarrays
    from fedmammobench.models import build_model

    manifest, image_root = tiny_dataset
    cfg = _make_cfg(manifest, image_root, image_size=32)
    datasets = build_dataset(cfg)
    parts = partition_indices(datasets["train"].labels, num_clients=2, scheme="iid", seed=0)
    sub_train = datasets["train"].subset(parts[0])
    device = torch.device("cpu")

    # Force a single-class val set so roc_auc is NaN: keep only benign samples.
    train = datasets["train"]
    benign_idx = [i for i, lbl in enumerate(train.labels.tolist()) if lbl == 0]
    assert benign_idx, "fixture must contain benign samples"
    mono_val = train.subset(benign_idx)
    assert len(set(mono_val.labels.tolist())) == 1, "val set must be single-class"

    template = build_model(cfg.model).to(device)
    init_params = state_dict_to_ndarrays(template)

    client = FedMammoBenchClient(0, cfg, sub_train, mono_val, device)
    _, _, eval_metrics = client.evaluate(init_params, {})

    # roc_auc must not be present (NaN → omitted) or, if present, must not be 0.0
    assert "roc_auc" not in eval_metrics or eval_metrics["roc_auc"] != 0.0, (
        "roc_auc=0.0 indicates the NaN-to-zero bug is still present"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
