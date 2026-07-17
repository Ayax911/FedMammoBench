"""Centralized (non-federated) training baseline.

Usage::

    python scripts/run_centralized.py --config configs/radimagenet_resnet50_centralized.yaml

Loads the config, builds a single train/val/test pipeline, trains for
``training.epochs`` epochs, and saves metrics CSV + TensorBoard logs under
``<output_dir>/<name>/``. The final checkpoint goes to ``<output_dir>/weights/``
instead, a sibling of ``<name>/`` and ``eval/``, so it can be excluded from a
sync/upload of the rest of the run by folder alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    """Make ``src/fedmammobench`` importable when running without ``pip install``."""
    here = Path(__file__).resolve()
    src = here.parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

from fedmammobench.configs import load_config, save_config  # noqa: E402
from fedmammobench.datasets import build_dataloader, build_dataset  # noqa: E402
from fedmammobench.evaluation import Evaluator  # noqa: E402
from fedmammobench.models import build_model  # noqa: E402
from fedmammobench.training import Trainer, build_loss, build_optimizer, build_scheduler  # noqa: E402
from fedmammobench.utils import (  # noqa: E402
    CSVLogger,
    TensorBoardWriter,
    get_logger,
    resolve_device,
    save_checkpoint,
    set_global_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centralized training entrypoint.")
    p.add_argument("--config", "-c", required=True, type=str, help="Path to a YAML config.")
    p.add_argument("--output-dir", type=str, default=None, help="Override cfg.output_dir.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.mode != "centralized":
        print(
            f"Warning: cfg.mode={cfg.mode!r} but you launched the centralized script. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

    out_root = Path(args.output_dir or Path(cfg.output_dir) / cfg.name).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out_root / "run.log")
    logger = get_logger("centralized")

    set_global_seed(cfg.seed, deterministic=True)
    save_config(cfg, out_root / "config.snapshot.yaml")

    device = resolve_device(cfg.device)
    logger.info("Device: %s", device)

    datasets = build_dataset(cfg)
    train_loader = build_dataloader(
        datasets["train"],
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
        balance_classes=cfg.data.balance_classes,
        pin_memory=(device.type == "cuda"),
        seed=cfg.seed,
    )
    val_loader = build_dataloader(
        datasets["val"],
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    test_loader = build_dataloader(
        datasets["test"],
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )

    model = build_model(cfg.model).to(device)
    criterion = build_loss(
        cfg.training.loss,
        train_labels=datasets["train"].labels,
        num_classes=cfg.model.num_classes,
    ).to(device)
    optimizer = build_optimizer(model, cfg.training.optimizer)
    scheduler = build_scheduler(optimizer, cfg.training.scheduler)

    tb_writer = TensorBoardWriter(out_root / "tb")
    csv_logger = CSVLogger(out_root / "metrics.csv")
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        scheduler=scheduler,
        grad_clip_norm=cfg.training.grad_clip_norm,
        mixed_precision=cfg.training.mixed_precision,
        tb_writer=tb_writer,
        csv_logger=csv_logger,
        log_tag="centralized",
    )
    evaluator = Evaluator(model, device=device, threshold=cfg.evaluation.threshold)

    trainer.fit(
        train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        epochs=cfg.training.epochs,
    )

    # Weights live in a "weights/" sibling of out_root (normally
    # <output_dir>/<name>/), not inside it, so the (large) checkpoint can be
    # excluded from a sync/upload of the rest of the run's metrics/logs by
    # folder alone. Derived from out_root (not cfg.output_dir) so it stays
    # correct even when --output-dir overrides the default.
    weights_dir = out_root.parent / "weights"
    save_checkpoint(weights_dir / "final.pt", model, optimizer=optimizer, epoch=cfg.training.epochs)

    test_metrics = evaluator.evaluate(test_loader, criterion=criterion)
    logger.info("Test metrics: %s", {k: v for k, v in test_metrics.items() if k != "y_true"})
    scalar_test = {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}
    test_csv = CSVLogger(out_root / "test_metrics.csv")
    test_csv.append({"epoch": -1, "phase": "test", **{f"test_{k}": v for k, v in scalar_test.items()}})

    tb_writer.close()
    logger.info("Run complete. Artifacts at %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
