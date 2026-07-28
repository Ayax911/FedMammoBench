"""Centralized (non-federated) training baseline.

Usage::

    python scripts/run_centralized.py --config configs/radimagenet_resnet50_centralized.yaml

Loads the config, builds a single train/val/test pipeline, trains for
``training.epochs`` epochs, and saves metrics CSV + TensorBoard logs under
``<output_dir>/<name>/``. Checkpoints go to ``<output_dir>/weights/`` instead,
a sibling of ``<name>/`` and ``eval/``, so they can be excluded from a
sync/upload of the rest of the run by folder alone: always ``final.pt`` (last
epoch), plus ``best.pt`` (best ``val_<training.best_checkpoint_metric>``) when
``training.save_best_checkpoint`` is set — in which case test evaluation uses
``best.pt``, not the last epoch's weights.
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
    build_metric_sink,
    get_logger,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_global_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Centralized training entrypoint.")
    p.add_argument("--config", "-c", required=True, type=str, help="Path to a YAML config.")
    p.add_argument("--output-dir", type=str, default=None, help="Override cfg.output_dir.")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from weights/final.pt if present (recovers model+optimizer+"
            "scheduler state and continues until training.epochs total). If no "
            "checkpoint exists yet, starts fresh but writes a per-epoch safety-net "
            "checkpoint to weights/final.pt so a later crash can be recovered from "
            "by re-running the same command with --resume again."
        ),
    )
    return p.parse_args()


def _resolve_centralized_resume(
    cfg, final_ckpt_path: Path, *, resume: bool, model, optimizer, scheduler
) -> tuple[int, int, dict | None]:
    """Returns ``(start_epoch, remaining_epochs, resume_state)``.

    When a checkpoint is found, loads model/optimizer/scheduler state into
    the given objects IN PLACE via ``load_checkpoint``. ``cfg.training.epochs``
    is treated as the TOTAL budget across the original run + any resumes, not
    "how many more" — so ``remaining_epochs`` can be <= 0 if the checkpoint
    already reached (or exceeds, e.g. after shrinking the config) that total.

    Known limitation: ``--resume`` combined with ``model.unfreeze_at_epoch``
    (progressive backbone unfreeze) is only safe when the crash happened
    BEFORE the unfreeze threshold. ``optimizer`` here is freshly built from a
    freshly-built (still fully frozen) ``model``, so it has only the original
    param group(s). If the crash happened AFTER
    ``Trainer._apply_progressive_unfreeze`` had already added a param group
    (e.g. for ``layer4``), ``optimizer.load_state_dict`` below raises
    ``ValueError`` (saved vs. current param-group count mismatch) instead of
    silently restoring the wrong state — a loud failure, not silent
    corruption, but resuming past that point currently requires dropping
    ``--resume`` and accepting the lost partial progress.
    """
    if not resume or not final_ckpt_path.is_file():
        return 0, cfg.training.epochs, None
    payload = load_checkpoint(
        final_ckpt_path, model, optimizer=optimizer, scheduler=scheduler, strict=True
    )
    # Absolute 0-based index of the last epoch that fully completed — NOT a
    # count (see Trainer.fit's resume_checkpoint_path docstring: the in-loop
    # safety net always stores the raw loop variable `epoch`, matching
    # best.pt's existing convention, not the epochs_run count final.pt used
    # to store pre-resume).
    resumed_epoch = int(payload.get("epoch", -1))
    start_epoch = resumed_epoch + 1
    return start_epoch, cfg.training.epochs - start_epoch, payload.get("extra", {}).get("resume_state")


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
    # Resolved once here (not inside Trainer) so Trainer stays decoupled from
    # OptimizerConfig — only needs the resulting float for newly-unfrozen
    # params (e.g. layer4 crossing model.unfreeze_at_epoch).
    unfreeze_lr = (
        cfg.training.optimizer.lr_backbone
        if cfg.training.optimizer.lr_backbone is not None
        else cfg.training.optimizer.lr
    )

    # sink fans out to TensorBoard + (if cfg.wandb.enabled) W&B — every
    # Trainer.log_scalars(...) call below reaches both with no further
    # changes, since MetricSink duck-types as a TensorBoardWriter. See
    # utils/metrics_sink.py.
    sink = build_metric_sink(cfg, out_root, job_type="centralized")
    csv_logger = CSVLogger(out_root / "metrics.csv")
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        scheduler=scheduler,
        grad_clip_norm=cfg.training.grad_clip_norm,
        mixed_precision=cfg.training.mixed_precision,
        tb_writer=sink,
        csv_logger=csv_logger,
        log_tag="centralized",
    )
    evaluator = Evaluator(model, device=device, threshold=cfg.evaluation.threshold)

    # Weights live in a "weights/" sibling of out_root (normally
    # <output_dir>/<name>/), not inside it, so the (large) checkpoint can be
    # excluded from a sync/upload of the rest of the run's metrics/logs by
    # folder alone. Derived from out_root (not cfg.output_dir) so it stays
    # correct even when --output-dir overrides the default.
    weights_dir = out_root.parent / "weights"
    best_ckpt_path = weights_dir / "best.pt"
    final_ckpt_path = weights_dir / "final.pt"

    start_epoch, remaining_epochs, resume_state = _resolve_centralized_resume(
        cfg, final_ckpt_path, resume=args.resume, model=model, optimizer=optimizer, scheduler=scheduler
    )
    if args.resume:
        logger.info(
            "Resume: start_epoch=%d remaining_epochs=%d (checkpoint %s)",
            start_epoch,
            remaining_epochs,
            "found" if final_ckpt_path.is_file() else "not found — starting fresh",
        )

    # Wrapped in try/finally so sink.close() always runs — previously
    # tb_writer.close() was unconditional at the end of main(), which meant
    # an exception in trainer.fit()/evaluator.evaluate() leaked the TB
    # writer. That was a minor bug on its own; it becomes load-bearing now
    # that `sink` can hold a live W&B run — an unclosed run never calls
    # `finish()` and stays "running" forever in the W&B UI. autoplot() runs
    # in the same finally, after the sink is closed so metrics.csv/
    # test_metrics.csv are fully flushed to disk.
    try:
        if remaining_epochs > 0:
            fit_result = trainer.fit(
                train_loader,
                val_loader=val_loader,
                evaluator=evaluator,
                epochs=remaining_epochs,
                start_epoch=start_epoch,
                best_checkpoint_metric=(
                    cfg.training.best_checkpoint_metric if cfg.training.save_best_checkpoint else None
                ),
                best_checkpoint_path=(best_ckpt_path if cfg.training.save_best_checkpoint else None),
                early_stopping_patience=cfg.training.early_stopping_patience,
                resume_checkpoint_path=(final_ckpt_path if args.resume else None),
                resume_state=resume_state,
                model_cfg=cfg.model,
                unfreeze_lr=unfreeze_lr,
            )
        else:
            logger.info(
                "training.epochs (%d) already reached by checkpoint (next epoch would "
                "be %d) — skipping fit().",
                cfg.training.epochs,
                start_epoch,
            )
            fit_result = {"epochs_run": 0, "early_stopped": False}
            if cfg.training.save_best_checkpoint and resume_state is not None:
                fit_result["best_epoch"] = resume_state.get("best_epoch")
                fit_result[f"best_val_{cfg.training.best_checkpoint_metric}"] = resume_state.get(
                    "best_value"
                )

        if not args.resume:
            save_checkpoint(
                final_ckpt_path,
                model,
                optimizer=optimizer,
                epoch=fit_result.get("epochs_run", cfg.training.epochs),
            )
        # else: Trainer.fit's in-loop safety net (resume_checkpoint_path) already
        # wrote final_ckpt_path with strictly more information (scheduler state +
        # resume_state) whenever it ran; re-saving here would regress it back to a
        # weaker checkpoint. If it didn't run (remaining_epochs <= 0), the existing
        # checkpoint on disk is already exactly what we just loaded from.

        # Trainer.fit() left `model` at its last-epoch weights. If best-checkpoint
        # tracking was enabled, reload the checkpoint that scored highest on
        # val_{best_checkpoint_metric} before evaluating on test — otherwise a run
        # that overfits past its optimum (common for small/frozen-backbone heads)
        # would report test metrics for a model worse than one seen mid-training.
        checkpoint_used = "final"
        if cfg.training.save_best_checkpoint and best_ckpt_path.is_file():
            load_checkpoint(best_ckpt_path, model, strict=True)
            best_epoch = fit_result.get("best_epoch")
            best_value = fit_result.get(f"best_val_{cfg.training.best_checkpoint_metric}")
            checkpoint_used = f"best_epoch_{best_epoch}"
            logger.info(
                "Loaded best checkpoint for test evaluation: epoch=%s %s=%s",
                best_epoch,
                cfg.training.best_checkpoint_metric,
                best_value,
            )

        test_metrics = evaluator.evaluate(test_loader, criterion=criterion)
        logger.info("Test metrics: %s", {k: v for k, v in test_metrics.items() if k != "y_true"})
        scalar_test = {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))}
        test_csv = CSVLogger(out_root / "test_metrics.csv")
        test_csv.append(
            {
                "epoch": -1,
                "phase": "test",
                "checkpoint": checkpoint_used,
                "early_stopped": fit_result.get("early_stopped", False),
                **{f"test_{k}": v for k, v in scalar_test.items()},
            }
        )
    finally:
        sink.close()
        from fedmammobench.plotting import autoplot  # lazy: matplotlib isn't a hard dep

        autoplot(out_root)

    logger.info("Run complete. Artifacts at %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
