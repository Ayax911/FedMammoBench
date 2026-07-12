"""Standalone evaluation of a checkpoint against a configured test set.

Usage::

    python scripts/run_evaluation.py \\
        --config configs/radimagenet_resnet50_centralized.yaml \\
        --checkpoint runs/radimagenet_resnet50_centralized/final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fedmammobench.configs import load_config  # noqa: E402
from fedmammobench.datasets import build_dataloader, build_dataset  # noqa: E402
from fedmammobench.evaluation import Evaluator  # noqa: E402
from fedmammobench.models import build_model  # noqa: E402
from fedmammobench.training import build_loss  # noqa: E402
from fedmammobench.utils import (  # noqa: E402
    get_logger,
    load_checkpoint,
    resolve_device,
    set_global_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    p.add_argument("--config", "-c", required=True, type=str)
    p.add_argument("--checkpoint", "-k", required=True, type=str)
    p.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Which split to evaluate.",
    )
    p.add_argument(
        "--predictions-out",
        default=None,
        type=str,
        help="Optional CSV path to dump per-sample predictions.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        type=str,
        help="Optional output directory for run.log and metrics.json.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    # Setup logging to file if output_dir is provided
    log_file = None
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "run.log"

    setup_logging(log_file=log_file)
    logger = get_logger("evaluate")
    set_global_seed(cfg.seed, deterministic=True)

    device = resolve_device(cfg.device)
    datasets = build_dataset(cfg)
    if args.split not in datasets or len(datasets[args.split]) == 0:
        logger.error("Split %r is empty or absent.", args.split)
        return 2

    loader = build_dataloader(
        datasets[args.split],
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
    )
    model = build_model(cfg.model).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)

    criterion = build_loss(
        cfg.training.loss,
        train_labels=datasets["train"].labels,
        num_classes=cfg.model.num_classes,
    ).to(device)
    evaluator = Evaluator(model, device=device, threshold=cfg.evaluation.threshold)
    result = evaluator.evaluate(
        loader, criterion=criterion, return_predictions=args.predictions_out is not None
    )

    summary = {k: v for k, v in result.items() if isinstance(v, (int, float))}
    logger.info("Metrics on %s: %s", args.split, json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    # Write metrics.json if output_dir is provided
    if args.output_dir:
        metrics_file = Path(args.output_dir) / "metrics.json"
        metrics_file.write_text(json.dumps(summary, indent=2, default=str))
        logger.info("Metrics written to %s", metrics_file)

    if args.predictions_out is not None and "y_true" in result and "y_prob" in result:
        y_true = np.asarray(result["y_true"])
        y_prob = np.asarray(result["y_prob"])
        y_pred = (y_prob >= cfg.evaluation.threshold).astype(int)
        # Predictions are produced with shuffle=False, so their order matches
        # the split's `samples` list one-to-one.
        samples = datasets[args.split].samples
        if len(samples) != len(y_true):
            logger.warning(
                "Sample/prediction count mismatch (%d vs %d); writing a bare "
                "predictions CSV without manifest columns.",
                len(samples),
                len(y_true),
            )
            df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_prob_malignant": y_prob})
        else:
            # Per-sample predictions keyed by the manifest's original image path
            # (stored in Sample.extra by the loader).
            key_col = "preprocessed_image_path"
            preds = pd.DataFrame(
                {
                    key_col: [s.extra.get(key_col, s.image_path) for s in samples],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_prob_malignant": y_prob,
                }
            )
            # Take the manifest and add the y_true / y_pred columns onto the
            # rows of the evaluated split (inner join → only evaluated images).
            try:
                manifest = pd.read_csv(cfg.data.manifest_path)
            except Exception as exc:  # noqa: BLE001 — fall back to bare predictions
                logger.warning(
                    "Could not read manifest %s (%s); writing predictions "
                    "without manifest columns.",
                    cfg.data.manifest_path,
                    exc,
                )
                df = preds
            else:
                if key_col in manifest.columns:
                    df = manifest.merge(preds, on=key_col, how="inner")
                else:
                    logger.warning(
                        "Manifest has no %r column; writing predictions keyed "
                        "by resolved image path instead.",
                        key_col,
                    )
                    df = preds
        out_path = Path(args.predictions_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(
            "Per-sample predictions (%d rows) written to %s", len(df), out_path
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
