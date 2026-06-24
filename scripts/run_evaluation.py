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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging()
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

    if args.predictions_out is not None and "y_true" in result and "y_prob" in result:
        y_true = np.asarray(result["y_true"])
        y_prob = np.asarray(result["y_prob"])
        # Predictions are produced with shuffle=False, so their order matches
        # the split's `samples` list one-to-one. Recover the image identity
        # (and patient id) from there to make the CSV self-describing.
        samples = datasets[args.split].samples
        if len(samples) != len(y_true):
            logger.warning(
                "Sample/prediction count mismatch (%d vs %d); image paths "
                "may be misaligned. Writing predictions without them.",
                len(samples),
                len(y_true),
            )
            data = {"y_true": y_true, "y_prob_malignant": y_prob}
        else:
            data = {
                "image_path": [s.image_path for s in samples],
                "patient_id": [s.patient_id for s in samples],
                "y_true": y_true,
                "y_pred": (y_prob >= cfg.evaluation.threshold).astype(int),
                "y_prob_malignant": y_prob,
            }
        df = pd.DataFrame(data)
        out_path = Path(args.predictions_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Per-sample predictions written to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
