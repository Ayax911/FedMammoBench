"""Run a federated simulation.

Usage::

    python scripts/run_federated.py --config configs/radimagenet_resnet50_fedavg.yaml
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

from fedmammobench.configs import load_config, save_config  # noqa: E402
from fedmammobench.federated.server import run_simulation  # noqa: E402
from fedmammobench.utils import get_logger, set_global_seed, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Federated simulation entrypoint.")
    p.add_argument("--config", "-c", required=True, type=str, help="Path to a YAML config.")
    p.add_argument("--output-dir", type=str, default=None, help="Override cfg.output_dir.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.mode != "federated":
        print(
            f"Warning: cfg.mode={cfg.mode!r} but you launched the federated script. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

    out_root = Path(args.output_dir or Path(cfg.output_dir) / cfg.name).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out_root / "run.log")
    logger = get_logger("federated")

    set_global_seed(cfg.seed, deterministic=True)
    save_config(cfg, out_root / "config.snapshot.yaml")

    history = run_simulation(cfg, output_dir=out_root)
    logger.info("History summary: %s", history)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
