"""CLI wrapper for ``fedmammobench.plotting`` — generate plots for a completed run.

The actual plotting logic lives in ``src/fedmammobench/plotting.py`` (so it's
importable/testable and reusable as the ``autoplot()`` hook called
automatically at the end of training). This script is just the
``--run-dir``/``--out-dir``/``--dpi`` CLI documented in CLAUDE.md and the
``/plot`` skill.

Usage::

    python scripts/plot_experiment.py --run-dir runs/exp08_centralized_resnet50
    python scripts/plot_experiment.py --run-dir runs/exp07_fedavg_resnet50 --dpi 150

See ``fedmammobench.plotting`` module docstring for the full list of PNGs
produced in centralized vs. federated mode.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    """Make ``src/fedmammobench`` importable when running without ``pip install``."""
    here = Path(__file__).resolve()
    src = here.parent.parent / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot experiment training curves.")
    p.add_argument(
        "--run-dir", "-r", required=True, type=str,
        help="Path to the experiment run directory (e.g. runs/exp08_centralized_resnet50).",
    )
    p.add_argument(
        "--out-dir", "-o", default=None, type=str,
        help="Output directory for plots. Defaults to <run-dir>/plots/.",
    )
    p.add_argument("--dpi", default=120, type=int, help="Plot resolution (default 120).")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    from fedmammobench.plotting import plot_run

    try:
        plot_run(args.run_dir, args.out_dir, args.dpi)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
