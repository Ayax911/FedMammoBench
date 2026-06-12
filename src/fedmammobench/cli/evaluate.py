"""Console entrypoint: ``fedmammo-evaluate``."""

from __future__ import annotations

from fedmammo.cli.centralized import _load_script_main


def main() -> int:
    return _load_script_main("run_evaluation.py")()


if __name__ == "__main__":
    raise SystemExit(main())
