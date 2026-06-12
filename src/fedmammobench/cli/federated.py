"""Console entrypoint: ``fedmammobench-federated``."""

from __future__ import annotations

from fedmammobench.cli.centralized import _load_script_main


def main() -> int:
    return _load_script_main("run_federated.py")()


if __name__ == "__main__":
    raise SystemExit(main())
