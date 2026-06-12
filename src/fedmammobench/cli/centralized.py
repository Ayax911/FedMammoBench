"""Console entrypoint: ``fedmammobench-centralized``.

Delegates to ``scripts/run_centralized.py``'s :func:`main`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_main(filename: str):  # noqa: ANN201
    """Load and return the ``main`` callable from ``scripts/<filename>``."""
    here = Path(__file__).resolve()
    # Walk up to repo root: src/fedmammobench/cli/x.py -> repo
    repo_root = here.parent.parent.parent.parent
    script = repo_root / "scripts" / filename
    if not script.is_file():
        raise FileNotFoundError(f"Could not locate {script}; install layout unexpected.")
    spec = importlib.util.spec_from_file_location(f"_fedmammobench_script_{filename}", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


def main() -> int:
    return _load_script_main("run_centralized.py")()


if __name__ == "__main__":
    raise SystemExit(main())
