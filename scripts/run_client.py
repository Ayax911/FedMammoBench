"""gRPC client entrypoint for real multi-device federated deployment.

Each physical node (hospital device) runs this script. The node loads its
local pre-partitioned dataset, connects to the central server, and participates
in federated training rounds.

The data on each node must be pre-partitioned before running the experiment:
  - A CSV manifest with the rows belonging to this node.
  - The corresponding image files under a local image root directory.

Usage::

    python scripts/run_client.py \\
        --config configs/fedavg_mammobench_client.yaml \\
        --server 192.168.1.10:8080 \\
        --client-id 0 \\
        --manifest data/mammobench/node0_manifest.csv \\
        --data-dir data/mammobench/images
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

import flwr as fl  # noqa: E402

from fedmammobench.configs import load_config, save_config  # noqa: E402
from fedmammobench.datasets import build_dataset  # noqa: E402
from fedmammobench.federated.client import FedMammoBenchClient  # noqa: E402
from fedmammobench.utils import get_logger, set_global_seed, setup_logging  # noqa: E402
from fedmammobench.utils.device import resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Flower gRPC client for real federated deployment. "
            "Connects to the central server and participates in FL rounds."
        )
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        type=str,
        help="Path to a YAML client config (e.g. configs/fedavg_mammobench_client.yaml).",
    )
    p.add_argument(
        "--server",
        required=True,
        type=str,
        help=(
            "Server address in HOST:PORT format. "
            "Must match the address the server is listening on "
            "(e.g. 192.168.1.10:8080)."
        ),
    )
    p.add_argument(
        "--client-id",
        required=True,
        type=int,
        help="Integer ID for this node (0-indexed). Must be unique across all nodes.",
    )
    p.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Override cfg.data.manifest_path with the path to this node's "
            "local CSV manifest. Required unless the YAML already points to "
            "the correct file."
        ),
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override cfg.data.image_root with this node's local image directory.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory for client-side logs.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if cfg.mode != "federated":
        print(
            f"Warning: cfg.mode={cfg.mode!r} but you launched the gRPC client script. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

    # Apply CLI overrides for per-node paths.
    if args.manifest:
        cfg.data.manifest_path = args.manifest
    if args.data_dir:
        cfg.data.image_root = args.data_dir

    out_root = (
        Path(args.output_dir or Path(cfg.output_dir) / cfg.name / f"client_{args.client_id}")
        .expanduser()
        .resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out_root / "client.log")
    logger = get_logger(f"run_client_{args.client_id}")

    # Each client gets a distinct but reproducible seed.
    set_global_seed(cfg.seed + args.client_id + 1, deterministic=True)
    save_config(cfg, out_root / "config.snapshot.yaml")

    logger.info(
        "Client %d starting. server=%s  dataset=%s  manifest=%s  image_root=%s",
        args.client_id,
        args.server,
        cfg.data.name,
        cfg.data.manifest_path,
        cfg.data.image_root,
    )

    # Build local datasets. Partitioning is NOT applied here — the data on
    # disk is already this node's pre-partitioned share. build_dataset()
    # simply loads the local manifest and produces train/val splits.
    datasets = build_dataset(cfg)
    train_dataset = datasets["train"]
    val_dataset = datasets.get("val")

    logger.info(
        "Local data loaded: train=%d samples  val=%s samples  class_counts=%s",
        len(train_dataset),
        len(val_dataset) if val_dataset is not None else 0,
        train_dataset.class_counts(),
    )

    device = resolve_device(cfg.device)

    # Instantiate the Flower client directly — no simulation, no Ray.
    np_client = FedMammoBenchClient(
        client_id=args.client_id,
        cfg=cfg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        out_root=out_root,
    )

    logger.info("Connecting to server at %s ...", args.server)
    fl.client.start_numpy_client(
        server_address=args.server,
        client=np_client,
    )

    logger.info("Client %d finished.", args.client_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
