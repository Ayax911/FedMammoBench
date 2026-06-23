"""gRPC server entrypoint for real multi-device federated deployment.

Starts a Flower gRPC server that waits for physical client devices to connect.
Each client device must run ``scripts/run_client.py`` pointing to this server's
IP address.

Usage::

    python scripts/run_server.py --config configs/fedavg_mammobench_server.yaml
    python scripts/run_server.py --config configs/fedavg_mammobench_server.yaml --address 0.0.0.0:8080
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
from fedmammobench.federated.server import run_grpc_server  # noqa: E402
from fedmammobench.utils import get_logger, set_global_seed, setup_logging  # noqa: E402
from fedmammobench.utils.device import log_device_info, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Flower gRPC server for real federated deployment. "
            "Waits for client devices to connect on the specified address."
        )
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        type=str,
        help="Path to a YAML server config (e.g. configs/fedavg_mammobench_server.yaml).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory for logs, metrics, and checkpoints.",
    )
    p.add_argument(
        "--address",
        type=str,
        default=None,
        help=(
            "Override cfg.federated.server_address. "
            "Format: 192.168.15.59:8080"
            "Clients must use the server's LAN IP (e.g. 192.168.1.10:8080)."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if cfg.mode != "federated":
        print(
            f"Warning: cfg.mode={cfg.mode!r} but you launched the gRPC server script. "
            "Proceeding anyway.",
            file=sys.stderr,
        )

    # CLI address override takes precedence over YAML.
    if args.address:
        cfg.federated.server_address = args.address

    out_root = Path(args.output_dir or Path(cfg.output_dir) / cfg.name).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out_root / "server.log")
    logger = get_logger("run_server")

    set_global_seed(cfg.seed, deterministic=True)
    save_config(cfg, out_root / "config.snapshot.yaml")

    log_device_info(resolve_device(cfg.device), logger)
    logger.info(
        "Server config loaded. address=%s  rounds=%d  strategy=%s  "
        "min_available_clients=%d",
        cfg.federated.server_address,
        cfg.federated.rounds,
        cfg.federated.strategy.name,
        cfg.federated.min_available_clients,
    )
    logger.info(
        "Artifacts will be written to: %s",
        out_root,
    )

    run_grpc_server(cfg, output_dir=out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
