"""Federated learning runtime configuration with built-in validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StrategyConfig:
    """Federated strategy selection.

    ``params`` is a free-form dict consumed by the strategy implementation
    (e.g. ``{"mu": 0.01}`` for FedProx).
    """

    name: Literal["fedavg", "fedprox", "scaffold", "fedbn"] = "fedavg"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerTrainingConfig:
    """Optional server-side training (hybrid federated learning).

    When ``enabled``, the central node owns a local dataset and, after each
    round's client aggregation, runs ``local_epochs`` of training on that data
    starting from the aggregated global weights. The result becomes the new
    global model (optionally interpolated via ``server_weight``). This turns the
    server from a pure aggregator into an additional training participant.

    The server dataset is built with the same loader as the clients
    (``dataset_name`` defaults to ``data.name``) but from its own
    ``manifest_path`` / ``image_root``; the entire manifest is used for training
    (no val/test split is carved on the server).

    Attributes:
        enabled: Master switch. When False (default) the server only aggregates.
        dataset_name: Registered dataset to use; defaults to ``data.name``.
        manifest_path: CSV manifest for the server's own data (required).
        image_root: Image root for the server's own data (required).
        local_epochs: Training epochs the server runs per round on its data.
        server_weight: Interpolation in ``(0, 1]`` between the aggregated
            weights and the server-trained weights:
            ``new = (1 - w) * aggregated + w * server_trained``. ``1.0`` (default)
            takes the server-trained weights outright (the server still starts
            from the aggregated weights, so it is a continuation, not a reset).
    """

    enabled: bool = False
    dataset_name: str | None = None
    manifest_path: str | None = None
    image_root: str | None = None
    local_epochs: int = 1
    server_weight: float = 1.0

    def validate(self) -> None:
        """Raise ValueError for invalid server-training settings."""
        if not self.enabled:
            return
        if self.local_epochs < 1:
            raise ValueError(
                f"server_training.local_epochs must be >= 1, got {self.local_epochs}"
            )
        if not (0.0 < self.server_weight <= 1.0):
            raise ValueError(
                f"server_training.server_weight must be in (0, 1], got {self.server_weight}"
            )
        if not self.manifest_path or not self.image_root:
            raise ValueError(
                "server_training.enabled=true requires both `manifest_path` and "
                "`image_root` pointing to the server's local dataset."
            )


@dataclass
class FederatedConfig:
    """Flower simulation / gRPC parameters.

    Attributes:
        num_clients: Number of simulated clients (i.e. virtual hospitals).
            In real gRPC mode this is informational; the actual count is
            determined by ``min_*_clients`` and which devices connect.
        rounds: Number of FL rounds.
        fraction_fit: Fraction of clients sampled for training per round.
        fraction_evaluate: Fraction sampled for federated evaluation per round.
        min_fit_clients: Hard minimum for training selection.
        min_evaluate_clients: Hard minimum for evaluation selection.
        min_available_clients: Minimum clients before a round starts.
        accept_failures: Whether to tolerate individual client failures.
        client_resources: Per-client Ray quotas (simulation only).
        ray_init_args: Forwarded to Ray (simulation only).
        server_address: ``HOST:PORT`` the gRPC server binds to / clients
            connect to. Used only by ``run_grpc_server`` and the standalone
            client script; ignored by ``run_simulation``.
        grpc_max_message_length: Maximum gRPC message size in bytes. Default
            512 MB. Increase for large models (e.g. InceptionV3) or many
            simultaneous clients. Set in both server and client configs.
        round_timeout_seconds: Seconds before a round is considered failed
            if not all min_fit_clients have responded. 0 disables the timeout
            (blocks indefinitely — default Flower behavior).
        strategy: Strategy selection (see :class:`StrategyConfig`).
        server_training: Optional hybrid server-side training
            (see :class:`ServerTrainingConfig`). Disabled by default.
    """

    num_clients: int = 4
    rounds: int = 10
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    accept_failures: bool = True
    client_resources: dict[str, float] = field(
        default_factory=lambda: {"num_cpus": 1.0, "num_gpus": 0.0}
    )
    ray_init_args: dict[str, Any] = field(default_factory=dict)
    server_address: str = "0.0.0.0:8080"
    grpc_max_message_length: int = 512 * 1024 * 1024  # 512 MB
    round_timeout_seconds: int = 0
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    server_training: ServerTrainingConfig = field(default_factory=ServerTrainingConfig)

    def validate(self) -> None:
        """Raise ValueError for invalid federated settings."""
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if not (0.0 < self.fraction_fit <= 1.0):
            raise ValueError(f"fraction_fit must be in (0, 1], got {self.fraction_fit}")
        if not (0.0 <= self.fraction_evaluate <= 1.0):
            raise ValueError(
                f"fraction_evaluate must be in [0, 1], got {self.fraction_evaluate}"
            )
        if self.min_fit_clients > self.num_clients:
            raise ValueError(
                f"min_fit_clients ({self.min_fit_clients}) must be <= "
                f"num_clients ({self.num_clients})"
            )
        if self.min_available_clients > self.num_clients:
            raise ValueError(
                f"min_available_clients ({self.min_available_clients}) must be <= "
                f"num_clients ({self.num_clients})"
            )
        if self.grpc_max_message_length < 1:
            raise ValueError(
                f"grpc_max_message_length must be > 0, got {self.grpc_max_message_length}"
            )
        if self.round_timeout_seconds < 0:
            raise ValueError(
                f"round_timeout_seconds must be >= 0, got {self.round_timeout_seconds}"
            )
        self.server_training.validate()

    def model_config_hash(self, model_config_fields: dict[str, Any]) -> str:
        """Return a short SHA-256 hex digest of the model config fields.

        Used for server↔client handshake: both sides compute this hash and
        compare before training starts to catch mismatched architectures or
        class counts early rather than at state_dict load time.

        Args:
            model_config_fields: Dict returned by
                :meth:`~fedmammobench.configs.model_config.ModelConfig.config_hash_fields`.
        """
        canonical = json.dumps(model_config_fields, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


__all__ = [
    "FederatedConfig",
    "ServerTrainingConfig",
    "StrategyConfig",
]
