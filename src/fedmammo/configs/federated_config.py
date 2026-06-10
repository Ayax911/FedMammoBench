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

    def model_config_hash(self, model_config_fields: dict[str, Any]) -> str:
        """Return a short SHA-256 hex digest of the model config fields.

        Used for server↔client handshake: both sides compute this hash and
        compare before training starts to catch mismatched architectures or
        class counts early rather than at state_dict load time.

        Args:
            model_config_fields: Dict returned by
                :meth:`~fedmammo.configs.model_config.ModelConfig.config_hash_fields`.
        """
        canonical = json.dumps(model_config_fields, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


__all__ = [
    "FederatedConfig",
    "StrategyConfig",
]
