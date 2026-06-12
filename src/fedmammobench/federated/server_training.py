"""Server-side training — hybrid federated learning.

When ``federated.server_training.enabled`` is set, the central node is no longer
a pure aggregator: it owns a local dataset and, after each round's client
aggregation, continues training from the aggregated weights for a few epochs on
that data. The server-updated weights (optionally interpolated with the
aggregated ones) become the new global model distributed to clients next round.

Two pieces live here:

- :class:`ServerTrainer` — builds the server's model/dataset/optimizer once and
  exposes :meth:`train`, which takes the aggregated parameters (as Flower
  ``NDArrays``) and returns the server-trained parameters.
- :func:`attach_server_training` — wraps any strategy's ``aggregate_fit`` so the
  post-aggregation server step happens transparently, mirroring how
  ``_attach_federated_logging`` wraps ``aggregate_evaluate`` in
  :mod:`fedmammobench.federated.server`.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from fedmammobench.configs.schema import ExperimentConfig
from fedmammobench.datasets import build_dataloader, build_dataset
from fedmammobench.federated.param_utils import (
    load_ndarrays_to_state_dict,
    state_dict_to_ndarrays,
)
from fedmammobench.models import build_model
from fedmammobench.training import Trainer, build_loss, build_optimizer
from fedmammobench.utils.device import resolve_device
from fedmammobench.utils.logging_utils import get_logger
from fedmammobench.utils.tensorboard_utils import TensorBoardWriter

_logger = get_logger(__name__)


class ServerTrainer:
    """Owns the server's local dataset and runs training from given weights."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        st = cfg.federated.server_training
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.local_epochs = int(st.local_epochs)

        # Build the server's own dataset with the same loader as the clients,
        # but from the server's manifest and using the *entire* manifest for
        # training (no val/test carved on the server).
        server_cfg = copy.deepcopy(cfg)
        server_cfg.data.name = st.dataset_name or cfg.data.name
        server_cfg.data.manifest_path = st.manifest_path
        server_cfg.data.image_root = st.image_root
        server_cfg.data.val_fraction = 0.0
        server_cfg.data.test_fraction = 0.0
        datasets = build_dataset(server_cfg)
        if "train" not in datasets or len(datasets["train"]) == 0:
            raise ValueError(
                "server_training: the server manifest produced no training samples. "
                f"Check manifest_path={st.manifest_path!r} and image_root={st.image_root!r}."
            )
        self.train_ds = datasets["train"]

        self.model = build_model(cfg.model).to(self.device)
        self.criterion = build_loss(
            cfg.training.loss,
            train_labels=self.train_ds.labels,
            num_classes=cfg.model.num_classes,
        ).to(self.device)
        self.loader = build_dataloader(
            self.train_ds,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            balance_classes=cfg.data.balance_classes,
            seed=cfg.seed,
        )
        self.last_loss: float = float("nan")
        _logger.info(
            "Server-side training enabled: %d local samples, %d epochs/round, weight=%.2f",
            len(self.train_ds),
            self.local_epochs,
            cfg.federated.server_training.server_weight,
        )

    def train(self, parameters: NDArrays, server_round: int) -> NDArrays:
        """Load ``parameters``, train ``local_epochs`` epochs, return new params."""
        load_ndarrays_to_state_dict(self.model, parameters, strict=True)
        # Fresh optimizer each round: the server fine-tunes from the current
        # global weights, so optimizer state should not persist across rounds.
        optimizer = build_optimizer(self.model, self.cfg.training.optimizer)
        trainer = Trainer(
            self.model,
            optimizer,
            self.criterion,
            self.device,
            grad_clip_norm=self.cfg.training.grad_clip_norm,
            mixed_precision=self.cfg.training.mixed_precision,
            log_tag="server",
        )
        last = float("nan")
        for epoch in range(self.local_epochs):
            stats = trainer.train_one_epoch(self.loader, epoch=epoch)
            last = float(stats.get("task_loss", stats.get("loss", float("nan"))))
        self.last_loss = last
        _logger.info(
            "[server] round %d server-train: %d epochs, final loss=%.4f",
            server_round,
            self.local_epochs,
            last,
        )
        return state_dict_to_ndarrays(self.model)


def attach_server_training(
    strategy: Any,
    trainer: ServerTrainer,
    *,
    server_weight: float,
    tb_writer: TensorBoardWriter | None = None,
) -> None:
    """Wrap ``strategy.aggregate_fit`` to run a server training step each round.

    The aggregated client weights are passed through :meth:`ServerTrainer.train`;
    the result (optionally interpolated with the aggregated weights via
    ``server_weight``) replaces the parameters the strategy hands back. The
    strategy's own metrics are preserved and a ``server_train_loss`` entry is
    added.
    """
    original = strategy.aggregate_fit

    def wrapped(
        server_round: int,
        results: Sequence[tuple[ClientProxy, FitRes]],
        failures: Sequence[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        aggregated, metrics = original(server_round, results, failures)
        if aggregated is None:
            return aggregated, metrics

        agg_nd = parameters_to_ndarrays(aggregated)
        server_nd = trainer.train(agg_nd, server_round)
        if server_weight < 1.0:
            w = float(server_weight)
            blended = [(1.0 - w) * a + w * s for a, s in zip(agg_nd, server_nd, strict=True)]
        else:
            blended = server_nd

        metrics = {**metrics, "server_train_loss": trainer.last_loss}
        if tb_writer is not None and trainer.last_loss == trainer.last_loss:  # not NaN
            tb_writer.log_scalars(
                "server/server_training", {"train_loss": trainer.last_loss}, server_round
            )
        return ndarrays_to_parameters(blended), metrics

    strategy.aggregate_fit = wrapped  # type: ignore[method-assign]


__all__ = ["ServerTrainer", "attach_server_training"]
