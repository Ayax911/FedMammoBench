"""Flower client for fedmammobench.

The client owns its local train and val DataLoaders and its model. On each
``fit`` call it:

1. Loads server-pushed parameters into the local model.
2. Runs ``local_epochs`` of training via :class:`fedmammobench.training.Trainer`.
3. Returns the updated parameters, number of training samples, and a metrics
   dict (training loss; per-class counts).

On ``evaluate`` it runs the :class:`Evaluator` over the local validation
loader and returns ``(loss, num_examples, metrics)``.

The :func:`client_fn_factory` returns a function that Flower's simulation
engine calls per virtual client, given a numeric client id.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import flwr as fl
from flwr.common import Context, NDArrays, Scalar
from torch.utils.data import DataLoader

from fedmammobench.configs.schema import ExperimentConfig
from fedmammobench.datasets import MammographyDataset, build_dataloader, partition_indices
from fedmammobench.evaluation import Evaluator
from fedmammobench.federated.param_utils import (
    load_ndarrays_to_state_dict,
    state_dict_to_ndarrays,
)
from fedmammobench.models import build_model
from fedmammobench.models.weight_loaders import apply_freeze_policy
from fedmammobench.training import Trainer, build_loss, build_optimizer, build_scheduler
from fedmammobench.utils.csv_logger import CSVLogger
from fedmammobench.utils.device import resolve_device
from fedmammobench.utils.logging_utils import get_logger
from fedmammobench.utils.seeding import set_global_seed

_logger = get_logger(__name__)


class FedMammoBenchClient(fl.client.NumPyClient):
    """A single federated client (one virtual hospital).

    Args:
        client_id: Integer id (0..num_clients-1). Used as a log prefix and as
            a seed offset so client RNG state is deterministic but distinct.
        cfg: Full experiment config (the client consumes ``model``,
            ``training``, ``data``, ``evaluation``).
        train_dataset: This client's local training partition.
        val_dataset: This client's local validation partition.
        device: torch.device to use.
        out_root: Directory for client-side artifacts. When set, enables
            per-epoch CSV logging and enriched prediction CSV with manifest
            columns. Simulation mode passes this; gRPC mode uses run_client.py.
    """

    def __init__(
        self,
        client_id: int,
        cfg: ExperimentConfig,
        train_dataset: MammographyDataset,
        val_dataset: MammographyDataset | None,
        device: torch.device,
        out_root: Path | None = None,
    ) -> None:
        self.client_id = int(client_id)
        self.cfg = cfg
        self.device = device
        self.out_root = Path(out_root) if out_root is not None else None
        self.val_dataset = val_dataset

        train_labels = train_dataset.labels
        if (train_labels == 1).sum() == 0:
            _logger.error(
                "Client %d: training set has NO malignant samples. "
                "The model will not learn to detect malignancy. "
                "Review the manifest and split configuration.",
                self.client_id,
            )

        # Build a fresh model from the same config; weights will be set by
        # the server in ``fit``/``evaluate`` (strict=True), so loading
        # pretrained weights here is redundant and would require the checkpoint
        # to be present on every node.
        self.model = build_model(cfg.model, load_pretrained_weights=False).to(device)

        self.train_loader: DataLoader = build_dataloader(
            train_dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=True,
            balance_classes=cfg.data.balance_classes,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            seed=cfg.seed + 1000 * self.client_id,
        )
        self.val_loader: DataLoader | None = None
        if val_dataset is not None and len(val_dataset) > 0:
            self.val_loader = build_dataloader(
                val_dataset,
                batch_size=cfg.data.batch_size,
                num_workers=cfg.data.num_workers,
                shuffle=False,
                balance_classes=False,
                seed=cfg.seed,
            )

        self.criterion = build_loss(
            cfg.training.loss,
            train_labels=train_dataset.labels,
            num_classes=cfg.model.num_classes,
        ).to(device)

        self.evaluator = Evaluator(
            self.model, device=device, threshold=cfg.evaluation.threshold
        )

        # Per-epoch CSV (train + val metrics each local epoch, like centralized).
        # Created lazily on the first fit call so the directory exists by then.
        self._epoch_csv: CSVLogger | None = None
        if self.out_root is not None:
            self.out_root.mkdir(parents=True, exist_ok=True)
            self._epoch_csv = CSVLogger(self.out_root / "epoch_metrics.csv")

    # ------------------------------------------------------------------
    # NumPyClient API
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:  # noqa: ARG002
        return state_dict_to_ndarrays(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        load_ndarrays_to_state_dict(self.model, parameters, strict=True)
        current_round = int(config.get("current_round", 0))
        apply_freeze_policy(self.model, self.cfg.model, current_round=current_round)
        local_epochs = int(config.get("local_epochs", self.cfg.training.local_epochs))
        proximal_mu = float(config.get("proximal_mu", 0.0))

        # Capture global parameters as a single flat vector before any local update.
        # Using parameters_to_vector reduces memory fragmentation and GC overhead
        # versus a list of per-layer tensors (~100 separate allocations for ResNet50).
        global_params: torch.Tensor | None = None
        if proximal_mu > 0.0:
            with torch.no_grad():
                global_params = torch.nn.utils.parameters_to_vector(
                    self.model.parameters()
                ).detach()

        # Fresh optimizer each round — standard in Flower; cross-round state
        # would otherwise pollute aggregation.
        optimizer = build_optimizer(self.model, self.cfg.training.optimizer)
        scheduler = build_scheduler(optimizer, self.cfg.training.scheduler)
        trainer = Trainer(
            self.model,
            optimizer,
            self.criterion,
            self.device,
            grad_clip_norm=self.cfg.training.grad_clip_norm,
            mixed_precision=self.cfg.training.mixed_precision,
            log_tag=f"client_{self.client_id}",
            scheduler=scheduler,
        )
        last: dict[str, Any] = {}
        local_unfreeze_ep = self.cfg.model.local_unfreeze_at_epoch
        fit_start = time.perf_counter()
        for ep in range(local_epochs):
            # Cyclic within-round unfreeze: at the configured local epoch,
            # partially unfreeze the backbone for the remaining epochs of
            # this round. The backbone is re-frozen at the start of the next
            # round by apply_freeze_policy (called above, before this loop).
            if local_unfreeze_ep is not None and ep == local_unfreeze_ep:
                backbone = getattr(self.model, "backbone", self.model)
                layers_to_unfreeze = self.cfg.model.unfreeze_layers or []
                if layers_to_unfreeze:
                    for ln in layers_to_unfreeze:
                        layer = getattr(backbone, ln, None)
                        if layer is not None:
                            for p in layer.parameters():
                                p.requires_grad = True
                else:
                    for p in self.model.parameters():
                        p.requires_grad = True
                _logger.info(
                    "client %d round %d: cyclic unfreeze at local epoch %d/%d — layers=%s",
                    self.client_id, current_round, ep, local_epochs,
                    layers_to_unfreeze or "all",
                )
            t_train = time.perf_counter()
            last = trainer.train_one_epoch(
                self.train_loader,
                epoch=ep,
                proximal_mu=proximal_mu,
                global_params=global_params,
            )
            train_seconds = time.perf_counter() - t_train

            if self._epoch_csv is not None:
                epoch_row: dict[str, Any] = {
                    "round": current_round,
                    "local_epoch": ep,
                    "train_loss": float(last.get("loss", float("nan"))),
                    "task_loss": float(last.get("task_loss", float("nan"))),
                    "train_seconds": round(train_seconds, 3),
                }
                if self.val_loader is not None:
                    t_val = time.perf_counter()
                    val_result = self.evaluator.evaluate(
                        self.val_loader, criterion=self.criterion
                    )
                    val_seconds = time.perf_counter() - t_val
                    epoch_row.update({
                        "val_loss": float(val_result.get("loss", float("nan"))),
                        "val_accuracy": float(val_result.get("accuracy", float("nan"))),
                        "val_f1": float(val_result.get("f1", float("nan"))),
                        "val_roc_auc": float(val_result.get("roc_auc", float("nan"))),
                        "val_sensitivity": float(val_result.get("sensitivity", float("nan"))),
                        "val_specificity": float(val_result.get("specificity", float("nan"))),
                        "val_seconds": round(val_seconds, 3),
                    })
                self._epoch_csv.append(epoch_row)

        fit_seconds = time.perf_counter() - fit_start
        n_samples = int(last.get("samples", len(self.train_loader.dataset)))  # type: ignore[arg-type]
        # train_loss includes the proximal penalty for FedProx; task_loss is
        # cross-entropy only and is the correct metric for strategy comparisons.
        train_loss = float(last.get("loss", 0.0))
        task_loss = float(last.get("task_loss", train_loss))
        metrics: dict[str, Scalar] = {
            "train_loss": train_loss,
            "task_loss": task_loss,
            "client_id": self.client_id,
            "num_samples": n_samples,
            # Wall-clock seconds this node spent in local training this round.
            "fit_seconds": float(fit_seconds),
            "local_epochs": local_epochs,
        }
        _logger.info(
            "client %d: fit done in %.2fs (%d epochs, %d samples)",
            self.client_id,
            fit_seconds,
            local_epochs,
            n_samples,
        )
        return state_dict_to_ndarrays(self.model), n_samples, metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        load_ndarrays_to_state_dict(self.model, parameters, strict=True)
        if self.val_loader is None:
            # Reporting num_examples=0 ensures Flower's weighted aggregator
            # ignores this client's row instead of polluting the global mean.
            return 0.0, 0, {"warning": "no_local_val"}
        server_round = int(config.get("current_round", 0))
        save_preds = bool(getattr(self.cfg.evaluation, "save_predictions", False))
        eval_start = time.perf_counter()
        result = self.evaluator.evaluate(
            self.val_loader,
            criterion=self.criterion,
            return_predictions=save_preds,
        )
        eval_seconds = time.perf_counter() - eval_start

        if save_preds and self.out_root is not None:
            self._save_predictions(server_round, result)

        loss = float(result.get("loss", 0.0))
        n_samples = int(len(self.val_loader.dataset))  # type: ignore[arg-type]
        numeric_keys = (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "sensitivity",
            "specificity",
        )
        metrics: dict[str, Scalar] = {}
        for k in numeric_keys:
            if k not in result:
                continue
            v = float(result[k])
            if np.isnan(v):
                _logger.debug(
                    "client %d: metric %r is NaN (likely single-class batch); omitting from aggregation",
                    self.client_id,
                    k,
                )
                continue
            metrics[k] = v
        metrics["client_id"] = self.client_id
        metrics["eval_seconds"] = float(eval_seconds)
        return loss, n_samples, metrics

    def _save_predictions(self, server_round: int, result: dict[str, Any]) -> None:
        y_true: np.ndarray | None = result.get("y_true")
        y_prob: np.ndarray | None = result.get("y_prob")
        if y_true is None or y_prob is None:
            return
        assert self.out_root is not None
        self.out_root.mkdir(parents=True, exist_ok=True)

        # val_loader uses shuffle=False, so samples are in the same order as
        # predictions. Zip with val_dataset.samples to include manifest metadata.
        samples = self.val_dataset.samples if self.val_dataset is not None else []
        extra_keys: list[str] = list(samples[0].extra.keys()) if samples and samples[0].extra else []
        fieldnames = ["round", "image_path", "patient_id"] + extra_keys + ["y_true", "y_prob", "y_pred"]

        pred_path = self.out_root / "predictions.csv"
        write_header = not pred_path.exists()
        threshold = float(getattr(self.cfg.evaluation, "threshold", 0.5))
        with pred_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for i, (yt, yp) in enumerate(zip(y_true.tolist(), y_prob.tolist())):
                row: dict[str, Any] = {
                    "round": server_round,
                    "y_true": int(yt),
                    "y_prob": round(float(yp), 6),
                    "y_pred": int(float(yp) >= threshold),
                }
                if i < len(samples):
                    s = samples[i]
                    row["image_path"] = s.image_path
                    row["patient_id"] = s.patient_id or ""
                    for k in extra_keys:
                        row[k] = s.extra.get(k, "")
                else:
                    row["image_path"] = ""
                    row["patient_id"] = ""
                    for k in extra_keys:
                        row[k] = ""
                writer.writerow(row)


# ----------------------------------------------------------------------
# Client factory
# ----------------------------------------------------------------------

def _materialize_client_partitions(
    cfg: ExperimentConfig,
    datasets: dict[str, MammographyDataset],
) -> list[tuple[MammographyDataset, MammographyDataset | None]]:
    """Return a list of (train_subset, val_subset) for each client.

    Training data is partitioned according to ``cfg.partitioning``. The
    validation set is also partitioned into per-client subsets using the same
    scheme so that each client evaluates on a locally-distinct validation split.
    This ensures federated validation metrics reflect true local distributions
    rather than a shared held-out set evaluated identically by all clients.
    """
    train_ds = datasets["train"]
    val_ds = datasets.get("val")
    labels = train_ds.labels

    # Build patient-id array; fall back to sample-level partitioning when any
    # patient_id is None or NaN (pandas reads missing values as float NaN).
    from fedmammobench.configs.data_config import check_patient_ids_for_nan

    raw_pids = train_ds.patient_ids
    if check_patient_ids_for_nan(raw_pids):
        _logger.warning(
            "Some training samples have no patient_id; falling back to "
            "sample-level partitioning (patient leakage not prevented)."
        )
        patient_ids_arr = None
    else:
        patient_ids_arr = np.asarray(raw_pids, dtype=object)

    client_indices = partition_indices(
        labels,
        num_clients=cfg.federated.num_clients,
        patient_ids=patient_ids_arr,
        scheme=cfg.partitioning.scheme,
        alpha=cfg.partitioning.alpha,
        min_per_client=cfg.partitioning.min_per_client,
        max_retries=cfg.partitioning.max_retries,
        quantity_skew_sigma=cfg.partitioning.quantity_skew_sigma,
        seed=cfg.seed,
    )

    # Partition val_ds per client (IID) so each client validates on a distinct
    # local subset, making federated evaluation metrics methodologically sound.
    val_client_indices: list[list[int]] | None = None
    if val_ds is not None and len(val_ds) >= cfg.federated.num_clients:
        val_labels = val_ds.labels
        val_client_indices = partition_indices(
            val_labels,
            num_clients=cfg.federated.num_clients,
            patient_ids=None,
            scheme="iid",
            seed=cfg.seed + 1000,  # distinct seed from train partition
        )

    pairs: list[tuple[MammographyDataset, MammographyDataset | None]] = []
    for ci, idxs in enumerate(client_indices):
        sub_train = train_ds.subset(idxs)
        if val_client_indices is not None:
            sub_val: MammographyDataset | None = val_ds.subset(val_client_indices[ci])  # type: ignore[union-attr]
        else:
            sub_val = val_ds  # fallback: shared val when too small to partition
        pairs.append((sub_train, sub_val))
        _logger.info(
            "Client %d: %d train samples (class counts=%s)",
            ci,
            len(sub_train),
            sub_train.class_counts(),
        )
    return pairs


def client_fn_factory(
    cfg: ExperimentConfig,
    datasets: dict[str, MammographyDataset],
    *,
    out_root: Path | None = None,
) -> Callable[[Context], fl.client.Client]:
    """Return a Flower ``client_fn`` consuming a Context and yielding Clients.

    The function captures pre-built dataset partitions and rebuilds a fresh
    model + client object for each invocation. Flower's simulation engine
    may call this many times across rounds.

    Args:
        out_root: Base directory for client artifacts (e.g.
            ``runs/<name>/clients``). Each client writes to
            ``out_root/client_<cid>/``. When None, per-epoch CSV and enriched
            predictions are disabled (simulation without artifact output).
    """
    pairs = _materialize_client_partitions(cfg, datasets)
    device = resolve_device(cfg.device)

    def client_fn(context: Context) -> fl.client.Client:
        # The new Flower API passes a Context; older code passed cid: str.
        # Resolve the integer id from whichever interface we are given.
        cid_raw = (
            context.node_config.get("partition-id")
            if hasattr(context, "node_config")
            else None
        )
        if cid_raw is None:
            # Fall back to legacy string-cid argument: ``client_fn(cid)``.
            cid_raw = getattr(context, "cid", 0)
        cid = int(cid_raw)
        if cid >= len(pairs):
            raise IndexError(
                f"client_fn received cid={cid} but only {len(pairs)} partitions exist."
            )
        train_ds, val_ds = pairs[cid]
        client_out = (Path(out_root) / f"client_{cid}") if out_root is not None else None
        # Seed each client distinctly so dropout/data augmentation order
        # diverges across clients but reruns are reproducible.
        set_global_seed(cfg.seed + cid + 1, deterministic=True)
        np_client = FedMammoBenchClient(
            client_id=cid,
            cfg=cfg,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device=device,
            out_root=client_out,
        )
        return np_client.to_client()

    return client_fn


__all__ = ["FedMammoBenchClient", "client_fn_factory"]
