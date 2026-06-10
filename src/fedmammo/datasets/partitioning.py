"""Federated data partitioning.

Given an array of training labels and a number of clients, produce a list of
index lists (one per client). Three schemes are supported:

- ``iid``           : shuffle-and-chunk; clients are statistically identical.
- ``dirichlet``     : draw class proportions per client from Dir(alpha).
                      Lower alpha => stronger label skew.
- ``quantity_skew`` : IID label distribution but unequal sample counts per
                      client (log-normal scaling with sigma).

All three schemes accept an optional ``patient_ids`` array. When provided,
every image belonging to the same patient is guaranteed to land on the same
client, eliminating cross-client patient leakage. Without ``patient_ids`` the
schemes operate on individual samples (original behavior, preserved for
backward compatibility).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from fedmammo.utils.logging_utils import get_logger

_logger = get_logger(__name__)

Scheme = Literal["iid", "dirichlet", "quantity_skew"]


def partition_indices(
    labels: np.ndarray,
    num_clients: int,
    *,
    patient_ids: np.ndarray | None = None,
    scheme: Scheme = "iid",
    alpha: float = 0.5,
    min_per_client: int = 16,
    max_retries: int = 20,
    quantity_skew_sigma: float = 0.5,
    seed: int = 42,
) -> list[list[int]]:
    """Partition ``len(labels)`` indices among ``num_clients`` clients.

    Args:
        labels: 1-D integer array of per-sample class labels.
        num_clients: Number of clients to partition into.
        patient_ids: Optional 1-D array of patient identifiers (same length as
            ``labels``). When provided, all samples belonging to the same
            patient are placed on the same client, preventing cross-client
            patient leakage. If ``None``, partitioning operates at the
            individual sample level.
        scheme: One of ``"iid"``, ``"dirichlet"``, or ``"quantity_skew"``.
        alpha: Dirichlet concentration parameter (only used by
            ``"dirichlet"``). Lower => more heterogeneous.
        min_per_client: Minimum number of *samples* (not patients) each client
            must receive. Dirichlet retries until this constraint is met.
        max_retries: Maximum Dirichlet retry attempts before raising.
        quantity_skew_sigma: Log-normal std-dev for ``"quantity_skew"``.
        seed: RNG seed for reproducibility.

    Returns:
        A list of length ``num_clients``; each entry is a list of integer
        indices into ``labels``.

    Raises:
        ValueError: on invalid arguments or if ``patient_ids`` contains None.
        RuntimeError: if Dirichlet partitioning cannot satisfy
            ``min_per_client`` within ``max_retries`` attempts.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    n = labels.shape[0]
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if num_clients > n:
        raise ValueError(f"num_clients ({num_clients}) exceeds dataset size ({n})")

    rng = np.random.default_rng(seed)

    if patient_ids is not None:
        patient_ids = np.asarray(patient_ids)
        if patient_ids.shape != labels.shape:
            raise ValueError(
                f"patient_ids shape {patient_ids.shape} must match labels shape {labels.shape}"
            )
        _, patient_labels, index_groups = _patient_groups(labels, patient_ids)
        n_patients = len(index_groups)
        if num_clients > n_patients:
            raise ValueError(
                f"num_clients ({num_clients}) exceeds number of unique patients ({n_patients})"
            )
        _logger.info(
            "Patient-aware partitioning: %d patients, %d samples, scheme=%s",
            n_patients,
            n,
            scheme,
        )
        if scheme == "iid":
            return _iid_partition_patients(index_groups, num_clients, rng, min_per_client)
        if scheme == "dirichlet":
            return _dirichlet_partition_patients(
                patient_labels, index_groups, num_clients, alpha, min_per_client, max_retries, rng
            )
        if scheme == "quantity_skew":
            return _quantity_skew_partition_patients(
                index_groups, num_clients, quantity_skew_sigma, rng, min_per_client
            )
        raise ValueError(f"Unknown partitioning scheme: {scheme!r}")

    if scheme == "iid":
        return _iid_partition(n, num_clients, rng)
    if scheme == "dirichlet":
        return _dirichlet_partition(
            labels, num_clients, alpha, min_per_client, max_retries, rng
        )
    if scheme == "quantity_skew":
        return _quantity_skew_partition(n, num_clients, quantity_skew_sigma, rng)
    raise ValueError(f"Unknown partitioning scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# Patient-grouping helper
# ---------------------------------------------------------------------------

def _patient_groups(
    labels: np.ndarray,
    patient_ids: np.ndarray,
) -> tuple[list[Any], np.ndarray, list[list[int]]]:
    """Group sample indices by patient, preserving encounter order.

    Returns:
        unique_pids: Ordered list of unique patient IDs.
        patient_labels: (P,) majority label per patient.
        index_groups: List of P index-lists (one per patient).
    """
    unique_pids: list[Any] = list(dict.fromkeys(patient_ids.tolist()))
    pid_to_pos = {pid: i for i, pid in enumerate(unique_pids)}
    P = len(unique_pids)
    index_groups: list[list[int]] = [[] for _ in range(P)]
    for sample_i, pid in enumerate(patient_ids.tolist()):
        index_groups[pid_to_pos[pid]].append(sample_i)
    patient_labels = np.array(
        [int(np.bincount(labels[grp]).argmax()) for grp in index_groups],
        dtype=np.int64,
    )
    return unique_pids, patient_labels, index_groups


# ---------------------------------------------------------------------------
# Patient-aware partition schemes
# ---------------------------------------------------------------------------

def _iid_partition_patients(
    index_groups: list[list[int]],
    num_clients: int,
    rng: np.random.Generator,
    min_per_client: int = 1,
) -> list[list[int]]:
    """IID partition at the patient level."""
    P = len(index_groups)
    patient_order = np.arange(P)
    rng.shuffle(patient_order)
    patient_splits = np.array_split(patient_order, num_clients)
    client_idx: list[list[int]] = []
    for pid_indices in patient_splits:
        indices: list[int] = []
        for pi in pid_indices:
            indices.extend(index_groups[pi])
        rng.shuffle(indices)
        client_idx.append(indices)

    # Enforce min_per_client by redistributing from the largest client.
    sizes = [len(c) for c in client_idx]
    for ci, sz in enumerate(sizes):
        if sz < min_per_client:
            donor = int(np.argmax(sizes))
            needed = min_per_client - sz
            if sizes[donor] - needed >= min_per_client:
                client_idx[ci].extend(client_idx[donor][-needed:])
                client_idx[donor] = client_idx[donor][:-needed]
                sizes[ci] = len(client_idx[ci])
                sizes[donor] = len(client_idx[donor])
            else:
                _logger.warning(
                    "IID patient partition: client %d has %d samples < min_per_client=%d "
                    "and redistribution is not possible. Increase dataset size or reduce "
                    "num_clients.",
                    ci,
                    sz,
                    min_per_client,
                )
    return client_idx


def _dirichlet_partition_patients(
    patient_labels: np.ndarray,
    index_groups: list[list[int]],
    num_clients: int,
    alpha: float,
    min_per_client: int,
    max_retries: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Non-IID Dirichlet partition at the patient level.

    Applies Dirichlet proportions per class at the patient level so that
    class skew is achieved without splitting patients across clients.
    The ``min_per_client`` constraint is checked in terms of *samples*
    (images), not patients.
    """
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")
    classes = np.unique(patient_labels)
    last_min = -1
    for attempt in range(max_retries):
        client_patient_lists: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            c_patient_idxs = np.where(patient_labels == c)[0]
            rng.shuffle(c_patient_idxs)
            proportions = rng.dirichlet([alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(c_patient_idxs)).astype(int)[:-1]
            parts = np.split(c_patient_idxs, split_points)
            for ci, p in enumerate(parts):
                client_patient_lists[ci].extend(p.tolist())

        # Expand patients → samples and check min constraint.
        client_idx: list[list[int]] = []
        for pi_list in client_patient_lists:
            indices: list[int] = []
            for pi in pi_list:
                indices.extend(index_groups[pi])
            client_idx.append(indices)

        sizes = [len(c) for c in client_idx]
        last_min = min(sizes)
        if last_min >= min_per_client:
            for c in client_idx:
                rng.shuffle(c)
            _logger.info(
                "Patient-Dirichlet partition (alpha=%.2f) succeeded on attempt %d; "
                "client sizes=%s",
                alpha,
                attempt + 1,
                sizes,
            )
            return client_idx
        _logger.debug(
            "Patient-Dirichlet retry %d/%d: min client size %d < %d",
            attempt + 1,
            max_retries,
            last_min,
            min_per_client,
        )
    raise RuntimeError(
        f"Patient-Dirichlet partitioning failed to meet min_per_client={min_per_client} "
        f"within {max_retries} retries (last min size={last_min}). "
        "Try a larger alpha, smaller min_per_client, or fewer clients."
    )


def _quantity_skew_partition_patients(
    index_groups: list[list[int]],
    num_clients: int,
    sigma: float,
    rng: np.random.Generator,
    min_per_client: int = 1,
) -> list[list[int]]:
    """Quantity-skew partition at the patient level."""
    if sigma < 0:
        raise ValueError(f"quantity_skew_sigma must be >= 0, got {sigma}")
    P = len(index_groups)
    patient_order = np.arange(P)
    rng.shuffle(patient_order)
    raw = rng.lognormal(mean=0.0, sigma=sigma, size=num_clients)
    proportions = raw / raw.sum()
    n_per_client = (proportions * P).astype(int)
    diff = P - int(n_per_client.sum())
    if diff != 0:
        n_per_client[0] += diff
    client_idx: list[list[int]] = []
    cursor = 0
    for count in n_per_client:
        indices: list[int] = []
        for pi in patient_order[cursor: cursor + count]:
            indices.extend(index_groups[pi])
        client_idx.append(indices)
        cursor += count

    # Enforce min_per_client by redistributing from the largest client.
    sizes = [len(c) for c in client_idx]
    for ci, sz in enumerate(sizes):
        if sz < min_per_client:
            donor = int(np.argmax(sizes))
            needed = min_per_client - sz
            if sizes[donor] - needed >= min_per_client:
                client_idx[ci].extend(client_idx[donor][-needed:])
                client_idx[donor] = client_idx[donor][:-needed]
                sizes[ci] = len(client_idx[ci])
                sizes[donor] = len(client_idx[donor])
            else:
                _logger.warning(
                    "Quantity-skew patient partition: client %d has %d samples < "
                    "min_per_client=%d and redistribution is not possible.",
                    ci,
                    sz,
                    min_per_client,
                )
    return client_idx


# ---------------------------------------------------------------------------
# Original sample-level partition schemes (unchanged)
# ---------------------------------------------------------------------------

def _iid_partition(n: int, num_clients: int, rng: np.random.Generator) -> list[list[int]]:
    idx = np.arange(n)
    rng.shuffle(idx)
    parts = np.array_split(idx, num_clients)
    return [p.tolist() for p in parts]


def _dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_per_client: int,
    max_retries: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Standard non-IID partition: per class, sample Dir(alpha) proportions."""
    if alpha <= 0:
        raise ValueError(f"Dirichlet alpha must be > 0, got {alpha}")
    classes = np.unique(labels)
    last_min = -1
    for attempt in range(max_retries):
        client_idx: list[list[int]] = [[] for _ in range(num_clients)]
        for c in classes:
            c_indices = np.where(labels == c)[0]
            rng.shuffle(c_indices)
            proportions = rng.dirichlet([alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(c_indices)).astype(int)[:-1]
            parts = np.split(c_indices, split_points)
            for ci, p in enumerate(parts):
                client_idx[ci].extend(p.tolist())
        sizes = [len(c) for c in client_idx]
        last_min = min(sizes)
        if last_min >= min_per_client:
            for c in client_idx:
                rng.shuffle(c)
            _logger.info(
                "Dirichlet partition (alpha=%.2f) succeeded on attempt %d; "
                "client sizes=%s",
                alpha,
                attempt + 1,
                sizes,
            )
            return client_idx
        _logger.debug(
            "Dirichlet retry %d/%d: min client size %d < %d",
            attempt + 1,
            max_retries,
            last_min,
            min_per_client,
        )
    raise RuntimeError(
        f"Dirichlet partitioning failed to meet min_per_client={min_per_client} "
        f"within {max_retries} retries (last min size={last_min}). "
        "Try a larger alpha, smaller min_per_client, or fewer clients."
    )


def _quantity_skew_partition(
    n: int, num_clients: int, sigma: float, rng: np.random.Generator
) -> list[list[int]]:
    """IID partition with log-normally distributed client sizes."""
    if sigma < 0:
        raise ValueError(f"quantity_skew_sigma must be >= 0, got {sigma}")
    raw = rng.lognormal(mean=0.0, sigma=sigma, size=num_clients)
    proportions = raw / raw.sum()
    sizes = (proportions * n).astype(int)
    # Make sure totals match n exactly.
    diff = n - int(sizes.sum())
    if diff != 0:
        sizes[0] += diff
    idx = np.arange(n)
    rng.shuffle(idx)
    client_idx: list[list[int]] = []
    cursor = 0
    for s in sizes:
        client_idx.append(idx[cursor : cursor + s].tolist())
        cursor += s
    return client_idx


__all__ = ["partition_indices", "Scheme"]
