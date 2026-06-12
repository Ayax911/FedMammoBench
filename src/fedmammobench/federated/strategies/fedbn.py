"""FedBN strategy — STUB.

FedBN (Li et al., 2021) keeps BatchNorm layers strictly local: clients do not
upload BN parameters or buffers, and the server does not aggregate them. The
intuition is that BN statistics encode site-specific intensity distributions
and aggregating them across hospitals hurts generalization.

Implementation plan when filled in:

1. On the **client**: when serializing parameters in ``get_parameters`` /
   ``fit``, drop entries whose key is a BatchNorm layer parameter or buffer.
   Symmetrically, when loading server-pushed parameters in
   ``set_parameters``, leave the existing local BN tensors untouched.
2. On the **server**: aggregation is identical to FedAvg over the
   non-BN parameter list. Practical implementation is a thin wrapper that
   uses the same FedAvg core but agrees with clients on the parameter
   ordering (the simplest convention is "filter BN by name from
   state_dict").

The naming convention is hard-coded knowledge of the model architecture,
which is why FedBN-as-a-strategy without client cooperation is an
incomplete spec. In this build the strategy raises clearly.
"""

from __future__ import annotations

from typing import Any

from flwr.server.strategy import Strategy

from fedmammo.federated.strategies.registry import register_strategy


@register_strategy("fedbn")
def build_fedbn(**_kwargs: Any) -> Strategy:
    raise NotImplementedError(
        "FedBN is registered but not implemented in this build. "
        "Implement the client-side BN exclusion in FedMammoClient.get_parameters / "
        "set_parameters and instantiate FedAvg here."
    )


__all__ = ["build_fedbn"]
