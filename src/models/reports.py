"""Dataclass definition for state_dict load reports and matching diagnostics."""

from dataclasses import dataclass


@dataclass
class LoadReport:
    """Encapsulates state_dict tensor loading statistics and validation diagnostics.

    Attributes:
        matched: Total number of parameter tensors successfully mapped and loaded into the model.
        missing: List of parameter tensor names expected by the model architecture but absent in the checkpoint.
        unexpected: List of tensor names present in the filtered checkpoint but unmapped to model parameters.

    Raises:
        RuntimeError: Evaluated during `__post_init__` if `matched == 0`, indicating a total key mapping failure.
    """

    matched: int
    missing: list[str]
    unexpected: list[str]

    def __post_init__(self) -> None:
        """Validates that at least one tensor successfully matched during state_dict loading."""
        if self.matched == 0:
            raise RuntimeError(
                "0 tensors matched during checkpoint weight loading, although the "
                "remapped state_dict was non-empty. Verify the key_remap dictionary "
                "against actual model attribute names."
            )