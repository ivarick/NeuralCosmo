"""Target scaling between physical and normalised units.

Plan reference: section 22.

Both targets are mapped into approximately [0, 1] using the FIXED Latin-
hypercube design ranges:

    omega_m_tilde = (omega_m - 0.1) / 0.4
    sigma8_tilde  = (sigma8  - 0.6) / 0.4

The distinction that matters for the domain-generalization protocol: these
constants come from the *design* of the CAMELS parameter sweep, which is public
and identical for every suite. They are not sample statistics estimated from
any particular suite's data. Using them therefore leaks no empirical
information about the sealed target suite, which a mean/std standardisation
fitted on target labels would.

All reported errors must be in PHYSICAL units, so every prediction is passed
back through :meth:`TargetScaler.inverse` before metrics are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = ["TargetScaler"]


@dataclass(frozen=True)
class TargetScaler:
    """Affine map between physical target values and [0, 1]."""

    names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (len(self.names) == len(self.lower) == len(self.upper)):
            raise ValueError("names, lower and upper must have equal length")
        for n, lo, hi in zip(self.names, self.lower, self.upper):
            if not hi > lo:
                raise ValueError(f"target {n!r} has a non-positive range: [{lo}, {hi}]")

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> TargetScaler:
        """Build from the ``targets`` block of a data config."""
        targets = cfg["targets"]
        scaling = targets.get("scaling", "fixed_lh_range")
        if scaling != "fixed_lh_range":
            raise NotImplementedError(
                f"Target scaling {scaling!r} is not implemented. Section 22 specifies "
                f"'fixed_lh_range'; any sample-statistic scaling must be justified "
                f"against the leakage rules of sections 20-22 before use."
            )
        names = tuple(targets["names"])
        ranges = targets["ranges"]
        return cls(
            names=names,
            lower=tuple(float(ranges[n][0]) for n in names),
            upper=tuple(float(ranges[n][1]) for n in names),
        )

    @property
    def n_targets(self) -> int:
        return len(self.names)

    @property
    def _lo(self) -> np.ndarray:
        return np.asarray(self.lower, dtype=np.float64)

    @property
    def _span(self) -> np.ndarray:
        return np.asarray(self.upper, dtype=np.float64) - self._lo

    def forward(self, physical: np.ndarray | Sequence[float]) -> np.ndarray:
        """Physical units -> approximately [0, 1]."""
        arr = np.asarray(physical, dtype=np.float64)
        return (arr - self._lo) / self._span

    def inverse(self, scaled: np.ndarray | Sequence[float]) -> np.ndarray:
        """Approximately [0, 1] -> physical units."""
        arr = np.asarray(scaled, dtype=np.float64)
        return arr * self._span + self._lo

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown target {name!r}; known: {self.names}") from exc

    def describe(self) -> str:
        parts = [
            f"{n}: [{lo}, {hi}] -> [0, 1]"
            for n, lo, hi in zip(self.names, self.lower, self.upper)
        ]
        return "; ".join(parts)
