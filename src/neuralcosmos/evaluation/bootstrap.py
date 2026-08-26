"""Confidence intervals by resampling simulations, never maps.

Plan reference: section 57.

Section 57 states the rule and the reason:

    "Because 15 maps from one simulation are correlated, do not bootstrap
     individual maps as independent samples. Bootstrap by simulation ID."

This is not a technicality. The 15 maps of a simulation are 15 projections of
one universe sharing one parameter vector and one realisation of the density
field. Treating them as independent inflates the effective sample size roughly
15-fold, and a confidence interval computed that way is about sqrt(15) ~ 3.9x
too narrow. Every interval would look impressive and none would be honest.

The procedure of section 57:

    1. sample simulation IDs with replacement
    2. include ALL maps belonging to each sampled simulation
    3. recompute the metric
    4. repeat
    5. report the percentile interval

Step 2 is the one that is easy to get wrong: resampling simulations but then
subsampling their maps would quietly reintroduce the independence assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "BootstrapResult",
    "group_keys",
    "bootstrap_metric",
    "bootstrap_difference",
]

# Section 57 leaves the replicate count to us and asks that it be documented.
# 2000 is the plan's own example and is comfortably converged for a percentile
# interval on a few hundred groups.
DEFAULT_REPLICATES = 2000


@dataclass
class BootstrapResult:
    """A point estimate with a resampling interval."""

    point: float
    lower: float
    upper: float
    mean: float
    std: float
    n_groups: int
    n_samples: int
    n_replicates: int
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "bootstrap_mean": self.mean,
            "bootstrap_std": self.std,
            "n_groups": self.n_groups,
            "n_samples": self.n_samples,
            "n_replicates": self.n_replicates,
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def group_keys(
    simulation_ids: np.ndarray,
    suite_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Composite (suite, simulation) key identifying one resampling unit.

    Simulation 7 of IllustrisTNG and simulation 7 of SIMBA are unrelated
    universes from different codes. Resampling on the bare simulation index
    would treat them as the same unit and couple two suites that share nothing.
    """
    sims = np.asarray(simulation_ids).astype(np.int64)
    if suite_ids is None:
        return sims
    suites = np.asarray(suite_ids).astype(np.int64)
    if suites.shape != sims.shape:
        raise ValueError("suite_ids and simulation_ids must have the same shape")
    return suites * (int(sims.max()) + 1) + sims


def bootstrap_metric(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    simulation_ids: np.ndarray,
    suite_ids: np.ndarray | None = None,
    n_replicates: int = DEFAULT_REPLICATES,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Percentile interval for a metric, resampling whole simulations.

    ``metric_fn`` receives ``(y_true, y_pred)`` subsets and returns a scalar.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")

    keys = group_keys(simulation_ids, suite_ids)
    unique, inverse = np.unique(keys, return_inverse=True)
    n_groups = unique.shape[0]
    if n_groups < 2:
        raise ValueError(
            f"bootstrap needs at least 2 simulations, got {n_groups}. "
            f"An interval from one group is meaningless."
        )

    # Precompute each group's member indices once. Resampling then costs a
    # concatenate rather than a scan over the whole array per replicate.
    order = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[order]
    boundaries = np.searchsorted(sorted_groups, np.arange(n_groups + 1))
    members = [order[boundaries[g] : boundaries[g + 1]] for g in range(n_groups)]

    point = float(metric_fn(yt, yp))

    rng = np.random.default_rng(seed)
    stats = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        picked = rng.integers(0, n_groups, size=n_groups)
        # ALL maps of each sampled simulation, per section 57 step 2.
        idx = np.concatenate([members[g] for g in picked])
        stats[r] = metric_fn(yt[idx], yp[idx])

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])

    return BootstrapResult(
        point=point,
        lower=float(lower),
        upper=float(upper),
        mean=float(stats.mean()),
        std=float(stats.std(ddof=1)),
        n_groups=n_groups,
        n_samples=int(yt.shape[0]),
        n_replicates=n_replicates,
        confidence=confidence,
    )


def bootstrap_difference(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    y_true_b: np.ndarray,
    y_pred_b: np.ndarray,
    simulation_ids_a: np.ndarray,
    simulation_ids_b: np.ndarray,
    suite_ids_a: np.ndarray | None = None,
    suite_ids_b: np.ndarray | None = None,
    n_replicates: int = DEFAULT_REPLICATES,
    confidence: float = 0.95,
    seed: int = 0,
    paired: bool = False,
) -> dict[str, Any]:
    """Interval for ``metric(B) - metric(A)``.

    ``paired=True`` resamples ONE set of simulations and applies it to both
    sides, which is correct when the two arms are two models evaluated on the
    same test simulations. That removes the shared test-set variance and gives
    a far tighter interval on the difference than resampling each arm
    independently -- the comparison of interest is "is B better than A on this
    data", not "are their absolute errors distinguishable".
    """
    yta, ypa = np.asarray(y_true_a), np.asarray(y_pred_a)
    ytb, ypb = np.asarray(y_true_b), np.asarray(y_pred_b)

    keys_a = group_keys(simulation_ids_a, suite_ids_a)
    keys_b = group_keys(simulation_ids_b, suite_ids_b)

    if paired:
        if not np.array_equal(np.unique(keys_a), np.unique(keys_b)):
            raise ValueError(
                "paired=True requires both arms to be evaluated on the same "
                "simulations; their group keys differ."
            )

    ua, inv_a = np.unique(keys_a, return_inverse=True)
    ub, inv_b = np.unique(keys_b, return_inverse=True)

    def _members(inv: np.ndarray, n: int) -> list[np.ndarray]:
        order = np.argsort(inv, kind="stable")
        s = inv[order]
        b = np.searchsorted(s, np.arange(n + 1))
        return [order[b[g] : b[g + 1]] for g in range(n)]

    mem_a = _members(inv_a, ua.shape[0])
    mem_b = _members(inv_b, ub.shape[0])

    point = float(metric_fn(ytb, ypb)) - float(metric_fn(yta, ypa))

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_replicates, dtype=np.float64)
    for r in range(n_replicates):
        pa = rng.integers(0, ua.shape[0], size=ua.shape[0])
        pb = pa if paired else rng.integers(0, ub.shape[0], size=ub.shape[0])
        ia = np.concatenate([mem_a[g] for g in pa])
        ib = np.concatenate([mem_b[g] for g in pb])
        diffs[r] = metric_fn(ytb[ib], ypb[ib]) - metric_fn(yta[ia], ypa[ia])

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])

    return {
        "difference": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_mean": float(diffs.mean()),
        "bootstrap_std": float(diffs.std(ddof=1)),
        # An interval excluding zero is the only basis on which one model may be
        # called better than another here.
        "excludes_zero": bool(lower > 0 or upper < 0),
        "paired": paired,
        "n_replicates": n_replicates,
        "confidence": confidence,
    }
