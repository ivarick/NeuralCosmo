"""Regression and domain-generalization metrics.

Plan reference: sections 54, 55, 56, 62.

All metrics are computed in PHYSICAL units. Predictions arrive in the
range-normalised space of section 22 and must be passed back through
``TargetScaler.inverse`` before reaching anything here, so that a reported MAE
means "0.02 in Omega_m" rather than "0.02 of an arbitrary interval".
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = [
    "mae",
    "rmse",
    "r2",
    "mean_relative_error",
    "nrmse",
    "regression_metrics",
    "aggregate_by_simulation",
    "selection_score",
    "generalization_ratio",
]


def _check(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty arrays")
    return yt, yp


def mae(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination.

    Section 55 says to report it for comparability but not to rely on it alone:
    R^2 is sensitive to the spread of the evaluation set, so it can look
    healthy on a wide parameter range while the model is regressing to the mean.
    """
    yt, yp = _check(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mean_relative_error(y_true, y_pred) -> float:
    """Mean |error| / |truth|.

    Safe here only because both targets stay well away from zero
    (Omega_m >= 0.1, sigma8 >= 0.6), as section 55 notes.
    """
    yt, yp = _check(y_true, y_pred)
    if np.any(np.abs(yt) < 1e-12):
        raise ValueError("relative error is undefined for targets at zero")
    return float(np.mean(np.abs(yt - yp) / np.abs(yt)))


def nrmse(y_true, y_pred, span: float | None = None) -> float:
    """RMSE normalised by the target's range.

    Used to combine differently scaled targets into one selection score
    (section 62). ``span`` defaults to the observed range of ``y_true``, but
    should be passed explicitly as the fixed design range so the score is
    comparable across evaluation sets of differing spread.
    """
    yt, yp = _check(y_true, y_pred)
    denom = span if span is not None else float(yt.max() - yt.min())
    if not denom > 0:
        raise ValueError(f"non-positive normalisation span: {denom}")
    return rmse(yt, yp) / denom


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str],
    spans: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Per-target metrics plus their mean, for an (N, T) prediction array."""
    yt, yp = _check(y_true, y_pred)
    if yt.ndim != 2:
        raise ValueError(f"expected (N, T) arrays, got {yt.shape}")
    if yt.shape[1] != len(target_names):
        raise ValueError(f"{yt.shape[1]} targets but {len(target_names)} names")

    out: dict[str, Any] = {"n": int(yt.shape[0]), "per_target": {}}
    for i, name in enumerate(target_names):
        span = spans[i] if spans is not None else None
        out["per_target"][name] = {
            "mae": mae(yt[:, i], yp[:, i]),
            "rmse": rmse(yt[:, i], yp[:, i]),
            "r2": r2(yt[:, i], yp[:, i]),
            "mean_relative_error": mean_relative_error(yt[:, i], yp[:, i]),
            "nrmse": nrmse(yt[:, i], yp[:, i], span=span),
        }
    for key in ("mae", "rmse", "r2", "nrmse"):
        out[f"mean_{key}"] = float(
            np.mean([out["per_target"][n][key] for n in target_names])
        )
    return out


def aggregate_by_simulation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    simulation_ids: np.ndarray,
    suite_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average the 15 map predictions of each simulation (section 54).

    Simulation IDs are only unique *within* a suite -- simulation 7 of TNG and
    simulation 7 of SIMBA are unrelated universes -- so ``suite_ids`` must be
    supplied whenever more than one suite is present, or predictions from
    different suites would be silently averaged together.
    """
    yt, yp = _check(y_true, y_pred)
    sims = np.asarray(simulation_ids)
    if sims.shape[0] != yt.shape[0]:
        raise ValueError("simulation_ids length does not match predictions")

    if suite_ids is None:
        keys = sims
    else:
        suites = np.asarray(suite_ids)
        if suites.shape[0] != yt.shape[0]:
            raise ValueError("suite_ids length does not match predictions")
        if len(np.unique(suites)) > 1:
            # Compose a unique key per (suite, simulation).
            keys = suites.astype(np.int64) * (int(sims.max()) + 1) + sims.astype(np.int64)
        else:
            keys = sims

    unique, inverse = np.unique(keys, return_inverse=True)
    n_groups = unique.shape[0]
    counts = np.bincount(inverse, minlength=n_groups).astype(np.float64)

    true_sum = np.zeros((n_groups, yt.shape[1]))
    pred_sum = np.zeros((n_groups, yp.shape[1]))
    for j in range(yt.shape[1]):
        true_sum[:, j] = np.bincount(inverse, weights=yt[:, j], minlength=n_groups)
        pred_sum[:, j] = np.bincount(inverse, weights=yp[:, j], minlength=n_groups)

    return true_sum / counts[:, None], pred_sum / counts[:, None], unique


def selection_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    spans: Sequence[float],
) -> float:
    """The checkpoint-selection metric of section 62.

    S = mean over targets of NRMSE. Fixed before training, computed on SOURCE
    validation only, and never on target data.
    """
    yt, yp = _check(y_true, y_pred)
    return float(np.mean([nrmse(yt[:, i], yp[:, i], spans[i]) for i in range(yt.shape[1])]))


def generalization_ratio(ood_error: float, id_error: float) -> float:
    """G = OOD error / ID error (section 56).

    G > 1 means the simulator shift costs accuracy. Section 35 sets the project
    decision threshold: if G stays below roughly 1.2-1.3 across both targets and
    both transfer directions, the shift may be too weak to motivate the method.
    """
    if id_error <= 0:
        raise ValueError(f"in-domain error must be positive, got {id_error}")
    return ood_error / id_error
