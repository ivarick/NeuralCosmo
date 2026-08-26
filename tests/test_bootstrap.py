"""Simulation-level bootstrap.

Plan reference: sections 54, 57, 66.6.

The central test is the one that would catch the mistake section 57 exists to
prevent: bootstrapping maps as if they were independent produces an interval
roughly sqrt(15) times too narrow. That error is invisible in the output -- the
interval simply looks better -- so it has to be asserted against.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralcosmos.evaluation.bootstrap import (
    bootstrap_difference,
    bootstrap_metric,
    group_keys,
)
from neuralcosmos.evaluation.metrics import mae

MAPS_PER_SIM = 15


def _correlated_dataset(n_sims: int = 60, seed: int = 0):
    """Maps whose error is shared within a simulation, as in the real data.

    Each simulation gets one systematic offset applied to all 15 of its maps,
    plus small per-map noise. This is the correlation structure that makes
    map-level bootstrapping wrong.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.1, 0.5, size=n_sims)
    per_sim_bias = rng.normal(scale=0.03, size=n_sims)

    y_true = np.repeat(truth, MAPS_PER_SIM)
    bias = np.repeat(per_sim_bias, MAPS_PER_SIM)
    y_pred = y_true + bias + rng.normal(scale=0.002, size=y_true.shape)
    sims = np.repeat(np.arange(n_sims), MAPS_PER_SIM)
    return y_true, y_pred, sims


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def test_group_keys_separate_suites_sharing_a_simulation_index():
    """Simulation 7 of one suite is not simulation 7 of another."""
    sims = np.array([7, 7, 3, 3])
    suites = np.array([0, 1, 0, 1])
    keys = group_keys(sims, suites)
    assert len(np.unique(keys)) == 4


def test_group_keys_without_suites_uses_simulation_alone():
    sims = np.array([1, 1, 2, 2])
    assert np.array_equal(group_keys(sims), sims)


def test_group_keys_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        group_keys(np.array([1, 2, 3]), np.array([0, 1]))


# --------------------------------------------------------------------------
# The property section 57 exists to enforce
# --------------------------------------------------------------------------


def test_map_level_bootstrap_would_be_far_too_narrow():
    """The mistake section 57 forbids, measured.

    Treating each of the 15 maps as an independent sample inflates the
    effective sample size ~15x, so the interval shrinks by roughly sqrt(15).
    An interval computed that way looks better and is wrong.
    """
    y_true, y_pred, sims = _correlated_dataset()

    correct = bootstrap_metric(mae, y_true, y_pred, sims, n_replicates=400, seed=0)

    # The wrong way: every map its own group.
    fake_ids = np.arange(y_true.shape[0])
    wrong = bootstrap_metric(mae, y_true, y_pred, fake_ids, n_replicates=400, seed=0)

    correct_width = correct.upper - correct.lower
    wrong_width = wrong.upper - wrong.lower
    assert wrong_width < correct_width / 2, (
        f"map-level bootstrap should be far narrower; got {wrong_width:.5f} "
        f"vs {correct_width:.5f}"
    )


def test_all_maps_of_a_sampled_simulation_are_included():
    """Section 57 step 2. Subsampling maps would reintroduce independence."""
    y_true, y_pred, sims = _correlated_dataset(n_sims=10)
    res = bootstrap_metric(mae, y_true, y_pred, sims, n_replicates=50, seed=0)
    # n_groups counts simulations, not maps.
    assert res.n_groups == 10
    assert res.n_samples == 10 * MAPS_PER_SIM


# --------------------------------------------------------------------------
# Interval behaviour
# --------------------------------------------------------------------------


def test_interval_brackets_the_point_estimate():
    y_true, y_pred, sims = _correlated_dataset()
    res = bootstrap_metric(mae, y_true, y_pred, sims, n_replicates=500, seed=1)
    assert res.lower <= res.point <= res.upper


def test_more_simulations_give_a_narrower_interval():
    small = bootstrap_metric(mae, *_correlated_dataset(n_sims=20, seed=2),
                             n_replicates=400, seed=0)
    large = bootstrap_metric(mae, *_correlated_dataset(n_sims=200, seed=2),
                             n_replicates=400, seed=0)
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_a_wider_confidence_level_gives_a_wider_interval():
    y_true, y_pred, sims = _correlated_dataset()
    narrow = bootstrap_metric(mae, y_true, y_pred, sims, n_replicates=400,
                              confidence=0.68, seed=0)
    wide = bootstrap_metric(mae, y_true, y_pred, sims, n_replicates=400,
                            confidence=0.99, seed=0)
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_perfect_predictions_give_a_zero_width_interval():
    y = np.repeat(np.linspace(0.1, 0.5, 20), MAPS_PER_SIM)
    sims = np.repeat(np.arange(20), MAPS_PER_SIM)
    res = bootstrap_metric(mae, y, y.copy(), sims, n_replicates=100, seed=0)
    assert res.point == pytest.approx(0.0)
    assert res.upper == pytest.approx(0.0)


def test_same_seed_reproduces_the_interval():
    y_true, y_pred, sims = _correlated_dataset()
    a = bootstrap_metric(mae, y_true, y_pred, sims, n_replicates=200, seed=7)
    b = bootstrap_metric(mae, y_true, y_pred, sims, n_replicates=200, seed=7)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_suite_ids_are_respected_in_grouping():
    """Two suites reusing the same simulation indices are 2N units, not N.

    Both suites number their simulations 0..N-1, but those are different
    universes from different codes. Without the suite in the key they would
    collapse into N groups and the two suites would resample together.
    """
    y_true, y_pred, sims = _correlated_dataset(n_sims=20)

    # Stack the same simulation indices under two different suites.
    y_true2 = np.concatenate([y_true, y_true])
    y_pred2 = np.concatenate([y_pred, y_pred])
    sims2 = np.concatenate([sims, sims])
    suites2 = np.concatenate([np.zeros_like(sims), np.ones_like(sims)])

    with_suite = bootstrap_metric(mae, y_true2, y_pred2, sims2, suites2,
                                  n_replicates=100, seed=0)
    without_suite = bootstrap_metric(mae, y_true2, y_pred2, sims2, None,
                                     n_replicates=100, seed=0)

    assert with_suite.n_groups == 40
    assert without_suite.n_groups == 20


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_single_simulation_is_rejected():
    y = np.ones(15)
    with pytest.raises(ValueError, match="at least 2 simulations"):
        bootstrap_metric(mae, y, y, np.zeros(15, dtype=int), n_replicates=10)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        bootstrap_metric(mae, np.zeros(30), np.zeros(15),
                         np.repeat(np.arange(2), 15), n_replicates=10)


def test_invalid_confidence_is_rejected():
    y_true, y_pred, sims = _correlated_dataset(n_sims=5)
    with pytest.raises(ValueError, match="confidence must be"):
        bootstrap_metric(mae, y_true, y_pred, sims, confidence=1.5, n_replicates=10)


# --------------------------------------------------------------------------
# Differences between models
# --------------------------------------------------------------------------


def test_difference_detects_a_real_gap():
    y_true, y_pred_a, sims = _correlated_dataset(seed=3)
    y_pred_b = y_true + (y_pred_a - y_true) * 0.3      # clearly better model

    out = bootstrap_difference(
        mae, y_true, y_pred_a, y_true, y_pred_b, sims, sims,
        n_replicates=400, seed=0, paired=True,
    )
    assert out["difference"] < 0
    assert out["excludes_zero"]


def test_difference_does_not_manufacture_a_gap():
    y_true, y_pred, sims = _correlated_dataset(seed=4)
    out = bootstrap_difference(
        mae, y_true, y_pred, y_true, y_pred.copy(), sims, sims,
        n_replicates=300, seed=0, paired=True,
    )
    assert out["difference"] == pytest.approx(0.0)
    assert not out["excludes_zero"]


def test_paired_resampling_is_tighter_than_unpaired():
    """Two models on the same test simulations share their sampling variance.

    Resampling each arm independently ignores that and inflates the interval on
    the difference, which is the quantity actually being compared.
    """
    y_true, y_pred_a, sims = _correlated_dataset(seed=5)
    y_pred_b = y_true + (y_pred_a - y_true) * 0.8

    paired = bootstrap_difference(
        mae, y_true, y_pred_a, y_true, y_pred_b, sims, sims,
        n_replicates=400, seed=0, paired=True,
    )
    unpaired = bootstrap_difference(
        mae, y_true, y_pred_a, y_true, y_pred_b, sims, sims,
        n_replicates=400, seed=0, paired=False,
    )
    w_paired = paired["ci_upper"] - paired["ci_lower"]
    w_unpaired = unpaired["ci_upper"] - unpaired["ci_lower"]
    assert w_paired < w_unpaired


def test_paired_requires_matching_simulations():
    y_true, y_pred, sims = _correlated_dataset(n_sims=10)
    other = sims + 100
    with pytest.raises(ValueError, match="same simulations"):
        bootstrap_difference(
            mae, y_true, y_pred, y_true, y_pred, sims, other,
            n_replicates=10, paired=True,
        )
