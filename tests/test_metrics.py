"""Metrics, checked against hand-computed values.

Plan reference: sections 54, 55, 56, 66.6.

Section 66.6: "Test formulas against small arrays with hand-computed results."
Every expected value below is derived by hand rather than by calling another
implementation, so the tests cannot agree with a shared bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralcosmos.evaluation.metrics import (
    aggregate_by_simulation,
    generalization_ratio,
    mae,
    mean_relative_error,
    nrmse,
    r2,
    regression_metrics,
    rmse,
    selection_score,
)


# --------------------------------------------------------------------------
# Hand-computed values
# --------------------------------------------------------------------------


def test_mae_hand_computed():
    # errors 1, 2, 3 -> mean 2
    assert mae([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(2.0)


def test_rmse_hand_computed():
    # errors 1, 2, 3 -> sqrt((1+4+9)/3) = sqrt(14/3)
    assert rmse([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(np.sqrt(14 / 3))


def test_rmse_is_never_below_mae():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=200), rng.normal(size=200)
    assert rmse(a, b) >= mae(a, b) - 1e-12


def test_r2_is_one_for_perfect_prediction():
    y = np.array([0.1, 0.3, 0.5])
    assert r2(y, y) == pytest.approx(1.0)


def test_r2_is_zero_for_predicting_the_mean():
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_r2_is_negative_for_worse_than_the_mean():
    y = np.array([1.0, 2.0, 3.0])
    assert r2(y, np.array([3.0, 2.0, 1.0])) < 0


def test_r2_undefined_for_constant_truth():
    assert np.isnan(r2([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]))


def test_mean_relative_error_hand_computed():
    # |0.1-0.2|/0.1 = 1.0 ; |0.4-0.3|/0.4 = 0.25 ; mean = 0.625
    assert mean_relative_error([0.1, 0.4], [0.2, 0.3]) == pytest.approx(0.625)


def test_relative_error_rejects_zero_target():
    with pytest.raises(ValueError, match="undefined for targets at zero"):
        mean_relative_error([0.0, 1.0], [1.0, 1.0])


def test_nrmse_uses_the_supplied_span():
    y = np.array([0.1, 0.5])
    p = np.array([0.2, 0.4])
    # RMSE = 0.1 ; span 0.4 -> 0.25
    assert nrmse(y, p, span=0.4) == pytest.approx(0.25)


def test_nrmse_span_defaults_to_observed_range():
    y = np.array([0.0, 2.0])
    p = np.array([1.0, 1.0])
    assert nrmse(y, p) == pytest.approx(1.0 / 2.0)


def test_nrmse_rejects_zero_span():
    with pytest.raises(ValueError, match="non-positive normalisation span"):
        nrmse([1.0, 1.0], [1.0, 1.0])


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        mae([1.0, 2.0], [1.0, 2.0, 3.0])


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        mae([], [])


# --------------------------------------------------------------------------
# Multi-target reporting
# --------------------------------------------------------------------------


def test_regression_metrics_reports_each_target_separately():
    y = np.array([[0.2, 0.7], [0.4, 0.9]])
    p = np.array([[0.3, 0.7], [0.4, 0.8]])
    out = regression_metrics(y, p, ["omega_m", "sigma8"], spans=[0.4, 0.4])

    assert out["n"] == 2
    # omega_m errors: 0.1, 0.0 -> MAE 0.05
    assert out["per_target"]["omega_m"]["mae"] == pytest.approx(0.05)
    # sigma8 errors: 0.0, 0.1 -> MAE 0.05
    assert out["per_target"]["sigma8"]["mae"] == pytest.approx(0.05)
    assert out["mean_mae"] == pytest.approx(0.05)


def test_regression_metrics_rejects_name_mismatch():
    y = np.zeros((3, 2))
    with pytest.raises(ValueError, match="2 targets but 3 names"):
        regression_metrics(y, y, ["a", "b", "c"])


def test_regression_metrics_requires_two_dimensions():
    with pytest.raises(ValueError, match=r"expected \(N, T\)"):
        regression_metrics(np.zeros(4), np.zeros(4), ["a"])


# --------------------------------------------------------------------------
# Simulation-level aggregation (section 54)
# --------------------------------------------------------------------------


def test_aggregation_averages_maps_of_one_simulation():
    y = np.array([[0.3, 0.8]] * 4)
    p = np.array([[0.1, 0.8], [0.2, 0.8], [0.4, 0.8], [0.5, 0.8]])
    sims = np.array([0, 0, 0, 0])

    t, pr, keys = aggregate_by_simulation(y, p, sims)
    assert t.shape == (1, 2)
    assert pr[0, 0] == pytest.approx(0.3)   # mean of 0.1,0.2,0.4,0.5
    assert keys.tolist() == [0]


def test_aggregation_separates_distinct_simulations():
    y = np.array([[0.2, 0.7], [0.2, 0.7], [0.4, 0.9], [0.4, 0.9]])
    p = np.array([[0.1, 0.7], [0.3, 0.7], [0.5, 0.9], [0.3, 0.9]])
    sims = np.array([0, 0, 1, 1])

    t, pr, keys = aggregate_by_simulation(y, p, sims)
    assert t.shape == (2, 2)
    assert pr[0, 0] == pytest.approx(0.2)
    assert pr[1, 0] == pytest.approx(0.4)


def test_aggregation_does_not_merge_same_id_across_suites():
    """Simulation 7 of TNG and simulation 7 of SIMBA are unrelated universes.

    Averaging them together because they share an integer would silently
    corrupt every simulation-level number in the paper.
    """
    y = np.array([[0.2, 0.7], [0.4, 0.9]])
    p = np.array([[0.2, 0.7], [0.4, 0.9]])
    sims = np.array([7, 7])
    suites = np.array([0, 1])

    t, pr, keys = aggregate_by_simulation(y, p, sims, suites)
    assert t.shape == (2, 2), "predictions from two suites were merged"


def test_aggregation_with_single_suite_ignores_suite_key():
    y = np.array([[0.2, 0.7], [0.2, 0.7]])
    p = np.array([[0.1, 0.7], [0.3, 0.7]])
    t, pr, _ = aggregate_by_simulation(y, p, np.array([3, 3]), np.array([0, 0]))
    assert t.shape == (1, 2)


def test_aggregation_reduces_error_for_unbiased_noise():
    """Averaging 15 views should beat a single view when errors are independent."""
    rng = np.random.default_rng(3)
    n_sims, per_sim = 40, 15
    truth = rng.uniform(0.1, 0.5, size=(n_sims, 1))

    y = np.repeat(truth, per_sim, axis=0)
    noise = rng.normal(scale=0.05, size=y.shape)
    p = y + noise
    sims = np.repeat(np.arange(n_sims), per_sim)

    t_sim, p_sim, _ = aggregate_by_simulation(y, p, sims)
    assert rmse(t_sim, p_sim) < rmse(y, p)


def test_aggregation_rejects_length_mismatch():
    y = np.zeros((4, 2))
    with pytest.raises(ValueError, match="simulation_ids length"):
        aggregate_by_simulation(y, y, np.array([0, 1]))


# --------------------------------------------------------------------------
# Selection and DG metrics
# --------------------------------------------------------------------------


def test_selection_score_averages_normalised_rmse():
    y = np.array([[0.1, 0.6], [0.5, 1.0]])
    p = np.array([[0.2, 0.7], [0.4, 0.9]])
    # Both targets: RMSE 0.1, span 0.4 -> NRMSE 0.25 each -> mean 0.25
    assert selection_score(y, p, [0.4, 0.4]) == pytest.approx(0.25)


def test_selection_score_is_zero_for_perfect_prediction():
    y = np.array([[0.3, 0.8]])
    assert selection_score(y, y, [0.4, 0.4]) == pytest.approx(0.0)


def test_generalization_ratio():
    assert generalization_ratio(0.06, 0.02) == pytest.approx(3.0)
    assert generalization_ratio(0.02, 0.02) == pytest.approx(1.0)


def test_generalization_ratio_rejects_zero_denominator():
    with pytest.raises(ValueError, match="must be positive"):
        generalization_ratio(0.05, 0.0)


def test_kill_criterion_threshold_is_expressible():
    """Section 35: a ratio consistently below ~1.2-1.3 means the shift is weak."""
    assert generalization_ratio(0.023, 0.020) < 1.2
    assert generalization_ratio(0.070, 0.020) > 1.3
