"""Power spectrum estimator, checked against analytically known inputs.

Plan reference: sections 4 (Phase 4), 66.6.

A power-spectrum bug is unusually dangerous here because the output is a smooth
plausible-looking curve whether or not it is right. Each test below uses an
input whose spectrum is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralcosmos.evaluation.spectra import (
    BOX_SIZE_MPC_H,
    mean_power_spectrum,
    power_spectrum_2d,
    radial_bins,
)


# --------------------------------------------------------------------------
# Binning geometry
# --------------------------------------------------------------------------


def test_k_range_matches_the_box():
    npix, box = 256, 25.0
    _, k, n_modes = radial_bins(npix, box, n_bins=24)
    finite = np.isfinite(k) & (n_modes > 0)

    k_fundamental = 2 * np.pi / box          # ~0.251 h/Mpc
    k_nyquist = np.pi * npix / box           # ~32.2 h/Mpc

    assert k[finite].min() >= k_fundamental * 0.9
    assert k[finite].max() <= k_nyquist * 1.1


def test_bin_centres_increase_monotonically():
    _, k, n = radial_bins(128, 25.0, n_bins=20)
    valid = np.isfinite(k) & (n > 0)
    assert np.all(np.diff(k[valid]) > 0)


def test_dc_mode_is_excluded():
    """The DC mode carries no fluctuation information and must not be binned."""
    idx, _, _ = radial_bins(64, 25.0, n_bins=12)
    assert idx[0, 0] == -1


def test_bins_are_populated_where_the_hypothesis_is_tested():
    """Low-k bins may legitimately be empty; the small-scale regime must not be.

    On a discrete grid the available |k| are k_fund * sqrt(nx^2 + ny^2), which
    are sparse at small radius, so a narrow log bin can fall in a gap. That is
    harmless as long as the k > 1 h/Mpc range the feedback hypothesis concerns
    is fully populated.
    """
    _, k, n_modes = radial_bins(256, 25.0, n_bins=24)
    populated = n_modes > 0
    assert populated.sum() >= 20, "too many empty bins overall"

    high_k = np.isfinite(k) & (k > 1.0)
    assert np.all(n_modes[high_k] > 0), "an empty bin in the small-scale regime"


def test_empty_bins_are_reported_as_nan_not_zero():
    """A zero would be read as zero power; NaN forces the caller to filter."""
    _, k, n_modes = radial_bins(256, 25.0, n_bins=24)
    assert np.all(np.isnan(k[n_modes == 0]))


# --------------------------------------------------------------------------
# Known spectra
# --------------------------------------------------------------------------


def test_white_noise_gives_a_flat_spectrum():
    """Uncorrelated pixels have equal power at every k, by construction."""
    npix, n_bins = 128, 16
    rng = np.random.default_rng(0)
    idx, k, _ = radial_bins(npix, BOX_SIZE_MPC_H, n_bins)

    acc = np.zeros(n_bins)
    n = 60
    for _ in range(n):
        field = 1.0 + 0.1 * rng.normal(size=(npix, npix))
        acc += np.nan_to_num(power_spectrum_2d(field, idx, n_bins))
    mean = acc / n

    valid = np.isfinite(k) & (mean > 0)
    spread = mean[valid].std() / mean[valid].mean()
    assert spread < 0.1, f"white-noise spectrum is not flat: relative spread {spread:.3f}"


def test_single_sine_mode_puts_power_at_its_own_k():
    """A pure sinusoid must deposit its power in the bin containing its k."""
    npix, box, n_bins = 128, BOX_SIZE_MPC_H, 16
    idx, k, _ = radial_bins(npix, box, n_bins)

    n_wave = 8                                  # 8 periods across the box
    k_true = 2 * np.pi * n_wave / box
    x = np.arange(npix) * box / npix
    field = 1.0 + 0.2 * np.sin(2 * np.pi * n_wave * x / box)[:, None] * np.ones((1, npix))

    pk = power_spectrum_2d(field, idx, n_bins)
    valid = np.isfinite(k) & np.isfinite(pk)
    peak_k = k[valid][np.argmax(pk[valid])]

    # The peak must land within one bin width of the true mode.
    rel = abs(peak_k - k_true) / k_true
    assert rel < 0.25, f"peak at k={peak_k:.2f}, expected {k_true:.2f}"


def test_spectrum_is_invariant_to_overall_normalisation():
    """P(k) is computed on rho/mean - 1, so scaling the map must change nothing.

    This matters directly: the suites differ slightly in overall amplitude, and
    a spectrum estimator sensitive to that would confound amplitude with shape.
    """
    npix, n_bins = 64, 12
    rng = np.random.default_rng(1)
    idx, _, _ = radial_bins(npix, BOX_SIZE_MPC_H, n_bins)

    field = 1e10 * (1.0 + 0.3 * rng.normal(size=(npix, npix)))
    field = np.abs(field) + 1.0

    a = power_spectrum_2d(field, idx, n_bins)
    b = power_spectrum_2d(field * 137.0, idx, n_bins)
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), rtol=1e-9)


def test_smoothing_suppresses_small_scale_power():
    """The exact signature the hypothesis predicts for SIMBA.

    Convolving a field with a Gaussian must reduce high-k power far more than
    low-k power. If the estimator cannot detect this on synthetic data, it
    cannot be trusted to detect it between suites.
    """
    from scipy.ndimage import gaussian_filter

    npix, n_bins = 128, 16
    rng = np.random.default_rng(2)
    idx, k, _ = radial_bins(npix, BOX_SIZE_MPC_H, n_bins)

    base = 1.0 + 0.5 * rng.normal(size=(npix, npix))
    base = np.abs(base) + 0.1
    smoothed = gaussian_filter(base, sigma=2.0, mode="wrap")

    p_base = power_spectrum_2d(base, idx, n_bins)
    p_smooth = power_spectrum_2d(smoothed, idx, n_bins)

    valid = np.isfinite(k) & (p_base > 0)
    ratio = p_smooth[valid] / p_base[valid]
    kv = k[valid]

    low = ratio[kv < np.median(kv)].mean()
    high = ratio[kv > np.median(kv)].mean()
    assert high < low, "smoothing did not preferentially suppress high-k power"


def test_nonpositive_mean_is_rejected():
    idx, _, _ = radial_bins(32, BOX_SIZE_MPC_H, 8)
    with pytest.raises(ValueError, match="non-positive mean"):
        power_spectrum_2d(np.zeros((32, 32)), idx, 8)


# --------------------------------------------------------------------------
# Averaging over maps
# --------------------------------------------------------------------------


def test_mean_spectrum_matches_manual_average():
    npix, n_bins = 64, 12
    rng = np.random.default_rng(3)
    maps = np.abs(1.0 + 0.3 * rng.normal(size=(5, npix, npix))) + 0.1

    res = mean_power_spectrum(maps, np.arange(5), n_bins=n_bins)

    idx, _, _ = radial_bins(npix, BOX_SIZE_MPC_H, n_bins)
    manual = np.mean(
        [np.nan_to_num(power_spectrum_2d(maps[i], idx, n_bins)) for i in range(5)], axis=0
    )
    assert np.allclose(res.power, manual, rtol=1e-9)
    assert res.n_maps == 5


def test_mean_spectrum_respects_the_index_selection():
    rng = np.random.default_rng(4)
    maps = np.abs(1.0 + 0.3 * rng.normal(size=(10, 32, 32))) + 0.1
    res = mean_power_spectrum(maps, np.array([0, 2, 4]), n_bins=8)
    assert res.n_maps == 3
