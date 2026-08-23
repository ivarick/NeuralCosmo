"""The validator must actually reject bad data.

Plan reference: sections 12, 20.1.

A validator that only ever passes is worse than none, because it manufactures
confidence. Every test below corrupts the synthetic archive in one specific way
and asserts the corresponding check fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from neuralcosmos.data.validate import Severity, scan_pixel_stats, validate_all


def _check(report, suite: str, name: str):
    sv = next(s for s in report.suites if s.suite == suite)
    return next(c for c in sv.checks if c.name == name)


def _corrupt_maps(archive: Path, suite: str, fn):
    path = archive / f"Maps_Mtot_{suite}_LH_z=0.00.npy"
    maps = np.load(path)
    maps = fn(maps)
    np.save(path, maps)


# --------------------------------------------------------------------------
# Clean data
# --------------------------------------------------------------------------


def test_clean_archive_passes_everything(synthetic_config: Path, synthetic_archive: Path):
    report = validate_all(synthetic_config, data_root=synthetic_archive)
    assert report.ok
    failed = [c.name for s in report.suites for c in s.checks if not c.passed]
    assert failed == []


def test_clean_archive_reports_strict_positivity(synthetic_config: Path, synthetic_archive: Path):
    report = validate_all(synthetic_config, data_root=synthetic_archive)
    c = _check(report, "SuiteA", "strictly_positive")
    assert c.passed
    assert "log(x) is safe" in c.message


# --------------------------------------------------------------------------
# Corruptions that must be caught
# --------------------------------------------------------------------------


def test_nan_is_detected(synthetic_config: Path, synthetic_archive: Path):
    def inject(maps):
        maps[3, 0, 0] = np.nan
        return maps

    _corrupt_maps(synthetic_archive, "SuiteA", inject)
    report = validate_all(synthetic_config, data_root=synthetic_archive)

    c = _check(report, "SuiteA", "no_nan")
    assert not c.passed and c.is_blocking
    assert not report.ok


def test_inf_is_detected(synthetic_config: Path, synthetic_archive: Path):
    def inject(maps):
        maps[1, 2, 2] = np.inf
        return maps

    _corrupt_maps(synthetic_archive, "SuiteA", inject)
    report = validate_all(synthetic_config, data_root=synthetic_archive)
    assert not _check(report, "SuiteA", "no_inf").passed


def test_nonpositive_values_block_the_log_transform(
    synthetic_config: Path, synthetic_archive: Path
):
    """Section 20.1: a single non-positive value must stop the log transform."""

    def inject(maps):
        maps[0, 0, 0] = 0.0
        maps[0, 0, 1] = -5.0
        return maps

    _corrupt_maps(synthetic_archive, "SuiteA", inject)
    report = validate_all(synthetic_config, data_root=synthetic_archive)

    c = _check(report, "SuiteA", "strictly_positive")
    assert not c.passed
    assert "STOP" in c.message
    # It is a WARN, not an ERROR: the data may be legitimate, but the
    # preprocessing decision must be made by a human, not silently.
    assert c.severity == Severity.WARN


def test_wrong_shape_is_blocking(synthetic_config: Path, synthetic_archive: Path):
    def truncate(maps):
        return maps[:45]

    _corrupt_maps(synthetic_archive, "SuiteA", truncate)
    report = validate_all(synthetic_config, data_root=synthetic_archive)

    assert not _check(report, "SuiteA", "maps_shape").passed
    assert not _check(report, "SuiteA", "map_simulation_mapping").passed
    assert not report.ok


def test_broken_map_simulation_contract_is_blocking(
    synthetic_config: Path, synthetic_archive: Path, tmp_path: Path
):
    """15000 maps with 999 simulations must fail, even though both look plausible."""
    cfg = yaml.safe_load(synthetic_config.read_text(encoding="utf-8"))
    cfg["expected"]["params_shape"] = [5, 6]
    cfg["expected"]["n_simulations"] = 5
    patched = tmp_path / "patched.yaml"
    patched.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Drop one simulation from the parameter file but leave 90 maps in place.
    ppath = synthetic_archive / "SuiteA LH parameters.txt"
    params = np.loadtxt(ppath)[:5]
    np.savetxt(ppath, params, fmt="%.5f")

    report = validate_all(patched, data_root=synthetic_archive, suites=["SuiteA"])
    c = _check(report, "SuiteA", "map_simulation_mapping")
    assert not c.passed
    assert "90 maps != 5 x 15" in c.message


def test_out_of_range_target_is_blocking(synthetic_config: Path, synthetic_archive: Path):
    ppath = synthetic_archive / "SuiteA LH parameters.txt"
    params = np.loadtxt(ppath)
    params[0, 0] = 0.85  # omega_m far outside [0.1, 0.5]
    np.savetxt(ppath, params, fmt="%.5f")

    report = validate_all(synthetic_config, data_root=synthetic_archive)
    c = _check(report, "SuiteA", "range_omega_m")
    assert not c.passed and c.is_blocking


def test_small_range_deviation_is_tolerated(synthetic_config: Path, synthetic_archive: Path):
    """Section 12: do not hard-fail on tiny floating-point deviations."""
    ppath = synthetic_archive / "SuiteA LH parameters.txt"
    params = np.loadtxt(ppath)
    params[0, 0] = 0.0995  # 0.0005 below the nominal bound, tolerance is 0.01
    np.savetxt(ppath, params, fmt="%.5f")

    report = validate_all(synthetic_config, data_root=synthetic_archive)
    assert _check(report, "SuiteA", "range_omega_m").passed


def test_missing_file_is_blocking_and_stops_further_checks(
    synthetic_config: Path, synthetic_archive: Path
):
    (synthetic_archive / "SuiteB LH parameters.txt").unlink()
    report = validate_all(synthetic_config, data_root=synthetic_archive)

    sv = next(s for s in report.suites if s.suite == "SuiteB")
    assert not sv.ok
    # Once a file is missing there is nothing to scan; do not emit noise.
    assert {c.name for c in sv.checks} == {"map_file_exists", "param_file_exists"}


def test_one_bad_suite_fails_the_whole_report(synthetic_config: Path, synthetic_archive: Path):
    def inject(maps):
        maps[0, 0, 0] = np.nan
        return maps

    _corrupt_maps(synthetic_archive, "SuiteB", inject)
    report = validate_all(synthetic_config, data_root=synthetic_archive)

    assert next(s for s in report.suites if s.suite == "SuiteA").ok
    assert not next(s for s in report.suites if s.suite == "SuiteB").ok
    assert not report.ok


# --------------------------------------------------------------------------
# Quick mode
# --------------------------------------------------------------------------


def test_quick_mode_skips_the_scan_and_says_so(synthetic_config: Path, synthetic_archive: Path):
    def inject(maps):
        maps[0, 0, 0] = np.nan
        return maps

    _corrupt_maps(synthetic_archive, "SuiteA", inject)
    report = validate_all(synthetic_config, data_root=synthetic_archive, full_scan=False)

    # Quick mode cannot see the NaN, so it must not claim the data is verified.
    c = _check(report, "SuiteA", "pixel_scan")
    assert "SKIPPED" in c.message
    assert "Positivity is unverified" in c.message
    assert next(s for s in report.suites if s.suite == "SuiteA").stats is None


# --------------------------------------------------------------------------
# Streaming statistics correctness
# --------------------------------------------------------------------------


def test_streaming_stats_match_numpy_exactly(tmp_path: Path):
    """Chunked accumulation must agree with a whole-array computation."""
    rng = np.random.default_rng(7)
    arr = rng.lognormal(mean=25.0, sigma=1.5, size=(97, 6, 6)).astype(np.float32)
    path = tmp_path / "maps.npy"
    np.save(path, arr)

    # A chunk size that does not divide 97 evenly, to exercise the last partial chunk.
    st, shape, dtype = scan_pixel_stats(path, chunk_maps=10)

    ref = arr.astype(np.float64)
    assert shape == (97, 6, 6)
    assert dtype == "float32"
    assert st.count == ref.size
    assert st.minimum == pytest.approx(float(ref.min()))
    assert st.maximum == pytest.approx(float(ref.max()))
    assert st.mean == pytest.approx(float(ref.mean()), rel=1e-9)
    assert st.std == pytest.approx(float(ref.std()), rel=1e-6)
    assert st.n_nan == 0 and st.n_inf == 0 and st.n_nonpositive == 0


def test_streaming_stats_count_nonfinite_without_poisoning_moments(tmp_path: Path):
    arr = np.ones((10, 4, 4), dtype=np.float32) * 3.0
    arr[0, 0, 0] = np.nan
    arr[0, 0, 1] = np.inf
    arr[1, 0, 0] = -1.0
    path = tmp_path / "maps.npy"
    np.save(path, arr)

    st, _, _ = scan_pixel_stats(path, chunk_maps=3)

    assert st.n_nan == 1
    assert st.n_inf == 1
    assert st.n_nonpositive == 1
    # NaN and Inf are excluded from the moments, so the mean stays finite.
    assert np.isfinite(st.mean)
    assert st.count == arr.size - 2
