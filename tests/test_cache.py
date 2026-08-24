"""uint16 log-quantised cache.

Plan reference: section 83.

The cache is a lossy representation of scientific data, so the tests here are
mostly about bounding and characterising that loss, and about refusing to load
a cache whose encoding cannot be established.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralcosmos.data.builders import build_dataset, build_sources
from neuralcosmos.data.cache import QuantSpec, cache_paths, open_cache, write_cache
from neuralcosmos.data.dataset import LogNormalizer
from neuralcosmos.data.manifest import load_data_config
from neuralcosmos.data.splits import build_split_file

MAPS_PER_SIM = 15


@pytest.fixture
def cfg(synthetic_config: Path):
    return load_data_config(synthetic_config)


@pytest.fixture
def split_file():
    return build_split_file(
        suites={"SuiteA": 6, "SuiteB": 6}, master_seed=1,
        n_val=1, n_test=1, maps_per_simulation=MAPS_PER_SIM,
    )


@pytest.fixture
def norm():
    return LogNormalizer(mean=11.0, std=0.5, provenance="log10|train|SuiteA|fixture")


# --------------------------------------------------------------------------
# Quantisation arithmetic
# --------------------------------------------------------------------------


def test_round_trip_error_is_below_half_a_step():
    spec = QuantSpec(9.0, 16.0)
    rng = np.random.default_rng(0)
    vals = rng.uniform(9.0, 16.0, size=200_000)

    back = spec.decode(spec.encode(vals), dtype=np.float64)
    err = np.abs(back - vals)
    assert err.max() <= spec.step / 2 + 1e-9


def test_quantisation_noise_is_negligible_against_the_signal():
    """The whole justification for the cache rests on this number.

    Measured pixel standard deviation in log space is 0.4836 dex. Quantisation
    noise must be a tiny fraction of that, or the cache would be adding noise
    comparable to the physical variation it is meant to preserve.
    """
    spec = QuantSpec(9.0, 16.0)
    measured_sigma = 0.483565

    # Uniform quantisation noise has standard deviation step/sqrt(12).
    noise_sigma = spec.step / np.sqrt(12)
    assert noise_sigma / measured_sigma < 1e-4


def test_step_is_much_finer_than_float16_would_be():
    """Same 2 bytes; uint16 is the better spend for this dynamic range."""
    spec = QuantSpec(9.0, 16.0)
    # float16 spacing at magnitude ~15.6 is 2**-10 * 2**3.
    float16_spacing = 2.0**-10 * 2.0**3
    assert spec.step < float16_spacing / 50


def test_encoding_is_monotonic():
    spec = QuantSpec(9.0, 16.0)
    vals = np.linspace(9.0, 16.0, 5000)
    codes = spec.encode(vals)
    assert np.all(np.diff(codes.astype(np.int64)) >= 0)


def test_window_endpoints_map_to_extremes():
    spec = QuantSpec(9.0, 16.0)
    assert spec.encode(np.array([9.0]))[0] == 0
    assert spec.encode(np.array([16.0]))[0] == 65535


def test_values_outside_the_window_raise_rather_than_clip():
    """Clipping would silently delete the densest cells.

    Extreme densities are exactly where small-scale baryonic information lives,
    so quietly saturating them would remove the signal the project studies.
    """
    spec = QuantSpec(9.0, 16.0)
    with pytest.raises(ValueError, match="outside the quantisation window"):
        spec.encode(np.array([8.5, 10.0]))
    with pytest.raises(ValueError, match="outside the quantisation window"):
        spec.encode(np.array([10.0, 16.5]))


def test_real_archive_range_fits_the_default_window():
    """Observed across all 2.95e9 pixels: 4.665e9 to 4.204e15."""
    spec = QuantSpec()
    assert spec.lo < np.log10(4.66504e9)
    assert spec.hi > np.log10(4.20381e15)


def test_invalid_window_rejected():
    with pytest.raises(ValueError, match="invalid quantisation window"):
        QuantSpec(lo=16.0, hi=9.0)


# --------------------------------------------------------------------------
# Writing and reading a cache
# --------------------------------------------------------------------------


def test_write_cache_halves_the_size(synthetic_archive: Path, tmp_path: Path):
    src = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    cached = write_cache("SuiteA", src, tmp_path / "cache")

    out_bytes = cached.path.stat().st_size
    # uint16 vs float32, modulo the small .npy header.
    assert 0.45 < out_bytes / cached.source_bytes < 0.55


def test_cached_values_match_log10_of_source(synthetic_archive: Path, tmp_path: Path):
    src = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    cached = write_cache("SuiteA", src, tmp_path / "cache")

    original = np.load(src)
    codes = np.load(cached.path)
    decoded = cached.spec.decode(codes, dtype=np.float64)

    expected = np.log10(original.astype(np.float64))
    assert np.abs(decoded - expected).max() <= cached.spec.step / 2 + 1e-9


def test_open_cache_recovers_the_spec(synthetic_archive: Path, tmp_path: Path):
    src = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    spec = QuantSpec(9.5, 15.5)
    write_cache("SuiteA", src, tmp_path / "cache", spec=spec)

    path, recovered = open_cache(tmp_path / "cache", "SuiteA")
    assert path.exists()
    assert recovered == spec


def test_open_cache_refuses_without_metadata(synthetic_archive: Path, tmp_path: Path):
    """Guessing a quantisation window would silently corrupt every value."""
    src = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    write_cache("SuiteA", src, tmp_path / "cache")
    _, meta = cache_paths(tmp_path / "cache", "SuiteA")
    meta.unlink()

    with pytest.raises(FileNotFoundError, match="no metadata sidecar"):
        open_cache(tmp_path / "cache", "SuiteA")


def test_open_cache_rejects_unknown_format(synthetic_archive: Path, tmp_path: Path):
    src = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    write_cache("SuiteA", src, tmp_path / "cache")
    _, meta = cache_paths(tmp_path / "cache", "SuiteA")
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["magic"] = "something-else"
    meta.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(ValueError, match="Unrecognised cache format"):
        open_cache(tmp_path / "cache", "SuiteA")


def test_missing_cache_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No cache"):
        open_cache(tmp_path, "Nonexistent")


def test_write_refuses_to_clobber(synthetic_archive: Path, tmp_path: Path):
    src = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    write_cache("SuiteA", src, tmp_path / "cache")
    with pytest.raises(FileExistsError):
        write_cache("SuiteA", src, tmp_path / "cache")


def test_write_rejects_nonpositive_source(tmp_path: Path):
    bad = tmp_path / "bad.npy"
    arr = np.ones((4, 3, 3), dtype=np.float32) * 1e10
    arr[1, 0, 0] = 0.0
    np.save(bad, arr)

    with pytest.raises(ValueError, match="non-positive pixel"):
        write_cache("Bad", bad, tmp_path / "cache")


# --------------------------------------------------------------------------
# Integration with the dataset
# --------------------------------------------------------------------------


def test_dataset_reads_cache_transparently(cfg, synthetic_archive, split_file, norm, tmp_path):
    cache_root = tmp_path / "cache"
    for suite in ("SuiteA",):
        write_cache(suite, synthetic_archive / f"Maps_Mtot_{suite}_LH_z=0.00.npy", cache_root)

    raw_ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    cached_ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm,
                              cache_root=cache_root)

    assert len(raw_ds) == len(cached_ds)
    for i in (0, 7, len(raw_ds) - 1):
        a, b = raw_ds[i], cached_ds[i]
        assert a["map_id"] == b["map_id"]
        assert np.array_equal(a["target"], b["target"])
        # Same normalizer, so the tolerance is the quantisation step scaled by
        # 1/std: 1.07e-4 / 0.5 in this fixture.
        assert np.abs(a["image"] - b["image"]).max() < 1e-3


def test_partial_cache_falls_back_to_raw(cfg, synthetic_archive, split_file, norm, tmp_path):
    """Caching only source suites must be a supported configuration."""
    cache_root = tmp_path / "cache"
    write_cache("SuiteA", synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy", cache_root)

    sources = build_sources(cfg, synthetic_archive, split_file, ["SuiteA", "SuiteB"], "train",
                            cache_root=cache_root)
    by_name = {s.suite: s for s in sources}
    assert by_name["SuiteA"].is_cached
    assert not by_name["SuiteB"].is_cached


def test_cached_source_forbids_linear_mode(cfg, synthetic_archive, split_file, norm, tmp_path):
    """Mixing log and linear inputs across suites would be invisible and fatal."""
    cache_root = tmp_path / "cache"
    write_cache("SuiteA", synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy", cache_root)

    with pytest.raises(ValueError, match="log_transform=False"):
        build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm,
                      cache_root=cache_root, log_transform=False)


def test_cache_stores_unnormalised_values(synthetic_archive: Path, tmp_path: Path):
    """A cache with normalization baked in would break section 53's rotations."""
    src = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    cached = write_cache("SuiteA", src, tmp_path / "cache")
    _, meta = cache_paths(tmp_path / "cache", "SuiteA")
    doc = json.loads(meta.read_text(encoding="utf-8"))

    assert doc["normalised"] is False
    assert doc["transform"] == "log10"
    decoded = cached.spec.decode(np.load(cached.path)[:2], dtype=np.float64)
    # Raw log10 of ~1e10-scale densities, not a standardised quantity.
    assert decoded.mean() > 5.0
