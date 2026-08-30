"""Shared fixtures.

Plan reference: section 79 - "Raw data should not be required to run unit tests."

Every fixture here builds a synthetic mini-archive with the same *structure* as
CAMELS CMD but a tiny fraction of the size: small maps, few simulations. The
structural invariants under test (the maps-per-simulation contract, split
disjointness, parameter ranges) are identical to the real archive, so the tests
are meaningful without needing 11.8 GB on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Deliberately small, but the maps-per-simulation factor matches the real data
# because that constant is what the leakage tests are about.
N_SIMS = 6
MAPS_PER_SIM = 15
MAP_PIXELS = 8
SUITES = ("SuiteA", "SuiteB")


def _make_params(n_sims: int, seed: int) -> np.ndarray:
    """Latin-hypercube-like parameters spanning the documented CAMELS ranges."""
    rng = np.random.default_rng(seed)
    params = np.zeros((n_sims, 6), dtype=np.float64)
    # Endpoints are pinned so range checks see the true design bounds.
    params[:, 0] = np.linspace(0.1002, 0.4998, n_sims)          # omega_m
    params[:, 1] = np.linspace(0.6002, 0.9998, n_sims)          # sigma8
    params[:, 2:] = rng.uniform(0.25, 4.0, size=(n_sims, 4))    # feedback nuisances
    return params


def _make_maps(params: np.ndarray, maps_per_sim: int, pixels: int, seed: int) -> np.ndarray:
    """Strictly positive maps whose amplitude depends on the parameters.

    The dependence is what lets a test assert that a model or a summary
    statistic can in principle recover the targets. Values are strictly
    positive so the log-transform path is exercisable.
    """
    rng = np.random.default_rng(seed)
    n_sims = params.shape[0]
    maps = np.empty((n_sims * maps_per_sim, pixels, pixels), dtype=np.float32)
    for s in range(n_sims):
        omega_m, sigma8 = params[s, 0], params[s, 1]
        scale = 1e10 * (1.0 + 5.0 * omega_m) * (0.5 + sigma8)
        for m in range(maps_per_sim):
            noise = rng.lognormal(mean=0.0, sigma=0.3, size=(pixels, pixels))
            maps[s * maps_per_sim + m] = (scale * noise).astype(np.float32)
    return maps


@pytest.fixture
def synthetic_archive(tmp_path: Path) -> Path:
    """A directory laid out like the real archive, with tiny files."""
    root = tmp_path / "archive"
    root.mkdir()
    for i, suite in enumerate(SUITES):
        params = _make_params(N_SIMS, seed=100 + i)
        maps = _make_maps(params, MAPS_PER_SIM, MAP_PIXELS, seed=200 + i)
        np.save(root / f"Maps_Mtot_{suite}_LH_z=0.00.npy", maps)
        np.savetxt(root / f"{suite} LH parameters.txt", params, fmt="%.5f")
    return root


@pytest.fixture
def synthetic_paired_archive(tmp_path: Path) -> Path:
    """An archive with matched hydro and N-body maps per suite.

    The N-body map of a simulation shares its cosmology and its coarse
    structure with the hydro map but is a smoothed variant, standing in for a
    gravity-only run that lacks small-scale baryonic detail. Crucially, map i of
    the N-body file corresponds to map i of the hydro file, which is the
    correspondence the paired dataset asserts.
    """
    root = tmp_path / "paired"
    root.mkdir()
    for i, suite in enumerate(SUITES):
        params = _make_params(N_SIMS, seed=100 + i)
        hydro = _make_maps(params, MAPS_PER_SIM, MAP_PIXELS, seed=200 + i)

        # N-body counterpart: same per-simulation amplitude, a smoothed field,
        # index-aligned with hydro.
        rng = np.random.default_rng(300 + i)
        nbody = np.empty_like(hydro)
        for m in range(hydro.shape[0]):
            base = hydro[m].astype(np.float64)
            smoothed = 0.5 * base + 0.5 * base.mean()
            nbody[m] = (smoothed * rng.lognormal(0.0, 0.05, size=base.shape)).astype(np.float32)

        np.save(root / f"Maps_Mtot_{suite}_LH_z=0.00.npy", hydro)
        np.save(root / f"Maps_Mtot_{suite}_Nbody_LH_z=0.00.npy", nbody)
        np.savetxt(root / f"{suite} LH parameters.txt", params, fmt="%.5f")
    return root


@pytest.fixture
def synthetic_config(tmp_path: Path, synthetic_archive: Path) -> Path:
    """A data config matching the synthetic archive."""
    n_maps = N_SIMS * MAPS_PER_SIM
    map_bytes = n_maps * MAP_PIXELS * MAP_PIXELS * 4 + 128
    cfg = {
        "dataset": "SYNTHETIC",
        "set": "LH",
        "field": "Mtot",
        "redshift": 0.0,
        "maps_per_simulation": MAPS_PER_SIM,
        "expected": {
            "maps_shape": [n_maps, MAP_PIXELS, MAP_PIXELS],
            "maps_dtype": "float32",
            "params_shape": [N_SIMS, 6],
            "n_simulations": N_SIMS,
            "map_file_bytes": map_bytes,
        },
        "param_columns": {
            "omega_m": 0,
            "sigma8": 1,
            "a_sn1": 2,
            "a_agn1": 3,
            "a_sn2": 4,
            "a_agn2": 5,
        },
        "targets": {
            "names": ["omega_m", "sigma8"],
            "scaling": "fixed_lh_range",
            "ranges": {"omega_m": [0.1, 0.5], "sigma8": [0.6, 1.0]},
            "range_tolerance": 0.01,
        },
        "suites": {
            suite: {
                "map_file": f"Maps_Mtot_{suite}_LH_z=0.00.npy",
                "param_file": f"{suite} LH parameters.txt",
            }
            for suite in SUITES
        },
        "roles": {
            "development_suites": ["SuiteA"],
            "sealed_target_suites": ["SuiteB"],
        },
    }
    path = tmp_path / "synthetic_mtot.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
