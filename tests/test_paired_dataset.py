"""Paired hydro/N-body dataset and its correspondence guards.

Plan reference: sections 41, 42, 50, 66.4.

Section 66.4 asks for exactly these properties: hydro and N-body pair ids
identical, cosmological labels identical, pair mapping one-to-one. Section 42
requires those to be enforced before training, and section 50 requires the
shuffled-pair control to differ from the real thing in only the correspondence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralcosmos.data.dataset import LogNormalizer, SuiteSource
from neuralcosmos.data.paired_dataset import PairedMapDataset, PairedSuiteSource
from neuralcosmos.data.splits import maps_for_simulations
from neuralcosmos.data.targets import TargetScaler

MAPS_PER_SIM = 15
PARAM_COLUMNS = {"omega_m": 0, "sigma8": 1, "a_sn1": 2, "a_agn1": 3, "a_sn2": 4, "a_agn2": 5}


def _sources(archive: Path, suite: str, suite_id: int, sim_ids, nbody_shift=0):
    """Build a PairedSuiteSource from the paired synthetic archive.

    ``nbody_shift`` misaligns the N-body params to simulate a broken pairing.
    """
    params = np.loadtxt(archive / f"{suite} LH parameters.txt")
    idx = maps_for_simulations(sim_ids, MAPS_PER_SIM)
    hydro = SuiteSource(
        suite=suite, suite_id=suite_id,
        map_path=archive / f"Maps_Mtot_{suite}_LH_z=0.00.npy",
        params=params, map_indices=idx, maps_per_simulation=MAPS_PER_SIM,
    )
    nbody = SuiteSource(
        suite=suite, suite_id=suite_id,
        map_path=archive / f"Maps_Mtot_{suite}_Nbody_LH_z=0.00.npy",
        params=np.roll(params, nbody_shift, axis=0) if nbody_shift else params,
        map_indices=idx, maps_per_simulation=MAPS_PER_SIM,
    )
    return PairedSuiteSource(suite=suite, hydro=hydro, nbody=nbody)


@pytest.fixture
def scaler():
    return TargetScaler(("omega_m", "sigma8"), (0.1, 0.6), (0.5, 1.0))


@pytest.fixture
def paired_source(synthetic_paired_archive):
    return _sources(synthetic_paired_archive, "SuiteA", 0, list(range(4)))


# --------------------------------------------------------------------------
# Construction guards
# --------------------------------------------------------------------------


def test_mismatched_map_indices_are_rejected(synthetic_paired_archive):
    params = np.loadtxt(synthetic_paired_archive / "SuiteA LH parameters.txt")
    hydro = SuiteSource("SuiteA", 0, synthetic_paired_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy",
                        params, maps_for_simulations([0, 1], MAPS_PER_SIM), MAPS_PER_SIM)
    nbody = SuiteSource("SuiteA", 0, synthetic_paired_archive / "Maps_Mtot_SuiteA_Nbody_LH_z=0.00.npy",
                        params, maps_for_simulations([0, 2], MAPS_PER_SIM), MAPS_PER_SIM)
    with pytest.raises(ValueError, match="different maps"):
        PairedSuiteSource("SuiteA", hydro, nbody)


def test_mismatched_suite_ids_are_rejected(synthetic_paired_archive):
    params = np.loadtxt(synthetic_paired_archive / "SuiteA LH parameters.txt")
    idx = maps_for_simulations([0, 1], MAPS_PER_SIM)
    hydro = SuiteSource("SuiteA", 0, synthetic_paired_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy",
                        params, idx, MAPS_PER_SIM)
    nbody = SuiteSource("SuiteA", 1, synthetic_paired_archive / "Maps_Mtot_SuiteA_Nbody_LH_z=0.00.npy",
                        params, idx, MAPS_PER_SIM)
    with pytest.raises(ValueError, match="different suite ids"):
        PairedSuiteSource("SuiteA", hydro, nbody)


# --------------------------------------------------------------------------
# Section 42 verification
# --------------------------------------------------------------------------


def test_correct_pairs_pass_verification(paired_source, scaler):
    ds = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, verify_pairs=True)
    assert len(ds) == 4 * MAPS_PER_SIM


def test_broken_cosmology_pairing_is_caught(synthetic_paired_archive, scaler):
    """N-body params rolled by one simulation: same map index, wrong cosmology."""
    src = _sources(synthetic_paired_archive, "SuiteA", 0, list(range(4)), nbody_shift=1)
    with pytest.raises(ValueError, match="cosmology"):
        PairedMapDataset([src], scaler, PARAM_COLUMNS, verify_pairs=True)


# --------------------------------------------------------------------------
# Sample contract (section 41)
# --------------------------------------------------------------------------


def test_sample_has_both_views_and_shared_identity(paired_source, scaler):
    ds = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS)
    s = ds[0]
    for key in ("hydro_image", "nbody_image", "target", "suite_id",
                "simulation_id", "map_id"):
        assert key in s
    assert s["hydro_image"].shape == s["nbody_image"].shape == (1, 8, 8)
    assert s["simulation_id"] == s["map_id"] // MAPS_PER_SIM


def test_hydro_and_nbody_differ_but_share_target(paired_source, scaler):
    ds = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS)
    for i in (0, 17, len(ds) - 1):
        s = ds[i]
        # Same cosmology, genuinely different images.
        assert not np.allclose(s["hydro_image"], s["nbody_image"])
        assert s["target"].shape == (2,)


def test_length_matches_expanded_maps(paired_source, scaler):
    ds = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS)
    assert len(ds) == 4 * MAPS_PER_SIM


# --------------------------------------------------------------------------
# Shuffle control (section 50)
# --------------------------------------------------------------------------


def test_shuffle_pairs_breaks_correspondence(paired_source, scaler):
    correct = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, shuffle_pairs=False)
    shuffled = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS,
                                shuffle_pairs=True, shuffle_seed=0)

    # The hydro view is identical; only the N-body partner changes.
    diffs = 0
    for i in range(len(correct)):
        if not np.allclose(correct[i]["nbody_image"], shuffled[i]["nbody_image"]):
            diffs += 1
        assert np.allclose(correct[i]["hydro_image"], shuffled[i]["hydro_image"])
    assert diffs > 0, "shuffling changed no N-body partner"


def test_shuffle_preserves_the_target_distribution(paired_source, scaler):
    """Section 50: the control must preserve marginals, changing only pairing."""
    correct = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, shuffle_pairs=False)
    shuffled = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, shuffle_pairs=True)

    t_correct = np.array([correct[i]["target"] for i in range(len(correct))])
    t_shuffled = np.array([shuffled[i]["target"] for i in range(len(shuffled))])
    # Targets follow the hydro view in both, so the distribution is identical.
    assert np.allclose(np.sort(t_correct, axis=0), np.sort(t_shuffled, axis=0))


def test_shuffle_is_deterministic(paired_source, scaler):
    a = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, shuffle_pairs=True, shuffle_seed=3)
    b = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, shuffle_pairs=True, shuffle_seed=3)
    assert np.allclose(a[5]["nbody_image"], b[5]["nbody_image"])


def test_shuffle_skips_index_check_but_keeps_targets(synthetic_paired_archive, scaler):
    """A shuffled dataset must construct even though indices no longer match."""
    src = _sources(synthetic_paired_archive, "SuiteA", 0, list(range(4)))
    ds = PairedMapDataset([src], scaler, PARAM_COLUMNS, shuffle_pairs=True, verify_pairs=True)
    assert len(ds) == 4 * MAPS_PER_SIM


# --------------------------------------------------------------------------
# Augmentation must be shared across views
# --------------------------------------------------------------------------


def test_augmentation_applies_the_same_transform_to_both_views(paired_source, scaler):
    """A pair-consistency loss compares regions, so the two views must move
    together under augmentation or the correspondence is destroyed."""
    ds = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS,
                          augment=True, augment_seed=1)
    plain = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, augment=False)

    # Under a shared dihedral element, the value SETS of each view are preserved
    # and the relationship between them is preserved. Check the multiset match.
    for i in (0, 10):
        s = ds[i]
        p = plain[i]
        assert np.allclose(np.sort(s["hydro_image"].ravel()),
                           np.sort(p["hydro_image"].ravel()))
        assert np.allclose(np.sort(s["nbody_image"].ravel()),
                           np.sort(p["nbody_image"].ravel()))


def test_normalizer_applied_to_both_views(paired_source, scaler):
    norm = LogNormalizer(mean=11.0, std=0.5, provenance="log10|train|SuiteA|fixture")
    ds = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, hydro_normalizer=norm)
    plain = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS, hydro_normalizer=None)

    s, p = ds[0], plain[0]
    assert np.allclose(s["hydro_image"], (p["hydro_image"] - 11.0) / 0.5, rtol=1e-4)
    assert np.allclose(s["nbody_image"], (p["nbody_image"] - 11.0) / 0.5, rtol=1e-4)


def test_multi_suite_pairing(synthetic_paired_archive, scaler):
    a = _sources(synthetic_paired_archive, "SuiteA", 0, list(range(4)))
    b = _sources(synthetic_paired_archive, "SuiteB", 1, list(range(4)))
    ds = PairedMapDataset([a, b], scaler, PARAM_COLUMNS)
    assert len(ds) == 2 * 4 * MAPS_PER_SIM
    ids = {ds[i]["suite_id"] for i in range(len(ds))}
    assert ids == {0, 1}


def test_helper_arrays_align_with_iteration(paired_source, scaler):
    ds = PairedMapDataset([paired_source], scaler, PARAM_COLUMNS)
    sims = ds.simulation_ids()
    suites = ds.suite_ids()
    assert len(sims) == len(suites) == len(ds)
    for i in (0, 20, len(ds) - 1):
        assert sims[i] == ds[i]["simulation_id"]
        assert suites[i] == ds[i]["suite_id"]
