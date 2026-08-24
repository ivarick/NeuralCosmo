"""Dataset, target scaling and augmentation.

Plan reference: sections 14, 15, 22, 23.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralcosmos.data.builders import build_dataset, build_sources, suite_id_map
from neuralcosmos.data.dataset import LogNormalizer, dihedral_transform
from neuralcosmos.data.manifest import load_data_config
from neuralcosmos.data.splits import build_split_file
from neuralcosmos.data.targets import TargetScaler

MAPS_PER_SIM = 15


@pytest.fixture
def cfg(synthetic_config: Path):
    return load_data_config(synthetic_config)


@pytest.fixture
def split_file():
    # 6 simulations per suite: 4 train, 1 val, 1 test.
    return build_split_file(
        suites={"SuiteA": 6, "SuiteB": 6},
        master_seed=1,
        n_val=1,
        n_test=1,
        maps_per_simulation=MAPS_PER_SIM,
    )


@pytest.fixture
def norm():
    return LogNormalizer(mean=11.0, std=0.5, provenance="log10|train|SuiteA|test-fixture")


# --------------------------------------------------------------------------
# Target scaling (section 22)
# --------------------------------------------------------------------------


def test_target_scaler_maps_design_range_to_unit_interval(cfg):
    ts = TargetScaler.from_config(cfg)
    assert ts.forward([0.1, 0.6]).tolist() == pytest.approx([0.0, 0.0])
    assert ts.forward([0.5, 1.0]).tolist() == pytest.approx([1.0, 1.0])
    assert ts.forward([0.3, 0.8]).tolist() == pytest.approx([0.5, 0.5])


def test_target_scaler_round_trips(cfg):
    ts = TargetScaler.from_config(cfg)
    physical = np.array([[0.1234, 0.8765], [0.4999, 0.6001]])
    assert ts.inverse(ts.forward(physical)) == pytest.approx(physical)


def test_target_scaler_rejects_unimplemented_scaling(cfg):
    cfg = dict(cfg)
    cfg["targets"] = dict(cfg["targets"], scaling="zscore_from_data")
    with pytest.raises(NotImplementedError, match="fixed_lh_range"):
        TargetScaler.from_config(cfg)


def test_target_scaler_rejects_degenerate_range():
    with pytest.raises(ValueError, match="non-positive range"):
        TargetScaler(names=("x",), lower=(1.0,), upper=(1.0,))


# --------------------------------------------------------------------------
# Normalizer (section 20.2)
# --------------------------------------------------------------------------


def test_normalizer_requires_provenance():
    with pytest.raises(ValueError, match="provenance"):
        LogNormalizer(mean=1.0, std=1.0, provenance="")


def test_normalizer_rejects_nonpositive_std():
    with pytest.raises(ValueError, match="std must be positive"):
        LogNormalizer(mean=1.0, std=0.0, provenance="x")


def test_normalizer_rejects_nonfinite():
    with pytest.raises(ValueError, match="Non-finite"):
        LogNormalizer(mean=float("nan"), std=1.0, provenance="x")


# --------------------------------------------------------------------------
# Augmentation (section 23)
# --------------------------------------------------------------------------


def test_dihedral_group_has_eight_distinct_elements():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(7, 7))
    seen = {dihedral_transform(img, k).tobytes() for k in range(8)}
    assert len(seen) == 8


def test_dihedral_preserves_shape_and_values():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(9, 9))
    for k in range(8):
        out = dihedral_transform(img, k)
        assert out.shape == img.shape
        assert np.allclose(np.sort(out.ravel()), np.sort(img.ravel()))


def test_dihedral_output_is_contiguous():
    """rot90/fliplr return negative-stride views that torch cannot consume."""
    img = np.arange(16, dtype=np.float32).reshape(4, 4)
    for k in range(8):
        assert dihedral_transform(img, k).flags["C_CONTIGUOUS"]


def test_dihedral_identity_at_zero():
    img = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert np.array_equal(dihedral_transform(img, 0), img)


def test_dihedral_rejects_out_of_range():
    with pytest.raises(ValueError):
        dihedral_transform(np.zeros((3, 3)), 8)


# --------------------------------------------------------------------------
# Dataset behaviour (sections 14, 15)
# --------------------------------------------------------------------------


def test_dataset_length_matches_expanded_maps(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    assert len(ds) == 4 * MAPS_PER_SIM


def test_sample_has_every_required_key(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    s = ds[0]
    # Section 15 requires all of these; simulation_id in particular must never
    # be dropped, since bootstrap and grouped evaluation depend on it.
    for key in ("image", "target", "suite_id", "simulation_id", "map_id"):
        assert key in s


def test_image_shape_and_dtype(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    img = ds[0]["image"]
    assert img.shape == (1, 8, 8)
    assert img.dtype == np.float32


def test_simulation_id_is_consistent_with_map_id(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    for i in range(len(ds)):
        s = ds[i]
        assert s["simulation_id"] == s["map_id"] // MAPS_PER_SIM


def test_targets_are_constant_within_a_simulation(cfg, synthetic_archive, split_file, norm):
    """All 15 maps of a simulation must carry the identical parameter vector."""
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    by_sim: dict[int, list] = {}
    for i in range(len(ds)):
        s = ds[i]
        by_sim.setdefault(s["simulation_id"], []).append(s["target"])
    assert by_sim
    for sim, targets in by_sim.items():
        assert len(targets) == MAPS_PER_SIM
        for t in targets[1:]:
            assert np.array_equal(t, targets[0]), f"simulation {sim} has inconsistent targets"


def test_targets_lie_in_unit_interval(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    for i in range(0, len(ds), 7):
        t = ds[i]["target"]
        assert np.all(t >= -0.01) and np.all(t <= 1.01)


def test_dataset_only_returns_maps_from_its_split(cfg, synthetic_archive, split_file, norm):
    train = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    test = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "test", norm)

    train_maps = {train[i]["map_id"] for i in range(len(train))}
    test_maps = {test[i]["map_id"] for i in range(len(test))}
    assert train_maps & test_maps == set()


def test_multi_suite_dataset_concatenates(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA", "SuiteB"], "train", norm)
    assert len(ds) == 2 * 4 * MAPS_PER_SIM
    ids = {ds[i]["suite_id"] for i in range(len(ds))}
    assert len(ids) == 2


def test_suite_ids_are_stable_under_config_reordering(cfg):
    reordered = dict(cfg)
    reordered["suites"] = {k: cfg["suites"][k] for k in reversed(list(cfg["suites"]))}
    assert suite_id_map(cfg) == suite_id_map(reordered)


def test_log_transform_is_applied(cfg, synthetic_archive, split_file):
    raw = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train",
                        None, log_transform=False)
    logged = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train",
                           None, log_transform=True)
    assert np.allclose(logged[0]["image"], np.log10(raw[0]["image"]), rtol=1e-5)


def test_normalization_is_applied(cfg, synthetic_archive, split_file, norm):
    plain = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", None)
    scaled = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    expected = (plain[0]["image"] - norm.mean) / norm.std
    assert np.allclose(scaled[0]["image"], expected, rtol=1e-5)


def test_negative_indexing_and_bounds(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    assert ds[-1]["map_id"] == ds[len(ds) - 1]["map_id"]
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_helper_arrays_align_with_iteration_order(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA", "SuiteB"], "train", norm)
    sims = ds.simulation_ids()
    suites = ds.suite_ids()
    phys = ds.targets_physical()

    assert len(sims) == len(suites) == len(phys) == len(ds)
    for i in (0, 5, len(ds) // 2, len(ds) - 1):
        s = ds[i]
        assert sims[i] == s["simulation_id"]
        assert suites[i] == s["suite_id"]
        assert np.allclose(phys[i], s["target_physical"], rtol=1e-5)


def test_augmentation_changes_output_but_preserves_values(cfg, synthetic_archive, split_file, norm):
    plain = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    aug = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm,
                        augment=True, augment_seed=3)

    base = plain[0]["image"]
    variants = {aug[0]["image"].tobytes() for _ in range(30)}
    assert len(variants) > 1, "augmentation produced no variation"
    for _ in range(5):
        a = aug[0]["image"]
        assert np.allclose(np.sort(a.ravel()), np.sort(base.ravel()))


def test_augmentation_never_alters_the_target(cfg, synthetic_archive, split_file, norm):
    """A flip or rotation must not change the cosmology being predicted."""
    plain = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm)
    aug = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm,
                        augment=True, augment_seed=3)
    for i in (0, 10, 20):
        assert np.array_equal(aug[i]["target"], plain[i]["target"])


def test_max_simulations_truncates_train_only(cfg, synthetic_archive, split_file, norm):
    ds = build_dataset(cfg, synthetic_archive, split_file, ["SuiteA"], "train", norm,
                       max_simulations=2)
    assert len(ds) == 2 * MAPS_PER_SIM

    with pytest.raises(ValueError, match="may only truncate the training split"):
        build_sources(cfg, synthetic_archive, split_file, ["SuiteA"], "test", max_simulations=1)


def test_dataset_rejects_empty_source_list():
    from neuralcosmos.data.dataset import CAMELSMapDataset

    with pytest.raises(ValueError, match="at least one SuiteSource"):
        CAMELSMapDataset(
            sources=[],
            target_scaler=TargetScaler(("a",), (0.0,), (1.0,)),
            param_columns={"a": 0},
        )


def test_suite_source_drops_mmap_handle_when_pickled(cfg, synthetic_archive, split_file):
    """Windows DataLoader workers are spawned, so sources must pickle cleanly."""
    import pickle

    src = build_sources(cfg, synthetic_archive, split_file, ["SuiteA"], "train")[0]
    _ = src.maps()               # open the handle
    assert src._handle is not None

    restored = pickle.loads(pickle.dumps(src))
    assert restored._handle is None
    # And it must reopen transparently on first use in the new process.
    assert restored.maps().shape == (90, 8, 8)
