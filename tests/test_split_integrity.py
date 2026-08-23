"""Split integrity: the leakage guard.

Plan reference: sections 16, 17, 66.2, 66.3.

Section 17: "A failed split-integrity test must stop training."

The tests below check disjointness at BOTH levels. Simulation-level
disjointness is what the code asserts; map-level disjointness is what actually
matters scientifically, because the model sees maps, not simulations. They are
equivalent only if the map/simulation expansion is correct, so both are tested
rather than assuming the implication.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralcosmos.data.splits import (
    SPLIT_NAMES,
    build_split_file,
    load_split_file,
    make_suite_split,
    maps_for_simulations,
    suite_rng_seed,
)

MAPS_PER_SIM = 15


@pytest.fixture
def split_file():
    return build_split_file(
        suites={"IllustrisTNG": 1000, "SIMBA": 1000, "Astrid": 1000},
        master_seed=42,
        n_val=50,
        n_test=50,
        maps_per_simulation=MAPS_PER_SIM,
    )


# --------------------------------------------------------------------------
# Section 17 invariants
# --------------------------------------------------------------------------


def test_split_sizes_match_the_cmd_benchmark(split_file):
    for name in split_file.splits:
        sp = split_file.suite(name)
        assert len(sp.train) == 900
        assert len(sp.val) == 50
        assert len(sp.test) == 50


def test_simulation_sets_are_pairwise_disjoint(split_file):
    for name in split_file.splits:
        sp = split_file.suite(name)
        train, val, test = set(sp.train), set(sp.val), set(sp.test)
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()


def test_splits_partition_all_simulations(split_file):
    for name in split_file.splits:
        sp = split_file.suite(name)
        assert set(sp.train) | set(sp.val) | set(sp.test) == set(range(1000))


def test_no_simulation_id_is_duplicated(split_file):
    for name in split_file.splits:
        sp = split_file.suite(name)
        for s in SPLIT_NAMES:
            ids = sp.ids(s)
            assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# The property that actually matters: map-level disjointness
# --------------------------------------------------------------------------


def test_map_indices_are_disjoint_across_splits(split_file):
    for name in split_file.splits:
        sp = split_file.suite(name)
        maps = {s: set(maps_for_simulations(sp.ids(s), MAPS_PER_SIM).tolist()) for s in SPLIT_NAMES}
        assert maps["train"] & maps["val"] == set()
        assert maps["train"] & maps["test"] == set()
        assert maps["val"] & maps["test"] == set()


def test_map_indices_cover_every_map_exactly_once(split_file):
    for name in split_file.splits:
        sp = split_file.suite(name)
        all_maps: list[int] = []
        for s in SPLIT_NAMES:
            all_maps.extend(maps_for_simulations(sp.ids(s), MAPS_PER_SIM).tolist())
        assert len(all_maps) == 15000
        assert len(set(all_maps)) == 15000
        assert set(all_maps) == set(range(15000))


def test_all_fifteen_maps_of_a_simulation_share_one_split(split_file):
    """The core anti-leakage property of section 16."""
    for name in split_file.splits:
        sp = split_file.suite(name)
        owner: dict[int, str] = {}
        for s in SPLIT_NAMES:
            for m in maps_for_simulations(sp.ids(s), MAPS_PER_SIM):
                owner[int(m)] = s
        for sim in range(1000):
            labels = {owner[sim * MAPS_PER_SIM + k] for k in range(MAPS_PER_SIM)}
            assert len(labels) == 1, f"simulation {sim} is split across {labels}"


def test_maps_for_simulations_expands_correctly():
    got = maps_for_simulations([0, 2], MAPS_PER_SIM)
    expected = list(range(0, 15)) + list(range(30, 45))
    assert got.tolist() == expected


def test_maps_for_simulations_handles_empty():
    assert maps_for_simulations([], MAPS_PER_SIM).size == 0


def test_maps_for_simulations_deduplicates():
    assert maps_for_simulations([3, 3, 3], MAPS_PER_SIM).tolist() == list(range(45, 60))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_reproduces_the_same_split():
    a = make_suite_split("SIMBA", 1000, 50, 50, master_seed=42)
    b = make_suite_split("SIMBA", 1000, 50, 50, master_seed=42)
    assert a == b


def test_different_seed_gives_a_different_split():
    a = make_suite_split("SIMBA", 1000, 50, 50, master_seed=42)
    b = make_suite_split("SIMBA", 1000, 50, 50, master_seed=43)
    assert a.test != b.test


def test_suites_get_different_splits_under_one_master_seed():
    """Otherwise every suite would hold out the same simulation indices."""
    a = make_suite_split("IllustrisTNG", 1000, 50, 50, master_seed=42)
    b = make_suite_split("SIMBA", 1000, 50, 50, master_seed=42)
    assert a.test != b.test
    assert suite_rng_seed(42, "IllustrisTNG") != suite_rng_seed(42, "SIMBA")


def test_adding_a_suite_does_not_disturb_existing_splits():
    """A late-arriving suite must not silently reshuffle earlier partitions.

    This is why the seed is derived per suite rather than drawn from one shared
    RNG stream: with a shared stream, inserting a suite would shift every
    subsequent draw and invalidate results computed before it was added.
    """
    two = build_split_file({"IllustrisTNG": 1000, "SIMBA": 1000}, 42, 50, 50, MAPS_PER_SIM)
    three = build_split_file(
        {"IllustrisTNG": 1000, "SIMBA": 1000, "Astrid": 1000}, 42, 50, 50, MAPS_PER_SIM
    )
    for name in ("IllustrisTNG", "SIMBA"):
        assert two.suite(name) == three.suite(name)


def test_val_and_test_are_stable_when_the_training_set_shrinks():
    """Supports the data-efficiency ablation of section 15.

    Training on fewer simulations must not change what is held out, or the
    ablation would be comparing different test sets.
    """
    full = make_suite_split("SIMBA", 1000, 50, 50, master_seed=42)
    # A subset of training simulations, evaluated against the same val/test.
    subset = full.train[:250]
    assert set(subset).isdisjoint(set(full.val))
    assert set(subset).isdisjoint(set(full.test))


# --------------------------------------------------------------------------
# Corruption must be rejected
# --------------------------------------------------------------------------


def test_overlapping_split_is_rejected():
    from neuralcosmos.data.splits import SuiteSplit

    bad = SuiteSplit(
        suite="Bad",
        train=(0, 1, 2),
        val=(2, 3),          # 2 appears twice
        test=(4,),
        n_simulations=5,
        seed=0,
    )
    with pytest.raises(ValueError, match="train and val overlap"):
        bad.validate()


def test_incomplete_partition_is_rejected():
    from neuralcosmos.data.splits import SuiteSplit

    bad = SuiteSplit(
        suite="Bad",
        train=(0, 1),
        val=(2,),
        test=(3,),
        n_simulations=5,     # simulation 4 is unassigned
        seed=0,
    )
    with pytest.raises(ValueError, match="does not partition"):
        bad.validate()


def test_degenerate_request_is_rejected():
    with pytest.raises(ValueError, match="leaves no training simulations"):
        make_suite_split("Tiny", 10, n_val=6, n_test=5, master_seed=0)


def test_tampered_split_file_is_detected(tmp_path: Path, split_file):
    path = tmp_path / "split_v1.json"
    split_file.write(path)

    doc = json.loads(path.read_text(encoding="utf-8"))
    # Move one simulation from train to test, keeping the file well-formed.
    moved = doc["splits"]["SIMBA"]["train"].pop()
    doc["splits"]["SIMBA"]["test"].append(moved)
    path.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(ValueError, match="has been modified"):
        load_split_file(path)


def test_round_trip_preserves_everything(tmp_path: Path, split_file):
    path = tmp_path / "split_v1.json"
    split_file.write(path)
    loaded = load_split_file(path)

    assert loaded.content_hash() == split_file.content_hash()
    assert loaded.master_seed == split_file.master_seed
    for name in split_file.splits:
        assert loaded.suite(name) == split_file.suite(name)


def test_content_hash_ignores_timestamp(tmp_path: Path):
    a = build_split_file({"SIMBA": 1000}, 42, 50, 50, MAPS_PER_SIM)
    b = build_split_file({"SIMBA": 1000}, 42, 50, 50, MAPS_PER_SIM)
    assert a.generated_utc is not None
    assert a.content_hash() == b.content_hash()


# --------------------------------------------------------------------------
# The committed split file
# --------------------------------------------------------------------------


def test_committed_split_file_is_valid():
    from neuralcosmos.paths import repo_root

    path = repo_root() / "configs" / "splits" / "split_v1.json"
    if not path.exists():
        pytest.skip("split_v1.json not generated yet")

    sf = load_split_file(path)  # re-validates and checks the hash
    assert sf.maps_per_simulation == MAPS_PER_SIM
    for name in sf.splits:
        sp = sf.suite(name)
        assert (len(sp.train), len(sp.val), len(sp.test)) == (900, 50, 50)


def test_no_cross_suite_id_assumption(split_file):
    """Simulation 7 of TNG and simulation 7 of SIMBA are unrelated.

    They are different universes from different codes. Nothing may assume that
    a shared integer ID implies any relationship, so the splits are allowed to
    disagree between suites -- and in fact should.
    """
    tng = set(split_file.suite("IllustrisTNG").test)
    simba = set(split_file.suite("SIMBA").test)
    assert tng != simba
    # Some incidental overlap is expected by chance; near-total overlap would
    # mean the per-suite seeds are not actually independent.
    assert len(tng & simba) < 25


def test_expanded_map_counts_match_the_plan(split_file):
    sp = split_file.suite("IllustrisTNG")
    assert len(maps_for_simulations(sp.train, MAPS_PER_SIM)) == 13500
    assert len(maps_for_simulations(sp.val, MAPS_PER_SIM)) == 750
    assert len(maps_for_simulations(sp.test, MAPS_PER_SIM)) == 750


def test_split_ids_are_plain_ints_for_json_safety(split_file):
    """numpy integers serialise inconsistently; the split file must hold ints."""
    sp = split_file.suite("SIMBA")
    for s in SPLIT_NAMES:
        for v in sp.ids(s):
            assert type(v) is int
            assert not isinstance(v, np.integer)
