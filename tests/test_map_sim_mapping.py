"""The map-to-simulation contract.

Plan reference: sections 7.1, 15, 16.

Everything scientific in this project rests on one identity:

    simulation_id = map_index // maps_per_simulation

If it is wrong, labels are attached to the wrong maps, splits leak, and every
bootstrap confidence interval is invalid. These tests pin it down with hand-
computed values rather than by re-deriving it from the same expression under
test.
"""

from __future__ import annotations

import numpy as np
import pytest

MAPS_PER_SIM = 15


def sim_id(map_index: int, maps_per_sim: int = MAPS_PER_SIM) -> int:
    return map_index // maps_per_sim


def map_range(simulation_id: int, maps_per_sim: int = MAPS_PER_SIM) -> range:
    start = simulation_id * maps_per_sim
    return range(start, start + maps_per_sim)


@pytest.mark.parametrize(
    "map_index,expected",
    [
        # Boundaries quoted directly in section 7.1 of the plan.
        (0, 0),
        (14, 0),
        (15, 1),
        (29, 1),
        (14985, 999),
        (14999, 999),
    ],
)
def test_documented_boundaries(map_index: int, expected: int):
    assert sim_id(map_index) == expected


def test_first_and_last_map_of_a_simulation():
    # Off-by-one here is the single most likely silent bug in the pipeline.
    assert sim_id(15 * 7) == 7
    assert sim_id(15 * 7 + 14) == 7
    assert sim_id(15 * 7 - 1) == 6
    assert sim_id(15 * 8) == 8


def test_every_simulation_owns_exactly_fifteen_maps():
    n_sims = 1000
    counts: dict[int, int] = {}
    for i in range(n_sims * MAPS_PER_SIM):
        s = sim_id(i)
        counts[s] = counts.get(s, 0) + 1
    assert len(counts) == n_sims
    assert set(counts.values()) == {MAPS_PER_SIM}


def test_map_range_round_trips():
    for s in (0, 1, 42, 999):
        idxs = list(map_range(s))
        assert len(idxs) == MAPS_PER_SIM
        assert all(sim_id(i) == s for i in idxs)
        assert idxs[0] == s * MAPS_PER_SIM


def test_partition_is_complete_and_disjoint():
    """The union of all per-simulation map ranges must tile 0..N-1 exactly."""
    n_sims = 50
    seen: set[int] = set()
    for s in range(n_sims):
        block = set(map_range(s))
        assert not (seen & block), f"simulation {s} overlaps an earlier simulation"
        seen |= block
    assert seen == set(range(n_sims * MAPS_PER_SIM))


def test_vectorised_mapping_matches_scalar():
    """The loader will use NumPy integer division; confirm it agrees."""
    idx = np.arange(0, 15000)
    vectorised = idx // MAPS_PER_SIM
    scalar = np.array([sim_id(int(i)) for i in idx])
    assert np.array_equal(vectorised, scalar)


def test_labels_broadcast_correctly_to_maps():
    """Each map must receive the parameter vector of its own simulation."""
    rng = np.random.default_rng(0)
    n_sims = 20
    params = rng.uniform(0.1, 0.5, size=(n_sims, 6))

    map_idx = np.arange(n_sims * MAPS_PER_SIM)
    per_map = params[map_idx // MAPS_PER_SIM]

    assert per_map.shape == (n_sims * MAPS_PER_SIM, 6)
    # Every block of 15 consecutive rows must be identical.
    for s in range(n_sims):
        block = per_map[s * MAPS_PER_SIM : (s + 1) * MAPS_PER_SIM]
        assert np.array_equal(block, np.tile(params[s], (MAPS_PER_SIM, 1)))
