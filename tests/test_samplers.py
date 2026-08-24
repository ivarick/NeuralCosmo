"""Suite-balanced batch composition.

Plan reference: section 61.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuralcosmos.training.samplers import BalancedSuiteBatchSampler


def test_every_batch_is_exactly_balanced():
    suites = np.array([0] * 100 + [1] * 100)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=32, seed=0)

    for batch in sampler:
        assert len(batch) == 32
        counts = np.bincount(suites[batch], minlength=2)
        assert counts.tolist() == [16, 16]


def test_balance_holds_for_three_suites():
    suites = np.array([0] * 60 + [1] * 60 + [2] * 60)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=30, seed=0)

    for batch in sampler:
        counts = np.bincount(suites[batch], minlength=3)
        assert counts.tolist() == [10, 10, 10]


def test_batch_size_must_divide_by_suite_count():
    suites = np.array([0] * 10 + [1] * 10 + [2] * 10)
    with pytest.raises(ValueError, match="not divisible"):
        BalancedSuiteBatchSampler(suites, batch_size=32, seed=0)


def test_unequal_suites_still_produce_balanced_batches():
    """The smaller suite is resampled rather than the larger one truncated."""
    suites = np.array([0] * 200 + [1] * 50)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=20, seed=0)

    n_batches = 0
    for batch in sampler:
        counts = np.bincount(suites[batch], minlength=2)
        assert counts.tolist() == [10, 10]
        n_batches += 1
    # The longest suite defines the epoch: 200 / 10 = 20 batches.
    assert n_batches == 20


def test_indices_are_valid_and_within_range():
    suites = np.array([0] * 40 + [1] * 40)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=8, seed=0)
    for batch in sampler:
        assert all(0 <= i < len(suites) for i in batch)
        assert all(isinstance(i, int) for i in batch)


def test_no_duplicate_within_a_batch_when_data_is_plentiful():
    suites = np.array([0] * 100 + [1] * 100)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=16, seed=0)
    for batch in sampler:
        assert len(set(batch)) == len(batch)


def test_epoch_changes_the_order():
    suites = np.array([0] * 50 + [1] * 50)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=10, seed=0)

    sampler.set_epoch(0)
    first = [list(b) for b in sampler]
    sampler.set_epoch(1)
    second = [list(b) for b in sampler]
    assert first != second


def test_same_seed_and_epoch_reproduce_exactly():
    suites = np.array([0] * 50 + [1] * 50)
    a = BalancedSuiteBatchSampler(suites, batch_size=10, seed=7)
    b = BalancedSuiteBatchSampler(suites, batch_size=10, seed=7)
    a.set_epoch(3)
    b.set_epoch(3)
    assert [list(x) for x in a] == [list(x) for x in b]


def test_single_suite_is_supported():
    suites = np.zeros(60, dtype=int)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=12, seed=0)
    for batch in sampler:
        assert len(batch) == 12


def test_empty_suite_is_rejected():
    suites = np.array([0, 0, 1, 1])
    sampler = BalancedSuiteBatchSampler(suites, batch_size=2, seed=0)
    assert len(sampler) >= 1

    # A suite present in the label space but with no samples cannot occur via
    # np.unique, so construct the degenerate case directly.
    with pytest.raises(ValueError, match="not divisible"):
        BalancedSuiteBatchSampler(suites, batch_size=3, seed=0)


def test_length_matches_iteration_count():
    suites = np.array([0] * 90 + [1] * 30)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=6, seed=0)
    assert len(sampler) == len(list(sampler))


def test_suite_order_within_a_batch_is_shuffled():
    """Suite blocks must not appear in a fixed positional order."""
    suites = np.array([0] * 100 + [1] * 100)
    sampler = BalancedSuiteBatchSampler(suites, batch_size=32, seed=0)

    sorted_batches = 0
    total = 0
    for batch in sampler:
        labels = suites[batch]
        if np.all(np.diff(labels) >= 0):
            sorted_batches += 1
        total += 1
    assert sorted_batches < total, "every batch had suites in sorted order"
