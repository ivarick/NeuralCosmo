"""Batch composition for multi-source training.

Plan reference: section 61.

Section 61 requires batches balanced by source suite -- 16 TNG + 16 SIMBA for a
batch of 32 -- so that no domain dominates the alignment loss purely because of
how the batch happened to be filled. With equal-sized suites a shuffled sampler
would be balanced *on average*, but individual batches would fluctuate, and a
domain-adversarial or alignment loss reads each batch separately.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np

__all__ = ["BalancedSuiteBatchSampler"]


class BalancedSuiteBatchSampler:
    """Yields batches holding an equal number of samples from each suite.

    The longest suite defines the epoch length; shorter suites are resampled
    from a fresh shuffled permutation whenever they are exhausted. That keeps
    every batch exactly balanced without truncating the larger suite, at the
    cost of the smaller suite being seen more than once per epoch -- which is
    the right trade when the alternative is discarding data.
    """

    def __init__(
        self,
        suite_ids: Sequence[int] | np.ndarray,
        batch_size: int,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        suites = np.asarray(suite_ids)
        self.unique = np.unique(suites)
        n_suites = len(self.unique)

        if batch_size % n_suites != 0:
            raise ValueError(
                f"batch_size {batch_size} is not divisible by the number of "
                f"suites ({n_suites}); an exactly balanced batch is impossible. "
                f"Use a multiple of {n_suites}."
            )

        self.indices_by_suite = {
            int(s): np.flatnonzero(suites == s).astype(np.int64) for s in self.unique
        }
        for s, idx in self.indices_by_suite.items():
            if idx.size == 0:
                raise ValueError(f"suite {s} contributed no samples")

        self.batch_size = batch_size
        self.per_suite = batch_size // n_suites
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        longest = max(len(v) for v in self.indices_by_suite.values())
        self._n_batches = longest // self.per_suite
        if not drop_last and longest % self.per_suite:
            self._n_batches += 1

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministically per epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng((self.seed, self.epoch))

        pools = {s: rng.permutation(idx) for s, idx in self.indices_by_suite.items()}
        cursors = {s: 0 for s in pools}

        for _ in range(self._n_batches):
            batch: list[int] = []
            for s in sorted(pools):
                need = self.per_suite
                while need > 0:
                    pool, cur = pools[s], cursors[s]
                    take = min(need, len(pool) - cur)
                    if take <= 0:
                        # Exhausted: reshuffle and continue from the start.
                        pools[s] = rng.permutation(self.indices_by_suite[s])
                        cursors[s] = 0
                        continue
                    batch.extend(int(i) for i in pool[cur : cur + take])
                    cursors[s] = cur + take
                    need -= take
            # Shuffle within the batch so suite order carries no positional
            # signal, which matters if any layer is ever order-sensitive.
            rng.shuffle(batch)
            yield batch
