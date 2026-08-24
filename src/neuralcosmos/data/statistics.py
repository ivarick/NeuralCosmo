"""Normalization statistics, computed from source-training data only.

Plan reference: sections 20.2, 21, 67.

THE RULE
--------
Under the DG-strict condition, the mean and standard deviation used to
normalize every input -- including the sealed target suite -- must come from
the *source training simulations* alone. Normalizing the target suite with its
own statistics is legitimate under unsupervised adaptation, but it is target
information, and using it silently would turn a domain-generalization claim
into a domain-adaptation claim (section 21, Kill 6).

This module therefore refuses to compute statistics without being told which
simulations it may look at. There is no convenience path that infers the answer
from "whatever data is at hand".

READ ORDER
----------
Training map indices are scattered across the file (900 randomly chosen
simulations, 15 consecutive maps each). Reading them one at a time is
pathological on a spinning or USB-attached disk. Indices are therefore grouped
into maximal contiguous runs and read as slices, which turns ~13500 random
reads into ~900 sequential ones.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .dataset import LogNormalizer

__all__ = [
    "StatsAccumulator",
    "contiguous_runs",
    "compute_log_stats",
    "save_normalizer",
    "load_normalizer",
]


@dataclass
class StatsAccumulator:
    """Streaming mean and standard deviation in float64."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")

    def update(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64)
        self.count += int(v.size)
        self.total += float(v.sum())
        self.total_sq += float(np.square(v).sum())
        if v.size:
            self.minimum = min(self.minimum, float(v.min()))
            self.maximum = max(self.maximum, float(v.max()))

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("No values accumulated")
        return self.total / self.count

    @property
    def std(self) -> float:
        if self.count < 2:
            raise ValueError("Need at least two values for a standard deviation")
        var = self.total_sq / self.count - self.mean**2
        if var < 0:
            # Catastrophic cancellation would mean the sums have lost precision.
            # With float64 accumulation over ~10^9 log-space values this should
            # not happen; if it does, fail rather than return a fabricated 0.
            if var < -1e-6 * abs(self.mean):
                raise ValueError(f"Numerically unstable variance: {var}")
            var = 0.0
        return float(np.sqrt(var))


def contiguous_runs(indices: Sequence[int] | np.ndarray) -> list[tuple[int, int]]:
    """Group sorted indices into maximal ``[start, stop)`` runs."""
    idx = np.asarray(sorted(set(int(i) for i in indices)), dtype=np.int64)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [idx.size]])
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, stops)]


def compute_log_stats(
    sources: Iterable[tuple[str, Path, np.ndarray]],
    log_transform: bool = True,
    max_maps_per_suite: int | None = None,
    subsample_seed: int = 0,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[StatsAccumulator, dict[str, Any]]:
    """Accumulate pixel statistics over the given (suite, path, map_indices).

    Parameters
    ----------
    sources:
        Triples of suite name, ``.npy`` path, and the exact global map indices
        that may contribute. The caller is responsible for having derived those
        indices from source-training simulations only.
    max_maps_per_suite:
        Optionally subsample for speed. The subsample is deterministic and is
        recorded in the returned metadata so the statistics remain reproducible.
    """
    acc = StatsAccumulator()
    meta: dict[str, Any] = {"suites": [], "log_transform": log_transform}

    for suite, path, indices in sources:
        idx = np.asarray(indices, dtype=np.int64)
        n_available = int(idx.size)

        if max_maps_per_suite is not None and n_available > max_maps_per_suite:
            rng = np.random.default_rng((subsample_seed, suite))
            idx = np.sort(rng.choice(idx, size=max_maps_per_suite, replace=False))
            subsampled = True
        else:
            subsampled = False

        runs = contiguous_runs(idx)
        arr = np.load(path, mmap_mode="r")
        done = 0
        for start, stop in runs:
            block = np.asarray(arr[start:stop], dtype=np.float64)
            if log_transform:
                if np.any(block <= 0):
                    raise ValueError(
                        f"[{suite}] non-positive pixel in maps [{start}, {stop}). "
                        f"Section 20.1 forbids applying log without inspecting this."
                    )
                block = np.log10(block)
            acc.update(block)
            done += stop - start
            if progress is not None:
                progress(suite, done, int(idx.size))
        del arr

        meta["suites"].append(
            {
                "suite": suite,
                "n_maps_available": n_available,
                "n_maps_used": int(idx.size),
                "n_runs": len(runs),
                "subsampled": subsampled,
            }
        )

    return acc, meta


def save_normalizer(
    path: Path,
    normalizer: LogNormalizer,
    acc: StatsAccumulator,
    meta: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "normalizer": normalizer.to_dict(),
        "statistics": {
            "n_values": acc.count,
            "mean": acc.mean,
            "std": acc.std,
            "min": acc.minimum,
            "max": acc.maximum,
        },
        "metadata": meta,
        "note": (
            "Computed from source-TRAINING simulations only (plan section 20.2). "
            "Applying these to a target suite is the DG-strict condition. Never "
            "recompute them on target data without relabelling the experiment as "
            "target-aware adaptation (section 21)."
        ),
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_normalizer(path: str | Path) -> LogNormalizer:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    n = doc["normalizer"]
    return LogNormalizer(mean=float(n["mean"]), std=float(n["std"]), provenance=n["provenance"])
