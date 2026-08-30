"""Memory-mapped CAMELS map dataset.

Plan reference: sections 14, 15, 20, 22, 23, 61.

DESIGN NOTES
------------
1. Memory mapping (section 14). A 3.9 GB ``.npy`` is never loaded whole. Each
   ``__getitem__`` reads one 256x256 window. Parameter tables are tiny
   (1000 x 6) and are held in RAM.

2. Lazy per-worker mmap handles. ``np.memmap`` objects cannot be pickled, and
   on Windows ``DataLoader`` workers are spawned rather than forked, so any
   handle opened in the parent would either fail to pickle or be silently
   invalid in the child. Handles are therefore opened on first use *inside*
   whichever process is doing the reading, and cached per process.

3. Normalization is never inferred. The normalizer must be passed in
   explicitly, and it carries a provenance string describing exactly which
   simulations produced it. There is deliberately no "compute statistics from
   whatever data I was given" path, because on a target-suite dataset that
   would silently violate the DG-strict protocol of section 20.2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .cache import QuantSpec
from .splits import maps_for_simulations
from .targets import TargetScaler

__all__ = [
    "LogNormalizer",
    "SuiteSource",
    "CAMELSMapDataset",
    "dihedral_transform",
    "load_and_transform_map",
]


@dataclass(frozen=True)
class LogNormalizer:
    """Standardisation applied in log space.

    ``provenance`` is not decoration. Section 67 asks for methodological
    discipline to be executable; recording which simulations produced these two
    numbers is what lets a protocol check verify that no target-suite data
    contributed to them.
    """

    mean: float
    std: float
    provenance: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.std):
            raise ValueError(f"Non-finite normalizer: mean={self.mean} std={self.std}")
        if self.std <= 0:
            raise ValueError(f"Normalizer std must be positive, got {self.std}")
        if not self.provenance:
            raise ValueError(
                "LogNormalizer requires a provenance string naming the simulations "
                "its statistics were computed from (section 20.2)."
            )

    def apply(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "provenance": self.provenance}


def dihedral_transform(img: np.ndarray, k: int) -> np.ndarray:
    """Apply one of the 8 symmetries of the square (the dihedral group D4).

    ``k`` in 0..7: ``k % 4`` quarter turns, then a horizontal flip if ``k >= 4``.

    Section 23 permits flips and 90-degree rotations for these scalar physical
    fields, and forbids photographic augmentations such as colour jitter. The
    plan also asks that symmetry augmentation be *compared against* no
    augmentation before being adopted, so this is available but off by default.
    """
    if not 0 <= k < 8:
        raise ValueError(f"k must be in 0..7, got {k}")
    out = np.rot90(img, k % 4)
    if k >= 4:
        out = np.fliplr(out)
    # rot90/fliplr return views with negative strides; torch cannot consume those.
    return np.ascontiguousarray(out)


@dataclass
class SuiteSource:
    """One suite's contribution to a dataset."""

    suite: str
    suite_id: int
    map_path: Path
    params: np.ndarray           # (n_simulations, 6) in physical units
    map_indices: np.ndarray      # global map indices belonging to this split
    maps_per_simulation: int
    # When set, values are uint16 log-quantised codes rather than raw float32
    # densities, and decode straight to log10 (section 83).
    quant_spec: QuantSpec | None = None
    # An in-RAM uint16 array. When present it supersedes map_path entirely.
    # A spawned DataLoader worker would copy this, so a RAM-backed dataset must
    # run with num_workers=0 -- which costs nothing, since there is no I/O left
    # to overlap with.
    ram_array: np.ndarray | None = field(default=None, repr=False)
    _handle: Any = field(default=None, init=False, repr=False)
    _pid: int | None = field(default=None, init=False, repr=False)

    @property
    def is_cached(self) -> bool:
        return self.quant_spec is not None

    @property
    def in_ram(self) -> bool:
        return self.ram_array is not None

    def maps(self) -> np.ndarray:
        """Return this suite's map array.

        An in-RAM array is returned directly. Otherwise the memory-mapped file
        is opened, re-opening whenever the process ID changes -- which is what
        makes this safe across spawned DataLoader workers.
        """
        if self.ram_array is not None:
            return self.ram_array
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self._handle = np.load(self.map_path, mmap_mode="r")
            self._pid = pid
        return self._handle

    def __getstate__(self) -> dict[str, Any]:
        # Drop the unpicklable mmap handle before crossing a process boundary.
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state


def load_and_transform_map(
    source: SuiteSource,
    map_id: int,
    log_transform: bool = True,
    normalizer: "LogNormalizer | None" = None,
) -> np.ndarray:
    """Load one map and apply log + normalization, without augmentation.

    Shared by the single-map and paired datasets so the two cannot diverge in
    how a map is turned into a model input. Augmentation is deliberately NOT
    applied here: the paired dataset must apply the SAME dihedral element to
    both views of a pair, which it can only do by owning the transform itself.
    """
    raw = source.maps()[map_id]

    if source.is_cached:
        # Already log10, stored as uint16 codes. Decoding is a multiply-add.
        img = source.quant_spec.decode(np.asarray(raw), dtype=np.float32)
    else:
        img = np.array(raw, dtype=np.float32)
        if log_transform:
            # Positivity was verified across the whole archive by
            # validate_data.py (section 20.1), so no epsilon is added here. A
            # non-positive value would produce -inf and is caught below rather
            # than silently repaired.
            img = np.log10(img, dtype=np.float32)
            if not np.isfinite(img).all():
                raise ValueError(
                    f"Non-finite value after log10 in {source.suite} map {map_id}. "
                    f"Re-run scripts/validate_data.py: the archive is not strictly "
                    f"positive."
                )

    if normalizer is not None:
        img = normalizer.apply(img).astype(np.float32, copy=False)
    return img


class CAMELSMapDataset:
    """A map-level dataset over one or more suites.

    Yields the sample dictionary required by section 15. ``simulation_id`` is
    always included: it is needed for leakage checks, grouped evaluation and
    simulation-level bootstrap intervals (section 57), and must never be
    dropped for convenience.
    """

    def __init__(
        self,
        sources: Sequence[SuiteSource],
        target_scaler: TargetScaler,
        param_columns: dict[str, int],
        normalizer: LogNormalizer | None = None,
        log_transform: bool = True,
        augment: bool = False,
        augment_seed: int = 0,
    ) -> None:
        if not sources:
            raise ValueError("CAMELSMapDataset requires at least one SuiteSource")

        # A cached source always yields log10 values. Mixing that with
        # log_transform=False would hand the model log data from one suite and
        # linear data from another -- a difference far larger than the domain
        # shift being studied, and invisible in any metric.
        cached = [s.suite for s in sources if s.is_cached]
        if cached and not log_transform:
            raise ValueError(
                f"Suites {cached} read from a log-quantised cache, which always "
                f"yields log10 values, but log_transform=False was requested. "
                f"Use the raw archive for a linear-space experiment."
            )

        self.sources = list(sources)
        self.target_scaler = target_scaler
        self.param_columns = dict(param_columns)
        self.normalizer = normalizer
        self.log_transform = log_transform
        self.augment = augment
        self.augment_seed = augment_seed

        self._target_cols = np.array(
            [self.param_columns[n] for n in target_scaler.names], dtype=np.int64
        )

        # Flat index -> (source position, position within that source).
        lengths = [len(s.map_indices) for s in self.sources]
        self._offsets = np.cumsum([0] + lengths)
        self._length = int(self._offsets[-1])

        self._rng: np.random.Generator | None = None
        self._rng_pid: int | None = None

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return self._length

    @property
    def suites(self) -> list[str]:
        return [s.suite for s in self.sources]

    @property
    def in_ram(self) -> bool:
        """True if any source is backed by an in-memory array.

        Callers must then use ``num_workers=0``: a spawned worker would copy
        every RAM-backed array into its own process, multiplying a 3.7 GiB
        footprint by the worker count. There is nothing to gain anyway, since
        the point of the RAM cache is that no I/O remains to overlap.
        """
        return any(s.in_ram for s in self.sources)

    def safe_num_workers(self, requested: int) -> int:
        """Clamp a requested worker count to what this dataset can support."""
        return 0 if self.in_ram else requested

    def simulation_ids(self) -> np.ndarray:
        """Simulation ID of every sample, in dataset order.

        Used by the bootstrap (section 57), which must resample simulations
        rather than maps because 15 maps of one simulation are correlated.
        """
        out = []
        for s in self.sources:
            out.append(s.map_indices // s.maps_per_simulation)
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    def suite_ids(self) -> np.ndarray:
        return np.concatenate(
            [np.full(len(s.map_indices), s.suite_id, dtype=np.int64) for s in self.sources]
        )

    def targets_physical(self) -> np.ndarray:
        """All targets in physical units, in dataset order."""
        out = []
        for s in self.sources:
            sims = s.map_indices // s.maps_per_simulation
            out.append(s.params[np.ix_(sims, self._target_cols)])
        return np.concatenate(out, axis=0)

    # -- sampling ----------------------------------------------------------

    def _rng_for_worker(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            # Mixing the PID keeps workers from drawing identical augmentations
            # while remaining reproducible for a fixed worker layout.
            self._rng = np.random.default_rng((self.augment_seed, pid))
            self._rng_pid = pid
        return self._rng

    def _locate(self, index: int) -> tuple[SuiteSource, int]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"index {index} out of range for dataset of length {self._length}")
        pos = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return self.sources[pos], index - int(self._offsets[pos])

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, local = self._locate(index)
        map_id = int(source.map_indices[local])
        sim_id = map_id // source.maps_per_simulation

        img = load_and_transform_map(
            source, map_id, log_transform=self.log_transform, normalizer=self.normalizer
        )

        if self.augment:
            img = dihedral_transform(img, int(self._rng_for_worker().integers(0, 8)))

        physical = source.params[sim_id, self._target_cols]
        target = self.target_scaler.forward(physical).astype(np.float32)

        return {
            "image": img[None, ...],          # (1, H, W)
            "target": target,                 # (n_targets,) scaled
            "target_physical": physical.astype(np.float32),
            "suite_id": source.suite_id,
            "simulation_id": sim_id,
            "map_id": map_id,
        }
