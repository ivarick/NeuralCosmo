"""Matched hydrodynamic / N-body map pairs.

Plan reference: sections 41, 42, 45, 50.

This is the dataset the proposed method rests on. Each sample is a hydrodynamic
Mtot map together with its gravity-only (N-body) counterpart from the SAME
region of the SAME universe -- same cosmology, same initial random field, same
spatial slice. The N-body view lacks the hydrodynamic suite's baryonic feedback,
so it can serve as a nuisance-reduced anchor (section 4.1).

The premise is only meaningful if the correspondence is exact. Section 42 is
emphatic: before training, verify that a hydro map and its claimed N-body
partner share simulation index, map index, and cosmological parameters. If they
do not, the pairing is broken and the method has no foundation. Those checks are
therefore not optional diagnostics run elsewhere -- they are assertions inside
this dataset, on by default, so a broken pairing raises at construction rather
than silently training on mismatched views.

SHUFFLE CONTROL (section 50)
----------------------------
The single most important control in the paper. ``shuffle_pairs`` breaks the
correspondence on purpose: each hydro map is paired with a RANDOM N-body map of
the same suite, chosen to preserve the marginal target distribution. If a model
trained on correct pairs does no better than one trained on shuffled pairs, the
claimed pair mechanism is unsupported and Kill 4 applies. Building the control
into the same class guarantees it differs from the real thing in exactly one
respect -- the correspondence -- and nothing else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dataset import LogNormalizer, SuiteSource, dihedral_transform, load_and_transform_map
from .targets import TargetScaler

__all__ = ["PairedSuiteSource", "PairedMapDataset"]


@dataclass
class PairedSuiteSource:
    """One suite's hydro source and its matched N-body source.

    Both are ordinary :class:`SuiteSource` objects over the SAME split, so all
    the existing machinery -- memory mapping, the quantised cache, the RAM
    cache, pickling across workers -- applies to each unchanged.
    """

    suite: str
    hydro: SuiteSource
    nbody: SuiteSource

    def __post_init__(self) -> None:
        if self.hydro.suite_id != self.nbody.suite_id:
            raise ValueError(
                f"[{self.suite}] hydro and N-body carry different suite ids "
                f"({self.hydro.suite_id} vs {self.nbody.suite_id})"
            )
        if not np.array_equal(self.hydro.map_indices, self.nbody.map_indices):
            raise ValueError(
                f"[{self.suite}] hydro and N-body cover different maps. Both must "
                f"be built over the same split so map i corresponds to map i."
            )
        if self.hydro.maps_per_simulation != self.nbody.maps_per_simulation:
            raise ValueError(f"[{self.suite}] maps_per_simulation differs between views")

    @property
    def map_indices(self) -> np.ndarray:
        return self.hydro.map_indices

    @property
    def suite_id(self) -> int:
        return self.hydro.suite_id

    @property
    def maps_per_simulation(self) -> int:
        return self.hydro.maps_per_simulation

    @property
    def in_ram(self) -> bool:
        return self.hydro.in_ram or self.nbody.in_ram


class PairedMapDataset:
    """Yields matched hydro/N-body samples (section 41).

    Each item is::

        {
            "hydro_image":   x_h,      # (1, H, W)
            "nbody_image":   x_n,      # (1, H, W)
            "target":        y,        # (n_targets,) scaled
            "suite_id":      d,
            "simulation_id": s,
            "map_id":        m,
        }
    """

    def __init__(
        self,
        sources: list[PairedSuiteSource],
        target_scaler: TargetScaler,
        param_columns: dict[str, int],
        hydro_normalizer: LogNormalizer | None = None,
        nbody_normalizer: LogNormalizer | None = None,
        log_transform: bool = True,
        augment: bool = False,
        augment_seed: int = 0,
        shuffle_pairs: bool = False,
        shuffle_seed: int = 0,
        verify_pairs: bool = True,
    ) -> None:
        if not sources:
            raise ValueError("PairedMapDataset requires at least one PairedSuiteSource")

        self.sources = list(sources)
        self.target_scaler = target_scaler
        self.param_columns = dict(param_columns)
        self.hydro_normalizer = hydro_normalizer
        # If no separate N-body normalizer is given, reuse the hydro one. That is
        # the DG-strict default: a single source-train statistic applied to both
        # views keeps the anchor on the same scale as what it anchors.
        self.nbody_normalizer = nbody_normalizer or hydro_normalizer
        self.log_transform = log_transform
        self.augment = augment
        self.augment_seed = augment_seed
        self.shuffle_pairs = shuffle_pairs

        self._target_cols = np.array(
            [self.param_columns[n] for n in target_scaler.names], dtype=np.int64
        )

        lengths = [len(s.map_indices) for s in self.sources]
        self._offsets = np.cumsum([0] + lengths)
        self._length = int(self._offsets[-1])

        # Shuffled-pair permutation (section 50). Built once per source and
        # stored, so the broken correspondence is fixed for the whole run rather
        # than re-randomising every epoch, which would make the control a moving
        # target. Permutation is WITHIN a suite so the N-body marginal is
        # unchanged -- the control isolates correspondence, not distribution.
        self._nbody_perm: list[np.ndarray] = []
        rng = np.random.default_rng(shuffle_seed)
        for s in self.sources:
            n = len(s.map_indices)
            self._nbody_perm.append(rng.permutation(n) if shuffle_pairs else np.arange(n))

        if verify_pairs:
            self._verify()

        self._rng: np.random.Generator | None = None
        self._rng_pid: int | None = None

    # -- verification (section 42) -----------------------------------------

    def _verify(self, n_check: int | None = None) -> None:
        """Assert hydro and N-body agree on identity for a sample of pairs.

        Checks map index and cosmological parameters. With ``shuffle_pairs`` the
        indices are DELIBERATELY broken, so only the correct-pair case asserts
        index equality; both cases still assert the target distribution is
        preserved, which is the invariant the shuffle control must maintain.
        """
        for src, perm in zip(self.sources, self._nbody_perm):
            n = len(src.map_indices)
            step = 1 if n_check is None else max(1, n // n_check)
            for local in range(0, n, step):
                h_map = int(src.hydro.map_indices[local])
                n_map = int(src.nbody.map_indices[perm[local]])
                h_sim = h_map // src.maps_per_simulation
                n_sim = n_map // src.maps_per_simulation

                if not self.shuffle_pairs:
                    if h_map != n_map:
                        raise ValueError(
                            f"[{src.suite}] pair {local}: hydro map {h_map} != "
                            f"N-body map {n_map}. Correspondence is broken "
                            f"(section 42)."
                        )
                    h_y = src.hydro.params[h_sim, self._target_cols]
                    n_y = src.nbody.params[n_sim, self._target_cols]
                    if not np.allclose(h_y, n_y, atol=1e-4):
                        raise ValueError(
                            f"[{src.suite}] pair {local}: hydro cosmology {h_y} != "
                            f"N-body cosmology {n_y}. The paired N-body run must "
                            f"share the hydro run's parameters (section 42)."
                        )

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return self._length

    @property
    def in_ram(self) -> bool:
        return any(s.in_ram for s in self.sources)

    def safe_num_workers(self, requested: int) -> int:
        return 0 if self.in_ram else requested

    def simulation_ids(self) -> np.ndarray:
        return np.concatenate(
            [s.map_indices // s.maps_per_simulation for s in self.sources]
        )

    def suite_ids(self) -> np.ndarray:
        return np.concatenate(
            [np.full(len(s.map_indices), s.suite_id, dtype=np.int64) for s in self.sources]
        )

    # -- sampling ----------------------------------------------------------

    def _rng_for_worker(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            self._rng = np.random.default_rng((self.augment_seed, pid))
            self._rng_pid = pid
        return self._rng

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"index {index} out of range for length {self._length}")
        pos = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return pos, index - int(self._offsets[pos])

    def __getitem__(self, index: int) -> dict[str, Any]:
        pos, local = self._locate(index)
        src = self.sources[pos]

        h_map = int(src.hydro.map_indices[local])
        n_map = int(src.nbody.map_indices[self._nbody_perm[pos][local]])
        sim_id = h_map // src.maps_per_simulation

        x_h = load_and_transform_map(
            src.hydro, h_map, log_transform=self.log_transform,
            normalizer=self.hydro_normalizer,
        )
        x_n = load_and_transform_map(
            src.nbody, n_map, log_transform=self.log_transform,
            normalizer=self.nbody_normalizer,
        )

        if self.augment:
            # The SAME dihedral element applied to both views. A pair-consistency
            # loss compares corresponding regions, so augmenting the two views
            # independently would destroy exactly the correspondence the loss
            # depends on.
            k = int(self._rng_for_worker().integers(0, 8))
            x_h = dihedral_transform(x_h, k)
            x_n = dihedral_transform(x_n, k)

        physical = src.hydro.params[sim_id, self._target_cols]
        target = self.target_scaler.forward(physical).astype(np.float32)

        return {
            "hydro_image": x_h[None, ...],
            "nbody_image": x_n[None, ...],
            "target": target,
            "target_physical": physical.astype(np.float32),
            "suite_id": src.suite_id,
            "simulation_id": sim_id,
            "map_id": h_map,
        }
