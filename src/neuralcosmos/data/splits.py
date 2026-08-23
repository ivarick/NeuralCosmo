"""Simulation-level train/validation/test splits.

Plan reference: sections 16, 17, 64.

WHY THIS IS NON-NEGOTIABLE
--------------------------
Each CAMELS simulation is rendered as 15 maps that share one parameter vector
and one realisation of the density field. Splitting maps at random puts maps
from the same simulation on both sides of the train/test boundary, so the model
can memorise a simulation during training and be scored on a near-duplicate at
test time. The measured error is then an interpolation error, not a
generalisation error, and every number in the paper is inflated.

Splitting by SIMULATION removes that channel entirely. The original CMD
benchmark splits this way for the same reason.

DETERMINISM
-----------
The split for a suite is derived from ``sha256(master_seed:suite_name)`` rather
than from a single global RNG stream. That makes each suite's split independent
of which other suites were requested and of the order they were processed in.
Adding a fourth suite later cannot silently reshuffle the first three, which
would invalidate every result computed before it was added.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "SPLIT_NAMES",
    "SuiteSplit",
    "SplitFile",
    "suite_rng_seed",
    "make_suite_split",
    "build_split_file",
    "load_split_file",
    "maps_for_simulations",
]

SPLIT_NAMES = ("train", "val", "test")


def suite_rng_seed(master_seed: int, suite: str) -> int:
    """A per-suite seed derived deterministically from the master seed."""
    digest = hashlib.sha256(f"{master_seed}:{suite}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


@dataclass(frozen=True)
class SuiteSplit:
    """Simulation IDs assigned to each split for one suite."""

    suite: str
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    n_simulations: int
    seed: int

    def ids(self, split: str) -> tuple[int, ...]:
        if split not in SPLIT_NAMES:
            raise KeyError(f"Unknown split {split!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, split)

    def validate(self) -> None:
        """Assert the invariants of section 17. Raises on any violation."""
        train, val, test = set(self.train), set(self.val), set(self.test)

        if train & val:
            raise ValueError(f"[{self.suite}] train and val overlap: {sorted(train & val)[:10]}")
        if train & test:
            raise ValueError(f"[{self.suite}] train and test overlap: {sorted(train & test)[:10]}")
        if val & test:
            raise ValueError(f"[{self.suite}] val and test overlap: {sorted(val & test)[:10]}")

        union = train | val | test
        expected = set(range(self.n_simulations))
        if union != expected:
            missing = sorted(expected - union)[:10]
            extra = sorted(union - expected)[:10]
            raise ValueError(
                f"[{self.suite}] split does not partition 0..{self.n_simulations - 1}. "
                f"missing={missing} unexpected={extra}"
            )

        total = len(self.train) + len(self.val) + len(self.test)
        if total != self.n_simulations:
            raise ValueError(
                f"[{self.suite}] duplicate simulation IDs: {total} assigned "
                f"but only {len(union)} distinct"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "n_simulations": self.n_simulations,
            "seed": self.seed,
            "counts": {s: len(self.ids(s)) for s in SPLIT_NAMES},
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SuiteSplit:
        return cls(
            suite=d["suite"],
            train=tuple(d["train"]),
            val=tuple(d["val"]),
            test=tuple(d["test"]),
            n_simulations=int(d["n_simulations"]),
            seed=int(d["seed"]),
        )


def make_suite_split(
    suite: str,
    n_simulations: int,
    n_val: int,
    n_test: int,
    master_seed: int,
) -> SuiteSplit:
    """Partition ``0..n_simulations-1`` into train/val/test for one suite."""
    if n_val + n_test >= n_simulations:
        raise ValueError(
            f"[{suite}] n_val + n_test ({n_val + n_test}) leaves no training "
            f"simulations out of {n_simulations}"
        )

    seed = suite_rng_seed(master_seed, suite)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_simulations)

    # Take val and test from the front of the permutation so that shrinking the
    # training set later (the data-efficiency ablation of section 15) never
    # moves a simulation into or out of val/test.
    val = perm[:n_val]
    test = perm[n_val : n_val + n_test]
    train = perm[n_val + n_test :]

    split = SuiteSplit(
        suite=suite,
        train=tuple(sorted(int(i) for i in train)),
        val=tuple(sorted(int(i) for i in val)),
        test=tuple(sorted(int(i) for i in test)),
        n_simulations=n_simulations,
        seed=seed,
    )
    split.validate()
    return split


@dataclass
class SplitFile:
    """A complete, versioned set of splits across suites."""

    version: str
    master_seed: int
    n_val: int
    n_test: int
    maps_per_simulation: int
    generated_utc: str
    splits: dict[str, SuiteSplit]

    def suite(self, name: str) -> SuiteSplit:
        if name not in self.splits:
            known = ", ".join(sorted(self.splits))
            raise KeyError(f"No split for suite {name!r}. Available: {known}")
        return self.splits[name]

    def validate(self) -> None:
        for s in self.splits.values():
            s.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "master_seed": self.master_seed,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "maps_per_simulation": self.maps_per_simulation,
            "generated_utc": self.generated_utc,
            "note": (
                "Simulation-level splits (plan section 16). Generated once and "
                "committed. Do not regenerate per run."
            ),
            "splits": {name: s.to_dict() for name, s in sorted(self.splits.items())},
        }

    def content_hash(self) -> str:
        """Stable hash of the split assignment, ignoring the timestamp.

        Recorded in every run's metadata so a result can always be traced to the
        exact partition that produced it (section 64).
        """
        payload = {
            "version": self.version,
            "master_seed": self.master_seed,
            "splits": {
                name: {s: list(sp.ids(s)) for s in SPLIT_NAMES}
                for name, sp in sorted(self.splits.items())
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.to_dict()
        doc["split_content_hash"] = self.content_hash()
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path


def build_split_file(
    suites: dict[str, int],
    master_seed: int,
    n_val: int,
    n_test: int,
    maps_per_simulation: int,
    version: str = "v1",
) -> SplitFile:
    """Build splits for every suite in ``suites`` (name -> n_simulations)."""
    splits = {
        name: make_suite_split(name, n_sims, n_val, n_test, master_seed)
        for name, n_sims in sorted(suites.items())
    }
    sf = SplitFile(
        version=version,
        master_seed=master_seed,
        n_val=n_val,
        n_test=n_test,
        maps_per_simulation=maps_per_simulation,
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        splits=splits,
    )
    sf.validate()
    return sf


def load_split_file(path: str | Path) -> SplitFile:
    """Load a committed split file and re-validate it.

    Re-validation on load is deliberate: a hand-edited split file is a silent
    catastrophe, and this is the cheapest place to catch one.
    """
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    sf = SplitFile(
        version=doc["version"],
        master_seed=int(doc["master_seed"]),
        n_val=int(doc["n_val"]),
        n_test=int(doc["n_test"]),
        maps_per_simulation=int(doc["maps_per_simulation"]),
        generated_utc=doc.get("generated_utc", ""),
        splits={name: SuiteSplit.from_dict(d) for name, d in doc["splits"].items()},
    )
    sf.validate()

    recorded = doc.get("split_content_hash")
    if recorded and recorded != sf.content_hash():
        raise ValueError(
            f"Split file {path} has been modified: recorded hash {recorded} does not "
            f"match recomputed hash {sf.content_hash()}"
        )
    return sf


def maps_for_simulations(
    simulation_ids: Iterable[int],
    maps_per_simulation: int,
) -> np.ndarray:
    """Expand simulation IDs into the global map indices they own.

    ``simulation_id = map_index // maps_per_simulation``, so simulation *s* owns
    exactly ``[s*k, (s+1)*k)``.
    """
    sims = np.asarray(sorted(set(int(s) for s in simulation_ids)), dtype=np.int64)
    if sims.size == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.arange(maps_per_simulation, dtype=np.int64)
    return (sims[:, None] * maps_per_simulation + offsets[None, :]).ravel()
