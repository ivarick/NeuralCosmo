"""Assemble datasets from a data config, a split file and a suite selection.

Plan reference: sections 15, 16, 18, 61, 67.

Everything that touches raw files goes through here, so the protocol check has
exactly one place to stand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..protocol import ExperimentProtocol
from .cache import QuantSpec, load_into_ram, open_cache
from .dataset import CAMELSMapDataset, LogNormalizer, SuiteSource
from .manifest import load_data_config, resolve_suite_files
from .paired_dataset import PairedMapDataset, PairedSuiteSource
from .splits import SplitFile, maps_for_simulations
from .targets import TargetScaler

__all__ = [
    "suite_id_map",
    "load_suite_params",
    "build_sources",
    "build_dataset",
    "stats_sources",
    "nbody_map_file",
    "build_paired_dataset",
]


def nbody_map_file(hydro_map_file: str) -> str:
    """Derive the N-body map filename from the hydro one.

    ``Maps_Mtot_IllustrisTNG_LH_z=0.00.npy``
        -> ``Maps_Mtot_IllustrisTNG_Nbody_LH_z=0.00.npy``
    """
    return hydro_map_file.replace("_LH_", "_Nbody_LH_")


def suite_id_map(cfg: dict[str, Any]) -> dict[str, int]:
    """Stable integer IDs for suites.

    Sorted by name rather than by config order, so a config edit cannot silently
    renumber the domain-classifier labels of an already-trained model.
    """
    return {name: i for i, name in enumerate(sorted(cfg["suites"]))}


def load_suite_params(cfg: dict[str, Any], data_root: Path, suite: str) -> np.ndarray:
    """Load one suite's (n_simulations, 6) parameter table."""
    sf = resolve_suite_files(cfg, data_root, [suite])[0]
    params = np.loadtxt(sf.param_path)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    expected = int(cfg["expected"]["n_simulations"])
    if params.shape[0] != expected:
        raise ValueError(
            f"[{suite}] parameter table has {params.shape[0]} rows, expected {expected}. "
            f"Run scripts/validate_data.py."
        )
    return params


def build_sources(
    cfg: dict[str, Any],
    data_root: Path,
    split_file: SplitFile,
    suites: Sequence[str],
    split: str,
    max_simulations: int | None = None,
    cache_root: Path | None = None,
    ram_cache: bool = False,
    ram_suites: Sequence[str] | None = None,
    progress=None,
) -> list[SuiteSource]:
    """Build one :class:`SuiteSource` per requested suite for a given split.

    ``max_simulations`` truncates the *training* simulation list, for smoke runs
    (section 68 Phase 2) and the data-efficiency ablation (section 15). Because
    validation and test simulations are drawn from the front of the permutation
    (see ``splits.make_suite_split``), truncating train never disturbs them.

    ``cache_root`` opts into the on-disk uint16 log-quantised cache of section
    83. A suite that has no cache there falls back to the raw archive, so a
    partial cache (source suites only) is a supported configuration.

    ``ram_cache`` instead quantises into memory: 1.83 GiB per suite, no disk
    space, and faster than any disk. ``ram_suites`` restricts which suites are
    held in RAM, so the sealed target can stream from the archive while the
    source suites stay resident. The store is process-global and keyed by file,
    so the train and validation datasets of one suite share a single array.
    """
    ids = suite_id_map(cfg)
    maps_per_sim = int(cfg["maps_per_simulation"])
    field = str(cfg.get("field", "Mtot"))
    sources: list[SuiteSource] = []

    for suite in suites:
        sf = resolve_suite_files(cfg, data_root, [suite])[0]
        sim_ids = list(split_file.suite(suite).ids(split))

        if max_simulations is not None:
            if split != "train":
                raise ValueError(
                    "max_simulations may only truncate the training split; truncating "
                    f"{split!r} would change what the model is evaluated on."
                )
            sim_ids = sim_ids[:max_simulations]

        map_path, quant_spec, ram_array = sf.map_path, None, None

        want_ram = ram_cache and (ram_suites is None or suite in ram_suites)
        if want_ram:
            quant_spec = QuantSpec()
            ram_array = load_into_ram(
                sf.map_path,
                quant_spec,
                progress=(lambda d, t, _s=suite: progress(_s, d, t)) if progress else None,
            )
        elif cache_root is not None:
            try:
                map_path, quant_spec = open_cache(Path(cache_root), suite, field)
            except FileNotFoundError:
                # Falling back to the raw archive is correct, not a failure: the
                # cache is a speed optimisation and is intentionally partial.
                pass

        sources.append(
            SuiteSource(
                suite=suite,
                suite_id=ids[suite],
                map_path=map_path,
                params=load_suite_params(cfg, data_root, suite),
                map_indices=maps_for_simulations(sim_ids, maps_per_sim),
                maps_per_simulation=maps_per_sim,
                quant_spec=quant_spec,
                ram_array=ram_array,
            )
        )
    return sources


def build_dataset(
    cfg: dict[str, Any],
    data_root: Path,
    split_file: SplitFile,
    suites: Sequence[str],
    split: str,
    normalizer: LogNormalizer | None,
    protocol: ExperimentProtocol | None = None,
    role: str | None = None,
    log_transform: bool = True,
    augment: bool = False,
    augment_seed: int = 0,
    max_simulations: int | None = None,
    cache_root: Path | None = None,
    ram_cache: bool = False,
    ram_suites: Sequence[str] | None = None,
    progress=None,
) -> CAMELSMapDataset:
    """Build a dataset, enforcing the experiment protocol.

    ``role`` is one of ``train``, ``val`` or ``eval`` and selects which protocol
    check applies. It defaults to ``split``, so the common case is safe without
    the caller having to remember.
    """
    role = role or ("val" if split == "val" else "train" if split == "train" else "eval")

    if protocol is not None:
        if role == "train":
            protocol.check_training_suites(suites)
        elif role == "val":
            protocol.check_validation_suites(suites)
        else:
            protocol.check_evaluation_suites(suites)

        if normalizer is not None:
            protocol.check_normalizer(normalizer.provenance)

    sources = build_sources(
        cfg, data_root, split_file, suites, split,
        max_simulations=max_simulations, cache_root=cache_root,
        ram_cache=ram_cache, ram_suites=ram_suites, progress=progress,
    )
    return CAMELSMapDataset(
        sources=sources,
        target_scaler=TargetScaler.from_config(cfg),
        param_columns=cfg["param_columns"],
        normalizer=normalizer,
        log_transform=log_transform,
        augment=augment,
        augment_seed=augment_seed,
    )


def stats_sources(
    cfg: dict[str, Any],
    data_root: Path,
    split_file: SplitFile,
    source_suites: Sequence[str],
    protocol: ExperimentProtocol | None = None,
    max_simulations: int | None = None,
) -> list[tuple[str, Path, np.ndarray]]:
    """Triples for :func:`statistics.compute_log_stats`, training split only.

    Hard-codes ``split="train"``. Normalization statistics must not see
    validation or test simulations, even from a source suite: those are used for
    model selection and reporting, and letting their pixel statistics into the
    input scaling is a small but real leak.
    """
    if protocol is not None:
        protocol.check_training_suites(source_suites)

    sources = build_sources(
        cfg, data_root, split_file, source_suites, "train", max_simulations=max_simulations
    )
    return [(s.suite, s.map_path, s.map_indices) for s in sources]


def make_provenance(
    source_suites: Sequence[str],
    split_file: SplitFile,
    split: str = "train",
    log_transform: bool = True,
    extra: str = "",
) -> str:
    """A human- and machine-checkable description of what produced a normalizer."""
    parts = [
        "log10" if log_transform else "linear",
        f"{split}",
        "+".join(sorted(source_suites)),
        f"split_{split_file.version}:{split_file.content_hash()[:12]}",
    ]
    if extra:
        parts.append(extra)
    return "|".join(parts)


def build_paired_dataset(
    cfg: dict[str, Any],
    data_root: Path,
    split_file: SplitFile,
    suites: Sequence[str],
    split: str,
    hydro_normalizer: LogNormalizer | None,
    nbody_normalizer: LogNormalizer | None = None,
    protocol: ExperimentProtocol | None = None,
    role: str = "train",
    log_transform: bool = True,
    augment: bool = False,
    augment_seed: int = 0,
    max_simulations: int | None = None,
    shuffle_pairs: bool = False,
    shuffle_seed: int = 0,
    verify_pairs: bool = True,
) -> PairedMapDataset:
    """Assemble a :class:`PairedMapDataset` from config, split and N-body files.

    Enforces the same protocol as :func:`build_dataset`: a paired training set
    made of source suites still cannot include the sealed target, and the
    normalizer provenance is still checked. ``shuffle_pairs`` selects the
    section 50 control.
    """
    if protocol is not None:
        if role == "train":
            protocol.check_training_suites(suites)
        elif role == "val":
            protocol.check_validation_suites(suites)
        else:
            protocol.check_evaluation_suites(suites)
        if hydro_normalizer is not None:
            protocol.check_normalizer(hydro_normalizer.provenance)
        if nbody_normalizer is not None:
            protocol.check_normalizer(nbody_normalizer.provenance)

    ids = suite_id_map(cfg)
    maps_per_sim = int(cfg["maps_per_simulation"])
    paired_sources: list[PairedSuiteSource] = []

    for suite in suites:
        sf = resolve_suite_files(cfg, data_root, [suite])[0]
        nbody_path = data_root / nbody_map_file(cfg["suites"][suite]["map_file"])
        if not nbody_path.exists():
            raise FileNotFoundError(
                f"[{suite}] N-body map file not found: {nbody_path}. "
                f"Download the source N-body pairs (Phase 7) before paired training."
            )

        sim_ids = list(split_file.suite(suite).ids(split))
        if max_simulations is not None:
            if split != "train":
                raise ValueError("max_simulations may only truncate the training split")
            sim_ids = sim_ids[:max_simulations]
        idx = maps_for_simulations(sim_ids, maps_per_sim)
        params = load_suite_params(cfg, data_root, suite)

        hydro = SuiteSource(
            suite=suite, suite_id=ids[suite], map_path=sf.map_path,
            params=params, map_indices=idx, maps_per_simulation=maps_per_sim,
        )
        nbody = SuiteSource(
            suite=suite, suite_id=ids[suite], map_path=nbody_path,
            params=params, map_indices=idx, maps_per_simulation=maps_per_sim,
        )
        paired_sources.append(PairedSuiteSource(suite=suite, hydro=hydro, nbody=nbody))

    return PairedMapDataset(
        sources=paired_sources,
        target_scaler=TargetScaler.from_config(cfg),
        param_columns=cfg["param_columns"],
        hydro_normalizer=hydro_normalizer,
        nbody_normalizer=nbody_normalizer,
        log_transform=log_transform,
        augment=augment,
        augment_seed=augment_seed,
        shuffle_pairs=shuffle_pairs,
        shuffle_seed=shuffle_seed,
        verify_pairs=verify_pairs,
    )
