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
from .dataset import CAMELSMapDataset, LogNormalizer, SuiteSource
from .manifest import load_data_config, resolve_suite_files
from .splits import SplitFile, maps_for_simulations
from .targets import TargetScaler

__all__ = [
    "suite_id_map",
    "load_suite_params",
    "build_sources",
    "build_dataset",
    "stats_sources",
]


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
) -> list[SuiteSource]:
    """Build one :class:`SuiteSource` per requested suite for a given split.

    ``max_simulations`` truncates the *training* simulation list, for smoke runs
    (section 68 Phase 2) and the data-efficiency ablation (section 15). Because
    validation and test simulations are drawn from the front of the permutation
    (see ``splits.make_suite_split``), truncating train never disturbs them.
    """
    ids = suite_id_map(cfg)
    maps_per_sim = int(cfg["maps_per_simulation"])
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

        sources.append(
            SuiteSource(
                suite=suite,
                suite_id=ids[suite],
                map_path=sf.map_path,
                params=load_suite_params(cfg, data_root, suite),
                map_indices=maps_for_simulations(sim_ids, maps_per_sim),
                maps_per_simulation=maps_per_sim,
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
        cfg, data_root, split_file, suites, split, max_simulations=max_simulations
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
