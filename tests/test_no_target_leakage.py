"""Protocol enforcement: the sealed target must be unreachable.

Plan reference: sections 18, 19, 21, 66.3, 67, 78.

These tests are the executable form of the research-integrity rules. Each one
attempts a specific leak and asserts that the code refuses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralcosmos.data.builders import build_dataset, stats_sources
from neuralcosmos.data.dataset import LogNormalizer
from neuralcosmos.data.manifest import load_data_config
from neuralcosmos.data.splits import build_split_file
from neuralcosmos.protocol import (
    ExperimentProtocol,
    ProtocolViolation,
    default_dg_protocol,
)

MAPS_PER_SIM = 15


@pytest.fixture
def cfg(synthetic_config: Path):
    return load_data_config(synthetic_config)


@pytest.fixture
def split_file():
    return build_split_file(
        suites={"SuiteA": 6, "SuiteB": 6},
        master_seed=1,
        n_val=1,
        n_test=1,
        maps_per_simulation=MAPS_PER_SIM,
    )


@pytest.fixture
def dg():
    return default_dg_protocol(["SuiteA"], ["SuiteB"])


@pytest.fixture
def clean_norm():
    return LogNormalizer(mean=11.0, std=0.5, provenance="log10|train|SuiteA|split_v1:abc")


# --------------------------------------------------------------------------
# Protocol construction
# --------------------------------------------------------------------------


def test_primary_condition_forbids_everything_by_default(dg):
    assert dg.is_strict_dg
    assert not dg.allow_target_unlabeled
    assert not dg.allow_target_statistics
    assert not dg.allow_target_labels


def test_suite_cannot_be_both_source_and_target():
    with pytest.raises(ValueError, match="cannot be both source and target"):
        ExperimentProtocol(source_suites=("A", "B"), target_suites=("B",))


def test_protocol_requires_a_source():
    with pytest.raises(ValueError, match="at least one source suite"):
        ExperimentProtocol(source_suites=(), target_suites=("B",))


def test_from_config_reads_the_experiment_block():
    proto = ExperimentProtocol.from_config(
        {
            "data": {"source_suites": ["IllustrisTNG", "SIMBA"], "target_suites": ["Astrid"]},
            "protocol": {
                "use_target_for_training": False,
                "use_target_for_validation": False,
                "use_target_statistics": False,
            },
        }
    )
    assert proto.is_strict_dg
    assert proto.target_suites == ("Astrid",)


# --------------------------------------------------------------------------
# Training-time leaks
# --------------------------------------------------------------------------


def test_target_suite_cannot_enter_the_training_set(dg):
    with pytest.raises(ProtocolViolation, match="forbids target-suite data during training"):
        dg.check_training_suites(["SuiteA", "SuiteB"])


def test_target_suite_in_training_is_allowed_when_explicitly_declared():
    """Adaptation is legitimate research; it just must not be called DG."""
    adaptation = ExperimentProtocol(
        source_suites=("SuiteA",),
        target_suites=("SuiteB",),
        allow_target_unlabeled=True,
        name="uda",
    )
    adaptation.check_training_suites(["SuiteA", "SuiteB"])  # must not raise
    assert not adaptation.is_strict_dg


def test_build_dataset_blocks_target_in_training(cfg, synthetic_archive, split_file, dg, clean_norm):
    with pytest.raises(ProtocolViolation):
        build_dataset(
            cfg, synthetic_archive, split_file, ["SuiteA", "SuiteB"], "train",
            clean_norm, protocol=dg, role="train",
        )


def test_build_dataset_allows_source_only_training(cfg, synthetic_archive, split_file, dg, clean_norm):
    ds = build_dataset(
        cfg, synthetic_archive, split_file, ["SuiteA"], "train",
        clean_norm, protocol=dg, role="train",
    )
    assert len(ds) > 0


# --------------------------------------------------------------------------
# Model-selection leaks (section 19)
# --------------------------------------------------------------------------


def test_target_suite_cannot_be_used_for_validation(dg):
    """No gradient flows from validation, but every hyperparameter does."""
    with pytest.raises(ProtocolViolation, match="forbids selecting models on target data"):
        dg.check_validation_suites(["SuiteB"])


def test_build_dataset_blocks_target_in_validation(cfg, synthetic_archive, split_file, dg, clean_norm):
    with pytest.raises(ProtocolViolation):
        build_dataset(
            cfg, synthetic_archive, split_file, ["SuiteB"], "val",
            clean_norm, protocol=dg, role="val",
        )


# --------------------------------------------------------------------------
# Normalization leaks (sections 20.2, 21)
# --------------------------------------------------------------------------


def test_normalizer_derived_from_target_is_rejected(dg):
    tainted = "log10|train|SuiteA+SuiteB|split_v1:abc"
    with pytest.raises(ProtocolViolation, match="forbids target statistics"):
        dg.check_normalizer(tainted)


def test_clean_normalizer_passes(dg, clean_norm):
    dg.check_normalizer(clean_norm.provenance)  # must not raise


def test_build_dataset_rejects_tainted_normalizer(cfg, synthetic_archive, split_file, dg):
    tainted = LogNormalizer(mean=11.0, std=0.5, provenance="log10|train|SuiteB|split_v1:abc")
    with pytest.raises(ProtocolViolation, match="forbids target statistics"):
        build_dataset(
            cfg, synthetic_archive, split_file, ["SuiteA"], "train",
            tainted, protocol=dg, role="train",
        )


def test_target_aware_condition_permits_target_statistics():
    """Section 21 allows this as a separate, clearly labelled analysis."""
    aware = ExperimentProtocol(
        source_suites=("SuiteA",),
        target_suites=("SuiteB",),
        allow_target_statistics=True,
        name="target_stat_aware",
    )
    aware.check_normalizer("log10|train|SuiteA+SuiteB|split_v1:abc")  # must not raise
    assert not aware.is_strict_dg


# --------------------------------------------------------------------------
# Statistics must come from source TRAIN only
# --------------------------------------------------------------------------


def test_stats_sources_rejects_target_suite(cfg, synthetic_archive, split_file, dg):
    with pytest.raises(ProtocolViolation):
        stats_sources(cfg, synthetic_archive, split_file, ["SuiteA", "SuiteB"], protocol=dg)


def test_stats_sources_uses_only_training_simulations(cfg, synthetic_archive, split_file, dg):
    triples = stats_sources(cfg, synthetic_archive, split_file, ["SuiteA"], protocol=dg)
    assert len(triples) == 1
    _, _, indices = triples[0]

    train_sims = set(split_file.suite("SuiteA").train)
    val_sims = set(split_file.suite("SuiteA").val)
    test_sims = set(split_file.suite("SuiteA").test)

    used_sims = {int(i) // MAPS_PER_SIM for i in indices}
    assert used_sims <= train_sims
    assert used_sims & val_sims == set()
    assert used_sims & test_sims == set()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def test_evaluation_on_target_is_permitted(cfg, synthetic_archive, split_file, dg, clean_norm):
    """The whole point is to evaluate on the target -- once, after freezing."""
    ds = build_dataset(
        cfg, synthetic_archive, split_file, ["SuiteB"], "test",
        clean_norm, protocol=dg, role="eval",
    )
    assert len(ds) > 0


def test_evaluation_on_undeclared_suite_is_rejected(dg):
    with pytest.raises(ProtocolViolation, match="undeclared suites"):
        dg.check_evaluation_suites(["SuiteC"])


def test_describe_is_readable(dg):
    text = dg.describe()
    assert "dg_strict" in text
    assert "forbidden" in text


def test_protocol_round_trips_through_dict(dg):
    d = dg.to_dict()
    assert d["source_suites"] == ["SuiteA"]
    assert d["target_suites"] == ["SuiteB"]
    assert d["allow_target_statistics"] is False
