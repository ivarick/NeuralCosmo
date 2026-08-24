"""Executable leakage guardrails.

Plan reference: sections 18, 19, 21, 67, 78.

Section 67 asks for an ``ExperimentProtocol`` object that declares what an
experiment is permitted to touch, and for the trainer to *reject* anything that
violates it. The point is stated plainly in the plan: "This turns
methodological discipline into executable code."

The failure this defends against is not carelessness in the moment. It is the
slow drift where, three weeks into debugging, someone normalizes with target
statistics "just to check", the run is good, and it quietly becomes the
reported result. A protocol violation here raises an exception instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = ["ProtocolViolation", "ExperimentProtocol", "default_dg_protocol"]


class ProtocolViolation(RuntimeError):
    """Raised when an experiment tries to use data its protocol forbids."""


@dataclass(frozen=True)
class ExperimentProtocol:
    """Declares which suites an experiment may use, and how.

    For the primary domain-generalization condition all three ``allow_target_*``
    flags are False, which is the default. Turning one on is an explicit,
    reviewable act that also changes how the experiment must be described in the
    paper (section 21: never mix the conditions silently).
    """

    source_suites: tuple[str, ...]
    target_suites: tuple[str, ...] = ()
    allow_target_unlabeled: bool = False
    allow_target_statistics: bool = False
    allow_target_labels: bool = False
    name: str = "dg_strict"
    notes: str = ""

    def __post_init__(self) -> None:
        overlap = set(self.source_suites) & set(self.target_suites)
        if overlap:
            raise ValueError(
                f"A suite cannot be both source and target: {sorted(overlap)}. "
                f"That would make the transfer measurement meaningless."
            )
        if not self.source_suites:
            raise ValueError("An experiment needs at least one source suite")

    # -- classification ----------------------------------------------------

    def is_target(self, suite: str) -> bool:
        return suite in self.target_suites

    def is_source(self, suite: str) -> bool:
        return suite in self.source_suites

    def known(self, suite: str) -> bool:
        return self.is_source(suite) or self.is_target(suite)

    # -- checks ------------------------------------------------------------

    def check_training_suites(self, suites: Iterable[str]) -> None:
        """Reject target-suite data appearing in a training set."""
        offending = sorted({s for s in suites if self.is_target(s)})
        if offending and not self.allow_target_unlabeled:
            raise ProtocolViolation(
                f"Protocol {self.name!r} forbids target-suite data during training, "
                f"but the training set contains: {offending}. "
                f"Set allow_target_unlabeled=True only if this experiment is being "
                f"reported as domain ADAPTATION rather than domain generalization "
                f"(plan sections 21, 67; Kill 6)."
            )

    def check_validation_suites(self, suites: Iterable[str]) -> None:
        """Reject target-suite data used for model selection.

        Section 19 is explicit that the sealed target may not influence
        architecture, augmentation, loss weights, learning rate, early stopping,
        representation dimension or alignment strength. Validation drives all of
        those, so a target suite appearing here is a violation even though no
        gradient flows from it.
        """
        offending = sorted({s for s in suites if self.is_target(s)})
        if offending and not self.allow_target_labels:
            raise ProtocolViolation(
                f"Protocol {self.name!r} forbids selecting models on target data, "
                f"but the validation set contains: {offending}. "
                f"Checkpoint selection and early stopping must use source "
                f"validation only (plan sections 19, 62, 63)."
            )

    def check_normalizer(self, provenance: str) -> None:
        """Reject normalization statistics derived from a target suite.

        The check is deliberately a substring scan over the provenance string
        rather than a structured field. It errs toward false positives, which is
        the correct direction: a spurious failure costs a minute, a missed leak
        costs the paper.
        """
        for suite in self.target_suites:
            if suite in provenance and not self.allow_target_statistics:
                raise ProtocolViolation(
                    f"Protocol {self.name!r} forbids target statistics, but the "
                    f"normalizer provenance names target suite {suite!r}:\n"
                    f"  {provenance}\n"
                    f"Under DG-strict, normalization must be computed from "
                    f"source-training simulations only and then applied unchanged "
                    f"to the target (plan section 20.2)."
                )

    def check_evaluation_suites(self, suites: Iterable[str]) -> None:
        """Warn-free check that evaluation suites are declared somewhere."""
        unknown = sorted({s for s in suites if not self.known(s)})
        if unknown:
            raise ProtocolViolation(
                f"Evaluation requested on undeclared suites: {unknown}. "
                f"Add them to source_suites or target_suites so the protocol "
                f"records what this experiment touched."
            )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_suites": list(self.source_suites),
            "target_suites": list(self.target_suites),
            "allow_target_unlabeled": self.allow_target_unlabeled,
            "allow_target_statistics": self.allow_target_statistics,
            "allow_target_labels": self.allow_target_labels,
            "notes": self.notes,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> ExperimentProtocol:
        """Build from the ``data`` and ``protocol`` blocks of an experiment config."""
        data = cfg.get("data", {})
        proto = cfg.get("protocol", {})
        return cls(
            source_suites=tuple(data.get("source_suites", ())),
            target_suites=tuple(data.get("target_suites", ())),
            allow_target_unlabeled=bool(proto.get("use_target_for_training", False)),
            allow_target_labels=bool(proto.get("use_target_for_validation", False)),
            allow_target_statistics=bool(proto.get("use_target_statistics", False)),
            name=str(proto.get("name", "dg_strict")),
            notes=str(proto.get("notes", "")),
        )

    def describe(self) -> str:
        flags = [
            f"target_unlabeled={'ALLOWED' if self.allow_target_unlabeled else 'forbidden'}",
            f"target_statistics={'ALLOWED' if self.allow_target_statistics else 'forbidden'}",
            f"target_labels={'ALLOWED' if self.allow_target_labels else 'forbidden'}",
        ]
        return (
            f"protocol={self.name} "
            f"sources={list(self.source_suites)} targets={list(self.target_suites)} "
            + " ".join(flags)
        )

    @property
    def is_strict_dg(self) -> bool:
        return not (
            self.allow_target_unlabeled or self.allow_target_statistics or self.allow_target_labels
        )


def default_dg_protocol(
    source_suites: Sequence[str],
    target_suites: Sequence[str],
) -> ExperimentProtocol:
    """The primary condition of section 18: nothing about the target is used."""
    return ExperimentProtocol(
        source_suites=tuple(source_suites),
        target_suites=tuple(target_suites),
        allow_target_unlabeled=False,
        allow_target_statistics=False,
        allow_target_labels=False,
        name="dg_strict",
    )
