"""Domain-generalization baselines sharing one backbone.

Plan reference: sections 29, 30, 31, 32, 33, 34 (gate B3), 68 (Phase 5).

Phase 5 requires ERM, DANN, CORAL, MMD and a MIEST-comparable baseline to use
"exactly the same backbone where possible, split, source data, evaluation metric
and training budget", so that a reported improvement cannot be an architecture
difference wearing a method's name.

That constraint is enforced structurally here: every method is the same
``ERMModel`` plus an auxiliary loss computed from the same latent vector. Only
DANN adds parameters, and only a small classifier head.

None of these is the contribution. MIEST (adversarial de-classification) and
DA-GNN (MMD) are published on CAMELS already.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.alignment import coral_loss, mmd_rbf_loss
from ..losses.domain import dann_schedule, grad_reverse
from .erm import ERMModel
from .heads import DomainHead

__all__ = ["METHODS", "DGModel", "build_dg_model"]

METHODS = ("erm", "dann", "coral", "mmd", "miest_like")


def _build_domain_lut(domain_ids: Sequence[int] | None) -> torch.Tensor:
    """Lookup table mapping global suite ids onto contiguous local classes.

    An empty tensor means "identity": the caller did not declare which global
    ids are in play, so the labels are used as given.
    """
    if not domain_ids:
        return torch.zeros(0, dtype=torch.long)
    ids = sorted(int(d) for d in domain_ids)
    lut = torch.full((max(ids) + 1,), -1, dtype=torch.long)
    for local, global_id in enumerate(ids):
        lut[global_id] = local
    return lut


def _split_by_domain(z: torch.Tensor, domains: torch.Tensor) -> list[torch.Tensor]:
    """Group a batch's latents by suite id."""
    return [z[domains == d] for d in torch.unique(domains)]


def _pairwise_alignment(
    z: torch.Tensor,
    domains: torch.Tensor,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Average an alignment loss over every unordered pair of source domains.

    With only two sources this is a single term. Written for the general case
    because section 53's rotated evaluations use different source pairs and a
    three-source variant is a natural extension.
    """
    groups = [g for g in _split_by_domain(z, domains) if g.shape[0] >= 2]
    if len(groups) < 2:
        return z.sum() * 0.0
    total = z.sum() * 0.0
    n = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total = total + fn(groups[i], groups[j])
            n += 1
    return total / max(n, 1)


class DGModel(nn.Module):
    """An ERM model plus a method-specific auxiliary objective.

    ``forward_train`` returns the prediction, the latent, and a dict of scalar
    auxiliary losses. The trainer decides how to weight them, so the weighting
    stays visible in the config rather than buried in the model.
    """

    def __init__(
        self,
        base: ERMModel,
        method: str = "erm",
        n_domains: int = 2,
        alignment_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        grl_gamma: float = 10.0,
        bottleneck_weight: float = 0.0,
        domain_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")

        # ``suite_id`` is a GLOBAL index over every suite in the data config,
        # assigned in sorted order: Astrid=0, IllustrisTNG=1, SIMBA=2. Training
        # on IllustrisTNG+SIMBA therefore yields labels {1, 2}, which a
        # two-class head cannot accept -- cross_entropy fails inside a CUDA
        # kernel with an assertion naming neither the cause nor the fix.
        # Remap the global ids onto contiguous local classes 0..n-1.
        self.register_buffer("_domain_lut", _build_domain_lut(domain_ids), persistent=False)

        self.base = base
        self.method = method
        self.alignment_weight = alignment_weight
        self.adversarial_weight = adversarial_weight
        self.grl_gamma = grl_gamma
        self.bottleneck_weight = bottleneck_weight
        self.latent_dim = base.latent_dim
        self.n_targets = base.n_targets

        self.domain_head: nn.Module | None = None
        if method in ("dann", "miest_like"):
            self.domain_head = DomainHead(base.latent_dim, n_domains=n_domains)

    # -- inference ---------------------------------------------------------

    @property
    def backbone(self) -> nn.Module:
        """Expose the encoder directly.

        Everything that probes a representation -- section 37's domain probe,
        section 38's target probe, section 39's latent geometry -- reaches for
        ``model.backbone``. Without this a DG model is not a drop-in for an ERM
        model and every diagnostic needs a special case.
        """
        return self.base.backbone

    @property
    def regressor(self) -> nn.Module:
        return self.base.regressor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def forward_features(self, x: torch.Tensor):
        return self.base.forward_features(x)

    # -- training ----------------------------------------------------------

    def forward_train(
        self,
        x: torch.Tensor,
        domains: torch.Tensor,
        progress: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        pred, z = self.base.forward_features(x)
        aux: dict[str, torch.Tensor] = {}

        if self.method == "coral":
            aux["align"] = self.alignment_weight * _pairwise_alignment(z, domains, coral_loss)

        elif self.method == "mmd":
            aux["align"] = self.alignment_weight * _pairwise_alignment(z, domains, mmd_rbf_loss)

        elif self.method in ("dann", "miest_like"):
            domains = self._to_local_domains(domains)
            # The reversal strength is ramped: at initialisation the domain head
            # is random, so reversing its gradient at full strength injects pure
            # noise into the encoder before any domain signal exists to remove.
            lambd = dann_schedule(progress, gamma=self.grl_gamma)
            logits = self.domain_head(grad_reverse(z, lambd))
            aux["domain"] = self.adversarial_weight * F.cross_entropy(logits, domains)
            aux["_lambda"] = torch.tensor(lambd, device=z.device)

            if self.method == "miest_like" and self.bottleneck_weight > 0:
                # MIEST couples de-classification with an information bottleneck.
                # This is a comparable stand-in, not a reproduction: MIEST works
                # on HI maps with a different architecture, and section 32
                # requires those differences to be documented rather than
                # papered over.
                aux["bottleneck"] = self.bottleneck_weight * z.pow(2).mean()

        return pred, z, aux

    def _to_local_domains(self, domains: torch.Tensor) -> torch.Tensor:
        """Map global suite ids to contiguous classes the domain head accepts."""
        if self._domain_lut.numel() == 0:
            return domains
        if int(domains.max()) >= self._domain_lut.numel():
            raise ValueError(
                f"suite id {int(domains.max())} is outside the declared domains; "
                f"pass domain_ids covering every suite present in the batch."
            )
        local = self._domain_lut.to(domains.device)[domains]
        if bool((local < 0).any()):
            missing = sorted({int(d) for d, m in zip(domains.tolist(), (local < 0).tolist()) if m})
            raise ValueError(
                f"batch contains undeclared suite ids {missing}; the domain head "
                f"was built for a different set of source suites."
            )
        return local

    @torch.no_grad()
    def domain_accuracy(self, z: torch.Tensor, domains: torch.Tensor) -> float:
        """Training-time diagnostic: how well the adversary is doing."""
        if self.domain_head is None:
            return float("nan")
        pred = self.domain_head(z).argmax(dim=1)
        return float((pred == self._to_local_domains(domains)).float().mean().item())

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_dg_model(
    cfg: dict[str, Any],
    n_targets: int = 2,
    n_domains: int = 2,
    domain_ids: Sequence[int] | None = None,
) -> DGModel:
    """Construct a DG baseline from an experiment config.

    ``domain_ids`` are the GLOBAL suite ids of the source suites, which are not
    generally 0..n-1: they index every suite in the data config, sealed target
    included. Passing them lets the model remap onto the classes its head has.
    """
    method_cfg = cfg.get("method", {})
    method = str(method_cfg.get("name", "erm"))
    base = ERMModel.from_config(cfg, n_targets=n_targets)
    return DGModel(
        base=base,
        method=method,
        n_domains=n_domains,
        domain_ids=domain_ids,
        alignment_weight=float(method_cfg.get("alignment_weight", 1.0)),
        adversarial_weight=float(method_cfg.get("adversarial_weight", 1.0)),
        grl_gamma=float(method_cfg.get("grl_gamma", 10.0)),
        bottleneck_weight=float(method_cfg.get("bottleneck_weight", 0.0)),
    )
