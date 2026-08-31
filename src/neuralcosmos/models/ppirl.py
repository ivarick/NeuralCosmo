r"""Paired-Physics Invariant Representation Learning model.

Plan reference: sections 43, 44, 45, 48, 49.

Working name only. The novelty audit (reports/novelty_audit.md) found the paired
hydro/N-body structure is NOT unexploited, so this is not claimed as novel until
the controls of sections 50 and 51 show correct pairing beating both shuffled
pairing and unpaired extra data. This module is the mechanism; the evidence is
separate.

Architecture (section 43):

    x_h --.
          |-- shared encoder f --> z_h --> regressor g --> y_h   (cosmology)
    x_n --'                    \-> z_n --> regressor g --> y_n   (optional)
                                 |
                    projection head q --> p_h, p_n --> pair loss

The same encoder processes both views because both are total-matter maps. A
separate projection head carries the alignment loss so the regression
representation is not forced to satisfy every consistency constraint directly
(section 43.3).

The full objective (section 48):

    L = L_reg,h + alpha L_reg,n + beta L_pair + var/cov guards

is assembled from components that are individually switchable, because section
49 requires building it up in stages (M0 hydro-only, M1 add N-body regression,
M2 add pair alignment) rather than launching the whole thing at once.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.paired import covariance_loss, pair_consistency_loss, variance_loss
from .erm import ERMModel
from .heads import ProjectionHead

__all__ = ["PPIRLModel", "build_ppirl_model"]


class PPIRLModel(nn.Module):
    """Shared encoder + regression head + projection head over paired views.

    ``forward`` is the inference path: predict cosmology from a single
    (hydrodynamic) map, so a trained PPIRL model is a drop-in regressor at test
    time and every existing evaluation and probe works unchanged.

    ``forward_pair`` is the training path: it takes both views and the scaled
    target and returns the hydro prediction plus a dict of weighted auxiliary
    losses, matching the contract the trainer already uses for the DG baselines.
    """

    def __init__(
        self,
        base: ERMModel,
        projection_dim: int = 128,
        projection_hidden: int = 256,
        nbody_reg_weight: float = 0.0,
        pair_weight: float = 1.0,
        var_weight: float = 1.0,
        cov_weight: float = 0.04,
        consistency: str = "mse",
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.latent_dim = base.latent_dim
        self.n_targets = base.n_targets

        self.use_projection = use_projection
        self.projector: nn.Module | None = (
            ProjectionHead(base.latent_dim, out_dim=projection_dim, hidden=projection_hidden)
            if use_projection
            else None
        )

        self.nbody_reg_weight = nbody_reg_weight
        self.pair_weight = pair_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.consistency = consistency

    # -- inference (hydro only) --------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)

    def forward_features(self, x: torch.Tensor):
        return self.base.forward_features(x)

    @property
    def backbone(self) -> nn.Module:
        return self.base.backbone

    @property
    def regressor(self) -> nn.Module:
        return self.base.regressor

    # -- training (paired) --------------------------------------------------

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        return self.projector(z) if self.projector is not None else z

    def forward_pair(
        self,
        x_hydro: torch.Tensor,
        x_nbody: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the hydro prediction and weighted auxiliary losses.

        ``target`` is the SCALED target the dataset produced. The trainer owns
        the hydro regression term (``mse(pred_h, target)``) exactly as for the
        other models; everything specific to the paired method is returned in
        the aux dict, already multiplied by its weight so the trainer only sums.
        """
        z_h = self.base.backbone(x_hydro)
        z_n = self.base.backbone(x_nbody)
        pred_h = self.base.regressor(z_h)

        aux: dict[str, torch.Tensor] = {}

        # Optional N-body regression (section 44 alpha term). The N-body view
        # shares the pair's cosmology, so its label is the same target.
        if self.nbody_reg_weight > 0:
            pred_n = self.base.regressor(z_n)
            aux["reg_nbody"] = self.nbody_reg_weight * F.mse_loss(pred_n, target)

        if self.pair_weight > 0 or self.var_weight > 0 or self.cov_weight > 0:
            p_h = self._project(z_h)
            p_n = self._project(z_n)

            if self.pair_weight > 0:
                aux["pair"] = self.pair_weight * pair_consistency_loss(
                    p_h, p_n, kind=self.consistency
                )
            if self.var_weight > 0:
                aux["var"] = self.var_weight * 0.5 * (variance_loss(p_h) + variance_loss(p_n))
            if self.cov_weight > 0:
                aux["cov"] = self.cov_weight * 0.5 * (
                    covariance_loss(p_h) + covariance_loss(p_n)
                )

        return pred_h, aux

    @property
    def is_paired(self) -> bool:
        return True

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_ppirl_model(cfg: dict[str, Any], n_targets: int = 2) -> PPIRLModel:
    """Construct a PPIRL model from an experiment config."""
    method_cfg = cfg.get("method", {})
    base = ERMModel.from_config(cfg, n_targets=n_targets)
    return PPIRLModel(
        base=base,
        projection_dim=int(method_cfg.get("projection_dim", 128)),
        projection_hidden=int(method_cfg.get("projection_hidden", 256)),
        nbody_reg_weight=float(method_cfg.get("nbody_reg_weight", 0.0)),
        pair_weight=float(method_cfg.get("pair_weight", 1.0)),
        var_weight=float(method_cfg.get("var_weight", 1.0)),
        cov_weight=float(method_cfg.get("cov_weight", 0.04)),
        consistency=str(method_cfg.get("consistency", "mse")),
        use_projection=bool(method_cfg.get("use_projection", True)),
    )
