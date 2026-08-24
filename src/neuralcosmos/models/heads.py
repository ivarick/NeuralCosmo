"""Prediction heads.

Plan reference: sections 43.2, 43.3, 44, 74.

Heads are kept separate from backbones so that every method in the comparison
(ERM, DANN, CORAL, MMD, paired) can share an identical encoder. Section 5 of
the plan requires this: if methods differ in backbone, a reported improvement
cannot be attributed to the method.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["RegressionHead", "ProjectionHead", "DomainHead"]


class RegressionHead(nn.Module):
    """Latent vector -> the cosmological targets.

    Outputs are unconstrained. Targets are range-normalised to roughly [0, 1]
    (section 22), but no sigmoid is applied: clamping the output range would
    make the model unable to express a confident error, which distorts the
    residual analysis of section 72 and hides exactly the regress-to-the-mean
    behaviour that analysis is meant to detect.
    """

    def __init__(
        self,
        latent_dim: int,
        n_targets: int = 2,
        hidden: int | None = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden:
            layers: list[nn.Module] = [nn.Linear(latent_dim, hidden), nn.SiLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden, n_targets))
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(latent_dim, n_targets)
        self.n_targets = n_targets

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ProjectionHead(nn.Module):
    """Latent vector -> alignment space (section 43.3).

    Alignment losses act here rather than on the regression representation
    directly, so that the representation used for prediction is not forced to
    satisfy every invariance constraint simultaneously.
    """

    def __init__(self, latent_dim: int, out_dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DomainHead(nn.Module):
    """Latent vector -> simulator logits.

    Used two ways, and the distinction matters:

    - as the adversary in DANN (section 29), trained through a gradient
      reversal layer;
    - as a FROZEN-encoder probe (section 37), trained post hoc to measure how
      much simulator information a representation retains.

    Section 37 also warns that a near-chance probe is not by itself evidence of
    a good representation, because a collapsed representation hides the domain
    too. Probe accuracy must always be reported alongside cosmological error.
    """

    def __init__(
        self,
        latent_dim: int,
        n_domains: int,
        hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, hidden), nn.SiLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, n_domains))
        self.net = nn.Sequential(*layers)
        self.n_domains = n_domains

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
