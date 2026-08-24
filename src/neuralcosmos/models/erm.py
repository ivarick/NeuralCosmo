"""Empirical risk minimisation baseline.

Plan reference: sections 24, 26, 33, 44.

The plainest possible model: encoder, then regression head. Everything else in
the comparison is this plus a term, which is what makes the ablation table of
section 17 (Phase 17) interpretable.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .backbones.small_cnn import SmallCNN
from .heads import RegressionHead

__all__ = ["ERMModel", "build_backbone"]


def build_backbone(cfg: dict[str, Any]) -> nn.Module:
    """Construct an encoder from a model config block."""
    kind = cfg.get("type", "small_cnn")
    if kind == "small_cnn":
        return SmallCNN(
            in_channels=int(cfg.get("in_channels", 1)),
            latent_dim=int(cfg.get("latent_dim", 256)),
            width=int(cfg.get("width", 32)),
            depth=int(cfg.get("depth", 5)),
            dropout=float(cfg.get("encoder_dropout", 0.0)),
        )
    raise NotImplementedError(
        f"Unknown backbone {kind!r}. Implemented: small_cnn. "
        f"resnet and the CMD benchmark CNN are sections 27-28 and not yet built."
    )


class ERMModel(nn.Module):
    """Encoder + regression head.

    ``forward`` returns predictions; ``forward_features`` additionally returns
    the latent vector, which the domain probe and representation analyses of
    sections 37-39 consume without needing a second forward pass.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_targets: int = 2,
        head_hidden: int | None = 128,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        latent_dim = getattr(backbone, "latent_dim", None)
        if latent_dim is None:
            raise AttributeError("backbone must expose a latent_dim attribute")
        self.regressor = RegressionHead(
            latent_dim=latent_dim,
            n_targets=n_targets,
            hidden=head_hidden,
            dropout=head_dropout,
        )
        self.latent_dim = latent_dim
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.backbone(x))

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.backbone(x)
        return self.regressor(z), z

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: dict[str, Any], n_targets: int = 2) -> ERMModel:
        model_cfg = cfg.get("model", cfg)
        return cls(
            backbone=build_backbone(model_cfg),
            n_targets=n_targets,
            head_hidden=model_cfg.get("head_hidden", 128),
            head_dropout=float(model_cfg.get("head_dropout", 0.0)),
        )
