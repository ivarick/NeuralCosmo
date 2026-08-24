"""A compact convolutional encoder.

Plan reference: sections 26, 60.

Purpose (section 26): verify the data pipeline, establish an inexpensive
baseline, make debugging fast, and support repeated experiments on a 12 GB GPU.
Explicitly NOT to be impressive. Section 60 is blunt that research value comes
from experiment quality, not parameter count.

The encoder maps a single-channel 256x256 field to a latent vector of
configurable width, and is deliberately separated from the regression head so
that the same backbone can be shared by ERM, DANN, CORAL, MMD and the paired
method (section 5 of the plan requires identical backbones across methods, so
that architecture size cannot masquerade as a method improvement).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SmallCNN", "ConvBlock"]


class ConvBlock(nn.Module):
    """Conv -> Norm -> activation, twice, then downsample.

    GroupNorm rather than BatchNorm. Three reasons specific to this project:

    1. Multi-source batches. Section 61 requires batches balanced across
       suites; BatchNorm would mix statistics across domains inside the
       normalisation itself, entangling the very thing the method is trying to
       separate and making domain-invariance results hard to attribute.
    2. Evaluation on an unseen suite. BatchNorm's running statistics are
       estimated on source data and silently become a form of domain
       adaptation when the target distribution differs.
    3. Small batches. Gradient accumulation on a 12 GB card can push the
       per-step batch low, where BatchNorm estimates get noisy.
    """

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__()
        g_in = min(groups, out_ch)
        while out_ch % g_in != 0 and g_in > 1:
            g_in -= 1

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g_in, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g_in, out_ch),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.block(x))


class SmallCNN(nn.Module):
    """Compact encoder: (B, 1, 256, 256) -> (B, latent_dim).

    Average pooling rather than max pooling throughout. These are projected
    matter-density fields, where the physically meaningful aggregation of a
    2x2 patch is its mean surface density; max pooling would discard mass and
    keep only peaks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        width: int = 32,
        depth: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")

        channels = [in_channels] + [min(width * 2**i, 512) for i in range(depth)]
        self.blocks = nn.Sequential(
            *[ConvBlock(channels[i], channels[i + 1]) for i in range(depth)]
        )
        # Global average pooling makes the encoder resolution-agnostic, which
        # the resolution-robustness study of section 14 (Phase 14) needs.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(channels[-1], latent_dim),
        )

        self.latent_dim = latent_dim
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channel(s), got {x.shape[1]}"
            )
        return self.head(self.pool(self.blocks(x)))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
