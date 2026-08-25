"""Gradient reversal for adversarial domain-invariance.

Plan reference: sections 29, 32, 33.

DANN is a BASELINE here, not the contribution. Section 3.3 records that MIEST
already applies adversarial de-classification to CAMELS, so section 29 states
plainly: "This is a baseline, not the paper contribution."
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GradientReversal", "grad_reverse", "dann_schedule"]


class _GradientReversalFn(torch.autograd.Function):
    """Identity forward, negated-and-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        # view_as rather than returning x directly, so autograd records a node
        # even when this is the first operation applied to a leaf tensor.
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module wrapper so the reversal strength can be scheduled during training."""

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return grad_reverse(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


def dann_schedule(progress: float, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """Ramp the reversal strength from 0 to ``max_lambda`` over training.

    lambda = 2 / (1 + exp(-gamma * p)) - 1, the schedule from the original DANN
    paper. The ramp is not cosmetic: at initialisation the domain classifier is
    random, so reversing its gradient at full strength injects noise into the
    encoder before there is any domain signal worth removing, and training
    frequently collapses.
    """
    p = min(max(progress, 0.0), 1.0)
    return float(max_lambda * (2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))).item() - 1.0))
