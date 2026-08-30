"""Pair-consistency and anti-collapse losses for paired-physics learning.

Plan reference: sections 45, 46.

The idea (section 45): for a matched hydro/N-body pair, the parts of the
representation used to infer cosmology should agree, because both views encode
the same cosmology and the same large-scale structure. The simplest expression
is a cosine or MSE consistency between projected embeddings.

The danger (section 45): consistency alone is trivially satisfied by a constant.
A network can drive the pair loss to zero by mapping every input to the same
vector, which is perfectly consistent and perfectly useless. Section 45 is
explicit that pair cosine loss alone must NOT be assumed sufficient, and section
46 points at VICReg-style variance and covariance terms as one collapse-resistant
option. Both are provided here so the method can be built with the guard from the
start rather than discovering the collapse empirically.

None of this is claimed as novel. VICReg is cited; the contribution, if any, is
the paired-physics *use* of these components, and that only survives the controls
of sections 50 and 51.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "pair_consistency_loss",
    "variance_loss",
    "covariance_loss",
    "vicreg_loss",
]


def pair_consistency_loss(
    p_hydro: torch.Tensor,
    p_nbody: torch.Tensor,
    kind: str = "cosine",
) -> torch.Tensor:
    """Invariance term: agreement between the two views of a pair (section 45).

    ``cosine`` returns ``1 - cos(p_h, p_n)`` averaged over the batch, in [0, 2].
    ``mse`` returns the mean squared difference, which unlike cosine also
    penalises a scale mismatch between the two embeddings.
    """
    if p_hydro.shape != p_nbody.shape:
        raise ValueError(f"pair embeddings differ in shape: {p_hydro.shape} vs {p_nbody.shape}")

    if kind == "cosine":
        return (1.0 - F.cosine_similarity(p_hydro, p_nbody, dim=-1)).mean()
    if kind == "mse":
        return F.mse_loss(p_hydro, p_nbody)
    raise ValueError(f"unknown consistency kind {kind!r}; expected 'cosine' or 'mse'")


def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance term: keep every latent dimension from collapsing.

    A hinge on the per-dimension standard deviation. As long as each dimension
    varies by at least ``gamma`` across the batch it contributes nothing; a
    dimension that collapses toward a constant is pushed back up. This is the
    term that makes the pair-consistency loss safe to minimise.
    """
    if z.shape[0] < 2:
        return z.sum() * 0.0
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """VICReg covariance term: discourage redundancy across dimensions.

    Penalises the squared off-diagonal covariances, pushing the dimensions to
    carry non-redundant information. Together with the variance term this makes
    a collapsed or degenerate representation costly rather than free.
    """
    n, d = z.shape
    if n < 2:
        return z.sum() * 0.0
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / d


def vicreg_loss(
    p_hydro: torch.Tensor,
    p_nbody: torch.Tensor,
    inv_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    consistency: str = "mse",
) -> dict[str, torch.Tensor]:
    """Full VICReg-style objective over a matched pair (sections 45, 46).

    Returns the components separately as well as the weighted total, so the
    trainer can log each and so an ablation can zero any one of them. The
    default weights are VICReg's published values; they are starting points,
    not tuned optima (section 48).
    """
    inv = pair_consistency_loss(p_hydro, p_nbody, kind=consistency)
    var = 0.5 * (variance_loss(p_hydro) + variance_loss(p_nbody))
    cov = 0.5 * (covariance_loss(p_hydro) + covariance_loss(p_nbody))
    total = inv_weight * inv + var_weight * var + cov_weight * cov
    return {"total": total, "invariance": inv, "variance": var, "covariance": cov}
