"""Feature-distribution alignment losses.

Plan reference: sections 30, 31, 33, 47.

These are BASELINES, not contributions. Section 3.2 records that MMD alignment
on CAMELS is already published (DA-GNN), and section 31 says so explicitly.

IMPORTANT — what is aligned, and why it is not obviously right
-------------------------------------------------------------
Under domain generalization the target suite is unavailable, so these losses
align the SOURCE domains with each other: IllustrisTNG against SIMBA. The hope
is that a representation which cannot distinguish two simulators also fails to
distinguish a third.

Section H3 of the plan warns this may be actively harmful. Omega_m and sigma_8
themselves change the data distribution, so forcing two source marginals
together can destroy task-relevant structure -- if one suite happens to contain
more high-Omega_m simulations in a batch, aligning marginals suppresses exactly
the signal being predicted. Lower domain-probe accuracy is therefore not
automatically better, which is why every alignment result must be reported
alongside cosmological error (section 37).
"""

from __future__ import annotations

import torch

__all__ = ["coral_loss", "mmd_rbf_loss", "conditional_alignment_loss"]


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL: squared Frobenius distance between feature covariances (section 30).

    Second-order alignment only. It matches the shape of the two feature clouds
    while ignoring everything beyond their covariance, which makes it a useful
    lower rung on the alignment ladder rather than a strong method.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("CORAL expects (batch, features) tensors")
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"feature dimensions differ: {source.shape[1]} vs {target.shape[1]}"
        )
    ns, nt, d = source.shape[0], target.shape[0], source.shape[1]
    if ns < 2 or nt < 2:
        # A covariance needs at least two samples; return an exact zero that
        # still carries a gradient path so the graph stays intact.
        return source.sum() * 0.0

    def _cov(x: torch.Tensor) -> torch.Tensor:
        xm = x - x.mean(dim=0, keepdim=True)
        return xm.t() @ xm / (x.shape[0] - 1)

    diff = _cov(source) - _cov(target)
    # The 4d^2 normalisation is the standard Deep CORAL scaling, which keeps the
    # loss magnitude roughly independent of the latent dimension -- important
    # here because section 16 (Phase 16) sweeps that dimension.
    return (diff * diff).sum() / (4 * d * d)


def _pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b, p=2.0) ** 2


def mmd_rbf_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    bandwidths: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Maximum Mean Discrepancy with a mixture of Gaussian kernels (section 31).

    Bandwidths are multipliers on the median pairwise squared distance of the
    pooled batch. Using the median heuristic rather than fixed absolute scales
    matters because the latent norm drifts during training; a fixed bandwidth
    would silently change what the loss measures from epoch to epoch.
    """
    if source.shape[1] != target.shape[1]:
        raise ValueError("MMD requires matching feature dimensions")
    ns, nt = source.shape[0], target.shape[0]
    if ns < 2 or nt < 2:
        return source.sum() * 0.0

    pooled = torch.cat([source, target], dim=0)
    with torch.no_grad():
        d2 = _pairwise_sq_dists(pooled, pooled)
        median = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0, device=d2.device)
        median = torch.clamp(median, min=1e-8)

    d_ss = _pairwise_sq_dists(source, source)
    d_tt = _pairwise_sq_dists(target, target)
    d_st = _pairwise_sq_dists(source, target)

    total = source.sum() * 0.0
    for b in bandwidths:
        gamma = 1.0 / (2.0 * b * median)
        k_ss = torch.exp(-gamma * d_ss)
        k_tt = torch.exp(-gamma * d_tt)
        k_st = torch.exp(-gamma * d_st)
        # Unbiased estimator: exclude the diagonal self-similarities, which are
        # identically 1 and would otherwise bias the statistic upward.
        sum_ss = (k_ss.sum() - k_ss.diag().sum()) / (ns * (ns - 1))
        sum_tt = (k_tt.sum() - k_tt.diag().sum()) / (nt * (nt - 1))
        sum_st = k_st.mean()
        total = total + sum_ss + sum_tt - 2 * sum_st

    return total / len(bandwidths)


def conditional_alignment_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    domains: torch.Tensor,
    tau_y: float = 0.1,
) -> torch.Tensor:
    """Target-conditioned cross-domain alignment (section 47).

    Naive marginal alignment asks a TNG sample at Omega_m = 0.11 and a SIMBA
    sample at Omega_m = 0.48 to look alike merely because they come from
    different suites, which is wrong: they describe genuinely different
    universes. This instead weights each cross-domain pair by how close their
    cosmologies are,

        w_ij = exp(-||y_i - y_j||^2 / (2 tau_y^2))

    so only samples that *should* look alike are pushed together.

    This is a candidate mechanism, not a novelty claim (section 47).
    """
    if not (features.shape[0] == targets.shape[0] == domains.shape[0]):
        raise ValueError("features, targets and domains must have equal batch size")
    if tau_y <= 0:
        raise ValueError(f"tau_y must be positive, got {tau_y}")

    cross = domains[:, None] != domains[None, :]
    if not cross.any():
        return features.sum() * 0.0

    dy2 = torch.cdist(targets, targets, p=2.0) ** 2
    w = torch.exp(-dy2 / (2 * tau_y**2)) * cross.float()

    denom = w.sum()
    if denom <= 0:
        return features.sum() * 0.0

    dz2 = torch.cdist(features, features, p=2.0) ** 2
    return (w * dz2).sum() / denom
