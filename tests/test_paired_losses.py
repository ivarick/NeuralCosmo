"""Pair-consistency and anti-collapse losses.

Plan reference: sections 45, 46.

The tests that matter most are the collapse tests: section 45 warns that pair
consistency alone is satisfied by a constant, so the variance and covariance
terms must actually make a collapsed representation costly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neuralcosmos.losses.paired import (  # noqa: E402
    covariance_loss,
    pair_consistency_loss,
    variance_loss,
    vicreg_loss,
)


# --------------------------------------------------------------------------
# Consistency
# --------------------------------------------------------------------------


def test_cosine_consistency_is_zero_for_identical_embeddings():
    z = torch.randn(16, 8)
    assert pair_consistency_loss(z, z.clone(), kind="cosine").item() == pytest.approx(0.0, abs=1e-6)


def test_cosine_consistency_is_positive_for_different_embeddings():
    torch.manual_seed(0)
    a, b = torch.randn(16, 8), torch.randn(16, 8)
    assert pair_consistency_loss(a, b, kind="cosine").item() > 0.1


def test_cosine_is_blind_to_scale_but_mse_is_not():
    """A key difference: cosine ignores a scale mismatch, MSE penalises it."""
    z = torch.randn(16, 8)
    scaled = z * 5.0
    assert pair_consistency_loss(z, scaled, kind="cosine").item() == pytest.approx(0.0, abs=1e-5)
    assert pair_consistency_loss(z, scaled, kind="mse").item() > 1.0


def test_consistency_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="differ in shape"):
        pair_consistency_loss(torch.randn(4, 8), torch.randn(4, 6))


def test_consistency_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown consistency kind"):
        pair_consistency_loss(torch.randn(4, 8), torch.randn(4, 8), kind="l1")


# --------------------------------------------------------------------------
# The collapse guards (section 46)
# --------------------------------------------------------------------------


def test_variance_loss_punishes_a_collapsed_batch():
    """Every row identical is the collapse the pair loss would otherwise invite."""
    collapsed = torch.ones(32, 8)
    varied = torch.randn(32, 8) * 3.0
    assert variance_loss(collapsed).item() > 0.5
    assert variance_loss(varied).item() < variance_loss(collapsed).item()


def test_variance_loss_is_near_zero_for_healthy_variance():
    z = torch.randn(256, 8) * 2.0        # sd ~2 > gamma=1
    assert variance_loss(z, gamma=1.0).item() == pytest.approx(0.0, abs=0.05)


def test_variance_loss_hinges_at_gamma():
    """A dimension with sd exactly gamma contributes nothing; below, it does."""
    low = torch.randn(512, 4) * 0.2      # sd well below gamma=1
    assert variance_loss(low, gamma=1.0).item() > 0.5


def test_covariance_loss_is_zero_for_decorrelated_dimensions():
    torch.manual_seed(0)
    z = torch.randn(4096, 8)             # independent columns
    assert covariance_loss(z).item() < 0.05


def test_covariance_loss_punishes_redundant_dimensions():
    base = torch.randn(256, 1)
    redundant = base.repeat(1, 8) + 0.01 * torch.randn(256, 8)
    assert covariance_loss(redundant).item() > covariance_loss(torch.randn(256, 8)).item()


# --------------------------------------------------------------------------
# The reason the guards exist: consistency alone rewards collapse
# --------------------------------------------------------------------------


def test_collapse_minimises_consistency_but_not_the_full_objective():
    """The central point of section 45, made concrete.

    A constant representation drives pair consistency to zero -- it looks
    perfect -- but the VICReg variance term makes the full objective large, so
    the guarded loss does not reward the degenerate solution.
    """
    collapsed = torch.ones(32, 8)

    consistency_only = pair_consistency_loss(collapsed, collapsed.clone(), kind="mse")
    assert consistency_only.item() == pytest.approx(0.0, abs=1e-6)

    full = vicreg_loss(collapsed, collapsed.clone())
    assert full["invariance"].item() == pytest.approx(0.0, abs=1e-6)
    assert full["variance"].item() > 0.5
    assert full["total"].item() > 1.0


def test_vicreg_returns_all_components():
    torch.manual_seed(0)
    a, b = torch.randn(32, 8), torch.randn(32, 8)
    out = vicreg_loss(a, b)
    for key in ("total", "invariance", "variance", "covariance"):
        assert key in out and torch.isfinite(out[key])


def test_vicreg_is_differentiable_to_both_views():
    a = torch.randn(32, 8, requires_grad=True)
    b = torch.randn(32, 8, requires_grad=True)
    vicreg_loss(a, b)["total"].backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_vicreg_weights_scale_the_components():
    torch.manual_seed(0)
    a, b = torch.randn(32, 8), torch.randn(32, 8)
    base = vicreg_loss(a, b, inv_weight=1.0, var_weight=0.0, cov_weight=0.0)
    inv_only = base["total"]
    assert inv_only.item() == pytest.approx(base["invariance"].item(), rel=1e-5)


def test_degenerate_single_sample_batch_is_safe():
    a = torch.randn(1, 8)
    out = vicreg_loss(a, a.clone())
    assert torch.isfinite(out["total"])
