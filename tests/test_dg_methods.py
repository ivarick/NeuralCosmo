"""Domain-generalization baselines.

Plan reference: sections 29, 30, 31, 32, 34 (gate B3), 66.5.

Section 66.5 asks specifically for "domain head gradient behavior correct for
adversarial baseline", which is the one part of DANN that fails silently: if the
reversal is dropped, the encoder simply *helps* the domain classifier and the
method becomes the opposite of what it claims, while the loss curve still looks
entirely reasonable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neuralcosmos.losses.alignment import (  # noqa: E402
    conditional_alignment_loss,
    coral_loss,
    mmd_rbf_loss,
)
from neuralcosmos.losses.domain import dann_schedule, grad_reverse  # noqa: E402
from neuralcosmos.models.backbones.small_cnn import SmallCNN  # noqa: E402
from neuralcosmos.models.dg_methods import METHODS, DGModel, build_dg_model  # noqa: E402
from neuralcosmos.models.erm import ERMModel  # noqa: E402


def _base(latent: int = 16):
    return ERMModel(backbone=SmallCNN(latent_dim=latent, width=4, depth=2))


# --------------------------------------------------------------------------
# Gradient reversal
# --------------------------------------------------------------------------


def test_grad_reverse_is_identity_forward():
    x = torch.randn(4, 8)
    assert torch.allclose(grad_reverse(x, 1.0), x)


def test_grad_reverse_negates_the_gradient():
    """The defining property. Without it DANN silently becomes its own opposite."""
    x = torch.randn(4, 8, requires_grad=True)
    grad_reverse(x, 1.0).sum().backward()
    assert torch.allclose(x.grad, -torch.ones_like(x))


def test_grad_reverse_scales_by_lambda():
    x = torch.randn(3, 5, requires_grad=True)
    grad_reverse(x, 0.25).sum().backward()
    assert torch.allclose(x.grad, -0.25 * torch.ones_like(x))


def test_grad_reverse_at_zero_blocks_the_gradient():
    x = torch.randn(3, 5, requires_grad=True)
    grad_reverse(x, 0.0).sum().backward()
    assert torch.allclose(x.grad, torch.zeros_like(x))


def test_dann_schedule_ramps_from_zero():
    assert dann_schedule(0.0) == pytest.approx(0.0, abs=1e-6)
    assert dann_schedule(1.0) > 0.99
    values = [dann_schedule(p) for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(b >= a for a, b in zip(values, values[1:])), "schedule must be monotonic"


# --------------------------------------------------------------------------
# CORAL
# --------------------------------------------------------------------------


def test_coral_is_zero_for_identical_batches():
    x = torch.randn(64, 12)
    assert coral_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-10)


def test_coral_is_positive_for_different_covariances():
    torch.manual_seed(0)
    a = torch.randn(128, 8)
    b = torch.randn(128, 8) * 4.0
    assert coral_loss(a, b).item() > 0.01


def test_coral_ignores_a_pure_mean_shift():
    """CORAL is second order only; a mean shift leaves covariance untouched."""
    torch.manual_seed(0)
    a = torch.randn(256, 6)
    b = a + 10.0
    assert coral_loss(a, b).item() == pytest.approx(0.0, abs=1e-6)


def test_coral_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="feature dimensions differ"):
        coral_loss(torch.randn(8, 4), torch.randn(8, 6))


def test_coral_handles_degenerate_batch_without_nan():
    out = coral_loss(torch.randn(1, 5), torch.randn(8, 5))
    assert torch.isfinite(out) and out.item() == 0.0


# --------------------------------------------------------------------------
# MMD
# --------------------------------------------------------------------------


def test_mmd_is_near_zero_for_same_distribution():
    torch.manual_seed(0)
    a, b = torch.randn(128, 8), torch.randn(128, 8)
    assert abs(mmd_rbf_loss(a, b).item()) < 0.05


def test_mmd_is_positive_for_shifted_distributions():
    torch.manual_seed(0)
    a = torch.randn(128, 8)
    b = torch.randn(128, 8) + 5.0
    assert mmd_rbf_loss(a, b).item() > 0.1


def test_mmd_detects_a_mean_shift_that_coral_misses():
    """MMD sees the full distribution, not only its second moment."""
    torch.manual_seed(0)
    a = torch.randn(256, 6)
    b = a + 3.0
    assert coral_loss(a, b).item() < 1e-5
    assert mmd_rbf_loss(a, b).item() > 0.1


def test_mmd_is_finite_and_differentiable():
    a = torch.randn(32, 8, requires_grad=True)
    b = torch.randn(32, 8)
    loss = mmd_rbf_loss(a, b)
    loss.backward()
    assert torch.isfinite(loss)
    assert a.grad is not None and torch.isfinite(a.grad).all()


# --------------------------------------------------------------------------
# Conditional alignment (section 47)
# --------------------------------------------------------------------------


def test_conditional_alignment_ignores_same_domain_pairs():
    z = torch.randn(8, 4)
    y = torch.rand(8, 2)
    same = torch.zeros(8, dtype=torch.long)
    assert conditional_alignment_loss(z, y, same).item() == 0.0


def test_conditional_alignment_weights_by_target_proximity():
    """Cross-domain samples with similar cosmology should dominate the loss."""
    z = torch.tensor([[0.0, 0.0], [5.0, 0.0], [0.0, 0.0], [5.0, 0.0]])
    d = torch.tensor([0, 0, 1, 1])

    # Case A: the matching pair across domains has identical targets.
    y_close = torch.tensor([[0.5, 0.5], [0.9, 0.9], [0.5, 0.5], [0.9, 0.9]])
    # Case B: targets are arranged so the aligned pairs are far apart.
    y_far = torch.tensor([[0.1, 0.1], [0.9, 0.9], [0.9, 0.9], [0.1, 0.1]])

    a = conditional_alignment_loss(z, y_close, d, tau_y=0.1).item()
    b = conditional_alignment_loss(z, y_far, d, tau_y=0.1).item()
    assert a < b, "loss should be lower when similar-cosmology pairs already agree"


def test_conditional_alignment_rejects_bad_tau():
    with pytest.raises(ValueError, match="tau_y must be positive"):
        conditional_alignment_loss(torch.randn(4, 2), torch.rand(4, 2),
                                   torch.tensor([0, 0, 1, 1]), tau_y=0.0)


# --------------------------------------------------------------------------
# DGModel
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_every_method_produces_finite_losses(method):
    model = DGModel(_base(), method=method, bottleneck_weight=0.01)
    x = torch.randn(8, 1, 32, 32)
    d = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    pred, z, aux = model.forward_train(x, d, progress=0.5)
    assert pred.shape == (8, 2)
    assert z.shape == (8, 16)
    for name, value in aux.items():
        assert torch.isfinite(value), f"{method}: {name} is not finite"


@pytest.mark.parametrize("method", ["dann", "coral", "mmd", "miest_like"])
def test_auxiliary_loss_reaches_the_encoder(method):
    """If only the head receives gradient, the method does nothing to the encoder."""
    model = DGModel(_base(), method=method, bottleneck_weight=0.01)
    x = torch.randn(8, 1, 32, 32)
    d = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    _, _, aux = model.forward_train(x, d, progress=1.0)
    total = sum(v for k, v in aux.items() if not k.startswith("_"))
    total.backward()

    grads = [p.grad for p in model.base.backbone.parameters() if p.grad is not None]
    assert grads, f"{method}: no encoder parameter received gradient"
    assert any(g.abs().sum() > 0 for g in grads), f"{method}: encoder gradient is all zero"


def test_erm_produces_no_auxiliary_loss():
    model = DGModel(_base(), method="erm")
    _, _, aux = model.forward_train(
        torch.randn(4, 1, 32, 32), torch.tensor([0, 0, 1, 1])
    )
    assert aux == {}


def test_only_adversarial_methods_add_parameters():
    erm = DGModel(_base(), method="erm").n_parameters
    coral = DGModel(_base(), method="coral").n_parameters
    dann = DGModel(_base(), method="dann").n_parameters

    # Section 5 requires identical backbones so architecture cannot masquerade
    # as method. Alignment losses are parameter-free; DANN adds only a head.
    assert erm == coral
    assert dann > erm


def test_dann_lambda_is_reported_and_ramps():
    model = DGModel(_base(), method="dann")
    x = torch.randn(4, 1, 32, 32)
    d = torch.tensor([0, 0, 1, 1])

    _, _, early = model.forward_train(x, d, progress=0.0)
    _, _, late = model.forward_train(x, d, progress=1.0)
    assert early["_lambda"].item() < late["_lambda"].item()
    assert early["_lambda"].item() == pytest.approx(0.0, abs=1e-6)


def test_single_domain_batch_gives_zero_alignment():
    """A batch that happens to hold one suite must not produce a spurious loss."""
    for method in ("coral", "mmd"):
        model = DGModel(_base(), method=method)
        _, _, aux = model.forward_train(
            torch.randn(6, 1, 32, 32), torch.zeros(6, dtype=torch.long)
        )
        assert aux["align"].item() == 0.0


def test_inference_path_matches_the_base_model():
    base = _base()
    model = DGModel(base, method="dann")
    model.eval()
    x = torch.randn(3, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), base(x))


def test_build_from_config_selects_the_method():
    cfg = {
        "model": {"type": "small_cnn", "latent_dim": 16, "width": 4, "depth": 2},
        "method": {"name": "mmd", "alignment_weight": 0.5},
    }
    m = build_dg_model(cfg)
    assert m.method == "mmd"
    assert m.alignment_weight == 0.5


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        DGModel(_base(), method="not_a_method")
