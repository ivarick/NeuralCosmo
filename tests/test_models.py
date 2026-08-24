"""Model contracts.

Plan reference: sections 26, 60, 66.5.

Section 66.5 asks for: output shape (batch, 2), finite loss, gradient reaching
the encoder, and correct domain-head gradient behaviour for the adversarial
baseline. The last is deferred until DANN exists; the rest are here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from neuralcosmos.models.backbones.small_cnn import SmallCNN  # noqa: E402
from neuralcosmos.models.erm import ERMModel, build_backbone  # noqa: E402
from neuralcosmos.models.heads import DomainHead, ProjectionHead, RegressionHead  # noqa: E402


@pytest.fixture
def model():
    # A narrow, shallow configuration so tests stay fast on CPU.
    return ERMModel(backbone=SmallCNN(latent_dim=32, width=4, depth=3))


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


def test_output_shape_is_batch_by_two(model):
    y = model(torch.randn(3, 1, 64, 64))
    assert y.shape == (3, 2)


def test_forward_features_returns_prediction_and_latent(model):
    y, z = model.forward_features(torch.randn(2, 1, 64, 64))
    assert y.shape == (2, 2)
    assert z.shape == (2, 32)


@pytest.mark.parametrize("size", [32, 64, 128])
def test_encoder_is_resolution_agnostic(size):
    """Global average pooling must let the resolution study of Phase 14 work."""
    enc = SmallCNN(latent_dim=16, width=4, depth=3)
    assert enc(torch.randn(2, 1, size, size)).shape == (2, 16)


def test_backbone_rejects_wrong_rank():
    enc = SmallCNN(latent_dim=8, width=4, depth=2)
    with pytest.raises(ValueError, match=r"expected \(B, C, H, W\)"):
        enc(torch.randn(2, 64, 64))


def test_backbone_rejects_wrong_channel_count():
    enc = SmallCNN(in_channels=1, latent_dim=8, width=4, depth=2)
    with pytest.raises(ValueError, match="input channel"):
        enc(torch.randn(2, 3, 64, 64))


def test_depth_must_be_positive():
    with pytest.raises(ValueError, match="depth must be at least 1"):
        SmallCNN(depth=0)


# --------------------------------------------------------------------------
# Gradients
# --------------------------------------------------------------------------


def test_loss_is_finite(model):
    pred = model(torch.randn(4, 1, 64, 64))
    loss = torch.nn.functional.mse_loss(pred, torch.rand(4, 2))
    assert torch.isfinite(loss)


def test_gradient_reaches_the_first_encoder_layer(model):
    """A head that trains while the encoder is detached is a silent failure."""
    pred = model(torch.randn(4, 1, 64, 64))
    torch.nn.functional.mse_loss(pred, torch.rand(4, 2)).backward()

    first_conv = next(
        p for name, p in model.backbone.named_parameters() if name.endswith("weight")
    )
    assert first_conv.grad is not None
    assert torch.isfinite(first_conv.grad).all()
    assert first_conv.grad.abs().sum() > 0, "no gradient signal reached the encoder"


def test_every_parameter_receives_gradient(model):
    pred = model(torch.randn(4, 1, 64, 64))
    torch.nn.functional.mse_loss(pred, torch.rand(4, 2)).backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert missing == [], f"parameters with no gradient: {missing}"


def test_a_training_step_reduces_loss_on_a_fixed_batch(model):
    """The cheapest possible check that the thing can learn at all."""
    torch.manual_seed(0)
    x = torch.randn(8, 1, 64, 64)
    y = torch.rand(8, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    first = torch.nn.functional.mse_loss(model(x), y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first


# --------------------------------------------------------------------------
# Determinism and normalisation choice
# --------------------------------------------------------------------------


def test_eval_mode_is_deterministic(model):
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


def test_no_batchnorm_anywhere(model):
    """GroupNorm is a deliberate choice, not an accident.

    BatchNorm would mix statistics across suites inside multi-source batches
    (section 61) and would turn evaluation on an unseen suite into a covert
    form of adaptation through its running statistics.
    """
    offenders = [
        type(m).__name__
        for m in model.modules()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    ]
    assert offenders == [], f"BatchNorm found: {offenders}"


def test_prediction_does_not_depend_on_other_batch_members(model):
    """With GroupNorm, a sample's output must be independent of its batch."""
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        batched = model(x)
        alone = model(x[:1])
    assert torch.allclose(batched[:1], alone, atol=1e-5)


# --------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------


def test_regression_head_output_is_unbounded():
    """No sigmoid: the model must be able to express a confident error."""
    head = RegressionHead(latent_dim=8, n_targets=2, hidden=None)
    with torch.no_grad():
        head.net.weight.fill_(0.0)
        head.net.bias.copy_(torch.tensor([-3.0, 7.0]))
    out = head(torch.zeros(1, 8))
    assert out[0, 0].item() < 0 and out[0, 1].item() > 1


def test_projection_head_shape():
    assert ProjectionHead(latent_dim=16, out_dim=8)(torch.randn(3, 16)).shape == (3, 8)


def test_domain_head_shape():
    assert DomainHead(latent_dim=16, n_domains=3)(torch.randn(5, 16)).shape == (5, 3)


# --------------------------------------------------------------------------
# Config construction
# --------------------------------------------------------------------------


def test_build_backbone_from_config():
    enc = build_backbone({"type": "small_cnn", "latent_dim": 64, "width": 8, "depth": 2})
    assert enc.latent_dim == 64
    assert enc(torch.randn(1, 1, 32, 32)).shape == (1, 64)


def test_unknown_backbone_is_rejected():
    with pytest.raises(NotImplementedError, match="small_cnn"):
        build_backbone({"type": "vit_gigantic"})


def test_model_from_config():
    m = ERMModel.from_config(
        {"model": {"type": "small_cnn", "latent_dim": 32, "width": 4, "depth": 2}}
    )
    assert m(torch.randn(2, 1, 32, 32)).shape == (2, 2)


def test_backbone_must_expose_latent_dim():
    class Bare(torch.nn.Module):
        pass

    with pytest.raises(AttributeError, match="latent_dim"):
        ERMModel(backbone=Bare())


def test_parameter_count_is_modest():
    """Section 60: value comes from experiment quality, not parameter count."""
    m = ERMModel(backbone=SmallCNN(latent_dim=256, width=32, depth=5))
    assert m.n_parameters < 20_000_000
