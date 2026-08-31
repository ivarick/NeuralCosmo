"""PPIRL model and the paired training path.

Plan reference: sections 43, 44, 45, 48, 49.

The tests follow the staged construction section 49 requires: M0 hydro-only
behaves exactly like ERM, each added component appears only when its weight is
non-zero, and the whole thing actually reduces loss on a fixed paired batch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralcosmos.data.dataset import SuiteSource  # noqa: E402
from neuralcosmos.data.paired_dataset import PairedMapDataset, PairedSuiteSource  # noqa: E402
from neuralcosmos.data.splits import maps_for_simulations  # noqa: E402
from neuralcosmos.data.targets import TargetScaler  # noqa: E402
from neuralcosmos.models.backbones.small_cnn import SmallCNN  # noqa: E402
from neuralcosmos.models.erm import ERMModel  # noqa: E402
from neuralcosmos.models.ppirl import PPIRLModel, build_ppirl_model  # noqa: E402

MAPS_PER_SIM = 15
PARAM_COLUMNS = {"omega_m": 0, "sigma8": 1, "a_sn1": 2, "a_agn1": 3, "a_sn2": 4, "a_agn2": 5}


def _base(latent: int = 16):
    return ERMModel(backbone=SmallCNN(latent_dim=latent, width=4, depth=2))


# --------------------------------------------------------------------------
# Inference path: PPIRL is a drop-in regressor at test time
# --------------------------------------------------------------------------


def test_forward_predicts_from_a_single_map():
    model = PPIRLModel(_base())
    y = model(torch.randn(4, 1, 32, 32))
    assert y.shape == (4, 2)


def test_inference_matches_the_underlying_erm():
    base = _base()
    model = PPIRLModel(base)
    model.eval()
    x = torch.randn(3, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(model(x), base(x))


def test_exposes_backbone_and_regressor_for_probing():
    model = PPIRLModel(_base())
    assert model.backbone is model.base.backbone
    assert model.regressor is model.base.regressor


# --------------------------------------------------------------------------
# Staged objective (section 49)
# --------------------------------------------------------------------------


def test_m0_hydro_only_produces_no_auxiliary_losses():
    """pair/var/cov and N-body regression all off -> pure ERM."""
    model = PPIRLModel(_base(), nbody_reg_weight=0.0, pair_weight=0.0,
                       var_weight=0.0, cov_weight=0.0)
    _, aux = model.forward_pair(
        torch.randn(4, 1, 32, 32), torch.randn(4, 1, 32, 32), torch.rand(4, 2)
    )
    assert aux == {}


def test_m1_nbody_regression_appears_only_with_weight():
    off = PPIRLModel(_base(), nbody_reg_weight=0.0, pair_weight=0.0,
                     var_weight=0.0, cov_weight=0.0)
    on = PPIRLModel(_base(), nbody_reg_weight=1.0, pair_weight=0.0,
                    var_weight=0.0, cov_weight=0.0)
    args = (torch.randn(4, 1, 32, 32), torch.randn(4, 1, 32, 32), torch.rand(4, 2))
    assert "reg_nbody" not in off.forward_pair(*args)[1]
    assert "reg_nbody" in on.forward_pair(*args)[1]


def test_m2_pair_alignment_appears_with_weight():
    model = PPIRLModel(_base(), pair_weight=1.0, var_weight=1.0, cov_weight=0.04)
    _, aux = model.forward_pair(
        torch.randn(4, 1, 32, 32), torch.randn(4, 1, 32, 32), torch.rand(4, 2)
    )
    assert "pair" in aux and "var" in aux and "cov" in aux


def test_all_auxiliary_losses_are_finite():
    model = PPIRLModel(_base(), nbody_reg_weight=1.0, pair_weight=1.0,
                       var_weight=1.0, cov_weight=0.04)
    _, aux = model.forward_pair(
        torch.randn(8, 1, 32, 32), torch.randn(8, 1, 32, 32), torch.rand(8, 2)
    )
    for name, value in aux.items():
        assert torch.isfinite(value), name


# --------------------------------------------------------------------------
# Gradients reach the encoder through both views
# --------------------------------------------------------------------------


def test_pair_loss_gradient_reaches_the_encoder():
    model = PPIRLModel(_base(), pair_weight=1.0, var_weight=0.0, cov_weight=0.0)
    pred, aux = model.forward_pair(
        torch.randn(8, 1, 32, 32), torch.randn(8, 1, 32, 32), torch.rand(8, 2)
    )
    aux["pair"].backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_projection_head_can_be_disabled():
    model = PPIRLModel(_base(), use_projection=False, pair_weight=1.0)
    assert model.projector is None
    # Without a projection head the pair loss acts directly on the latent.
    _, aux = model.forward_pair(
        torch.randn(4, 1, 32, 32), torch.randn(4, 1, 32, 32), torch.rand(4, 2)
    )
    assert "pair" in aux


def test_a_paired_step_reduces_the_combined_loss():
    torch.manual_seed(0)
    model = PPIRLModel(_base(), nbody_reg_weight=1.0, pair_weight=1.0,
                       var_weight=1.0, cov_weight=0.04)
    x_h = torch.randn(8, 1, 32, 32)
    x_n = torch.randn(8, 1, 32, 32)
    y = torch.rand(8, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def total():
        pred, aux = model.forward_pair(x_h, x_n, y)
        return torch.nn.functional.mse_loss(pred, y) + sum(aux.values())

    first = total().item()
    for _ in range(20):
        opt.zero_grad()
        loss = total()
        loss.backward()
        opt.step()
    assert total().item() < first


# --------------------------------------------------------------------------
# Config construction
# --------------------------------------------------------------------------


def test_build_from_config():
    cfg = {
        "model": {"type": "small_cnn", "latent_dim": 32, "width": 4, "depth": 2},
        "method": {"name": "ppirl", "pair_weight": 2.0, "nbody_reg_weight": 0.5},
    }
    m = build_ppirl_model(cfg)
    assert m.pair_weight == 2.0
    assert m.nbody_reg_weight == 0.5
    assert m.is_paired


# --------------------------------------------------------------------------
# End-to-end with the trainer and a real paired dataset
# --------------------------------------------------------------------------


def _paired_dataset(archive: Path, scaler):
    params = np.loadtxt(archive / "SuiteA LH parameters.txt")
    idx = maps_for_simulations(list(range(4)), MAPS_PER_SIM)
    hydro = SuiteSource("SuiteA", 0, archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy",
                        params, idx, MAPS_PER_SIM)
    nbody = SuiteSource("SuiteA", 0, archive / "Maps_Mtot_SuiteA_Nbody_LH_z=0.00.npy",
                        params, idx, MAPS_PER_SIM)
    src = PairedSuiteSource("SuiteA", hydro, nbody)
    return PairedMapDataset([src], scaler, PARAM_COLUMNS)


def test_trainer_runs_a_paired_epoch(tmp_path, synthetic_paired_archive):
    from neuralcosmos.training.trainer import TrainConfig, Trainer
    from neuralcosmos.data.dataset import CAMELSMapDataset

    scaler = TargetScaler(("omega_m", "sigma8"), (0.1, 0.6), (0.5, 1.0))
    train_ds = _paired_dataset(synthetic_paired_archive, scaler)

    # Validation is hydro-only maps: PPIRL predicts cosmology from one map.
    params = np.loadtxt(synthetic_paired_archive / "SuiteA LH parameters.txt")
    val_idx = maps_for_simulations([4, 5], MAPS_PER_SIM)
    val_src = SuiteSource("SuiteA", 0, synthetic_paired_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy",
                          params, val_idx, MAPS_PER_SIM)
    val_ds = CAMELSMapDataset([val_src], scaler, PARAM_COLUMNS)

    model = PPIRLModel(_base(), nbody_reg_weight=1.0, pair_weight=1.0,
                       var_weight=1.0, cov_weight=0.04)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_datasets={"SuiteA": val_ds},
        config=TrainConfig(epochs=2, batch_size=8, num_workers=0, amp=False,
                           balanced_batches=False, early_stopping_patience=None,
                           warmup_epochs=0),
        run_dir=tmp_path / "run",
        target_names=["omega_m", "sigma8"],
        target_spans=[0.4, 0.4],
        device=torch.device("cpu"),
    )
    summary = trainer.fit(scaler)
    assert summary["epochs_run"] == 2
    # The auxiliary losses were tracked, not silently dropped.
    assert any(k in trainer.history[-1].aux for k in ("pair", "var", "reg_nbody"))


def test_trainer_paired_and_shuffled_differ(tmp_path, synthetic_paired_archive):
    """The M2 vs M3 comparison is a dataset flag, not a model change."""
    scaler = TargetScaler(("omega_m", "sigma8"), (0.1, 0.6), (0.5, 1.0))
    params = np.loadtxt(synthetic_paired_archive / "SuiteA LH parameters.txt")
    idx = maps_for_simulations(list(range(4)), MAPS_PER_SIM)
    hydro = SuiteSource("SuiteA", 0, synthetic_paired_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy",
                        params, idx, MAPS_PER_SIM)
    nbody = SuiteSource("SuiteA", 0, synthetic_paired_archive / "Maps_Mtot_SuiteA_Nbody_LH_z=0.00.npy",
                        params, idx, MAPS_PER_SIM)
    src = PairedSuiteSource("SuiteA", hydro, nbody)

    correct = PairedMapDataset([src], scaler, PARAM_COLUMNS, shuffle_pairs=False)
    shuffled = PairedMapDataset([src], scaler, PARAM_COLUMNS, shuffle_pairs=True)

    # Same model class, same batch position, different N-body partner -> the
    # pair-consistency loss sees different inputs. That difference is the whole
    # experiment (section 50).
    model = PPIRLModel(_base(), pair_weight=1.0, var_weight=0.0, cov_weight=0.0)
    b_c = correct[0]
    b_s = shuffled[0]
    _, aux_c = model.forward_pair(
        torch.tensor(b_c["hydro_image"])[None], torch.tensor(b_c["nbody_image"])[None],
        torch.tensor(b_c["target"])[None],
    )
    _, aux_s = model.forward_pair(
        torch.tensor(b_s["hydro_image"])[None], torch.tensor(b_s["nbody_image"])[None],
        torch.tensor(b_s["target"])[None],
    )
    # Not asserting which is larger -- an untrained net has no reason to prefer
    # correct pairs -- only that the control actually changes the input.
    assert not np.allclose(b_c["nbody_image"], b_s["nbody_image"])
