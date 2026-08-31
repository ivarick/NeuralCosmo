"""Training loop with source-only model selection and full run provenance.

Plan reference: sections 61, 62, 63, 64, 65, 67.

Two rules are enforced structurally rather than by convention:

- Model selection uses SOURCE validation only. The trainer never receives a
  target-suite validation loader, and the protocol object is asked to confirm
  the validation suites before training starts (sections 19, 62, 63).
- Multi-domain validation is aggregated with EQUAL domain weight, not
  proportional to sample count (section 63). Otherwise a larger source suite
  would quietly dominate early stopping.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..evaluation.metrics import regression_metrics, selection_score
from ..protocol import ExperimentProtocol
from .samplers import BalancedSuiteBatchSampler
from .seed import describe_environment, seed_everything, worker_init_fn

__all__ = ["TrainConfig", "Trainer", "git_commit"]


def git_commit(repo: Path) -> dict[str, Any]:
    """Current commit and dirty state, for the run record (section 64)."""
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return None

    sha = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": sha,
        "dirty": bool(status) if status is not None else None,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


@dataclass
class TrainConfig:
    """Hyperparameters. Section 82's values are starting points, not optima."""

    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    amp: bool = True
    grad_clip: float | None = 1.0
    early_stopping_patience: int | None = 20
    warmup_epochs: int = 2
    min_lr_factor: float = 0.01
    seed: int = 0
    balanced_batches: bool = True
    log_every: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_score: float
    per_suite: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    lr: float = 0.0
    # Auxiliary losses for the DG baselines, kept separate from the task loss
    # so an alignment term shrinking while the regression worsens stays visible.
    aux: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Trainer:
    """Trains one model and records everything needed to reproduce the run."""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_datasets: dict[str, Any],
        config: TrainConfig,
        run_dir: Path,
        target_names: Sequence[str],
        target_spans: Sequence[float],
        protocol: ExperimentProtocol | None = None,
        device: torch.device | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        if protocol is not None:
            # Refuse before a single gradient step if selection would see the target.
            protocol.check_validation_suites(val_datasets.keys())

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move here rather than in fit(): evaluate() is also called directly on
        # a loaded checkpoint, and a CPU model fed CUDA inputs fails deep inside
        # the first convolution with a dtype error that names neither cause.
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_datasets = dict(val_datasets)
        self.cfg = config
        self.run_dir = Path(run_dir)
        self.target_names = list(target_names)
        self.target_spans = list(target_spans)
        self.protocol = protocol
        self.extra_metadata = extra_metadata or {}

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[EpochRecord] = []
        self.best_score = float("inf")
        self.best_epoch = -1

    # -- data ---------------------------------------------------------------

    def _train_loader(self, epoch: int) -> DataLoader:
        common = dict(
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_init_fn,
            persistent_workers=self.cfg.num_workers > 0,
        )
        suite_ids = self.train_dataset.suite_ids()

        if self.cfg.balanced_batches and len(np.unique(suite_ids)) > 1:
            sampler = BalancedSuiteBatchSampler(
                suite_ids, self.cfg.batch_size, seed=self.cfg.seed
            )
            sampler.set_epoch(epoch)
            return DataLoader(self.train_dataset, batch_sampler=sampler, **common)

        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            **common,
        )

    def _val_loader(self, dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
        )

    # -- one epoch ----------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer,
        scaler,
        use_amp: bool,
        progress: float = 0.0,
    ) -> tuple[float, dict[str, float]]:
        """One epoch. Returns the task loss and any auxiliary loss averages.

        A model exposing ``forward_train`` is a domain-generalization baseline
        and additionally returns auxiliary losses computed from the same latent
        vector. The task loss is tracked separately from the total so that
        methods remain comparable: an alignment term can shrink the total while
        the regression is getting worse, and averaging them into one number
        would hide exactly that.
        """
        self.model.train()
        total, n = 0.0, 0
        aux_totals: dict[str, float] = {}

        is_paired = getattr(self.model, "is_paired", False)
        is_dg = (not is_paired) and hasattr(self.model, "forward_train")

        for batch in loader:
            y = batch["target"].to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                if is_paired:
                    # Paired batch: hydro + N-body views. The hydro regression
                    # is the task loss; everything the paired method adds comes
                    # back pre-weighted in aux (sections 44-48).
                    x_h = batch["hydro_image"].to(self.device, non_blocking=True)
                    x_n = batch["nbody_image"].to(self.device, non_blocking=True)
                    pred, aux = self.model.forward_pair(x_h, x_n, y)
                    task_loss = nn.functional.mse_loss(pred, y)
                    loss = task_loss
                    for name, value in aux.items():
                        loss = loss + value
                    bs = x_h.shape[0]
                elif is_dg:
                    x = batch["image"].to(self.device, non_blocking=True)
                    domains = batch["suite_id"].to(self.device, non_blocking=True)
                    pred, _, aux = self.model.forward_train(x, domains, progress=progress)
                    task_loss = nn.functional.mse_loss(pred, y)
                    loss = task_loss
                    for name, value in aux.items():
                        if name.startswith("_"):      # diagnostics, not losses
                            continue
                        loss = loss + value
                    bs = x.shape[0]
                else:
                    x = batch["image"].to(self.device, non_blocking=True)
                    aux = {}
                    task_loss = nn.functional.mse_loss(self.model(x), y)
                    loss = task_loss
                    bs = x.shape[0]

            scaler.scale(loss).backward()
            if self.cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total += task_loss.item() * bs
            n += bs
            for name, value in aux.items():
                aux_totals[name] = aux_totals.get(name, 0.0) + float(value.item()) * bs

        denom = max(n, 1)
        return total / denom, {k: v / denom for k, v in aux_totals.items()}

    @torch.no_grad()
    def predict(self, dataset, use_amp: bool = True) -> dict[str, np.ndarray]:
        """Predictions in the scaled target space, plus grouping keys."""
        self.model.eval()
        preds, trues, sims, suites, maps = [], [], [], [], []

        for batch in self._val_loader(dataset):
            x = batch["image"].to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                out = self.model(x)
            preds.append(out.float().cpu().numpy())
            trues.append(batch["target"].numpy())
            sims.append(batch["simulation_id"].numpy())
            suites.append(batch["suite_id"].numpy())
            maps.append(batch["map_id"].numpy())

        return {
            "pred": np.concatenate(preds),
            "true": np.concatenate(trues),
            "simulation_id": np.concatenate(sims),
            "suite_id": np.concatenate(suites),
            "map_id": np.concatenate(maps),
        }

    def _validate(self, scaler_inverse) -> tuple[float, float, dict[str, float]]:
        """Return (mean loss, selection score, per-suite scores).

        Section 63: suites are weighted equally, not by sample count.
        """
        per_suite_score: dict[str, float] = {}
        per_suite_loss: dict[str, float] = {}

        for name, ds in self.val_datasets.items():
            out = self.predict(ds, use_amp=self.cfg.amp and self.device.type == "cuda")
            per_suite_loss[name] = float(np.mean((out["pred"] - out["true"]) ** 2))
            # Score in physical units so it is comparable across experiments.
            per_suite_score[name] = selection_score(
                scaler_inverse(out["true"]), scaler_inverse(out["pred"]), self.target_spans
            )

        return (
            float(np.mean(list(per_suite_loss.values()))),
            float(np.mean(list(per_suite_score.values()))),
            per_suite_score,
        )

    # -- checkpoints --------------------------------------------------------

    def _save(self, name: str, epoch: int, optimizer, score: float) -> Path:
        path = self.run_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "selection_score": score,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        return path

    # -- fit ----------------------------------------------------------------

    def fit(self, target_scaler) -> dict[str, Any]:
        seed_everything(self.cfg.seed)
        self.model.to(self.device)

        use_amp = self.cfg.amp and self.device.type == "cuda"
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        total_epochs = self.cfg.epochs
        warmup = max(0, min(self.cfg.warmup_epochs, total_epochs - 1))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / (warmup + 1)
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            cosine = 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
            return self.cfg.min_lr_factor + (1 - self.cfg.min_lr_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        inverse = target_scaler.inverse

        start = time.time()
        epochs_without_improvement = 0

        print(f"  device={self.device}  train={len(self.train_dataset):,} maps  "
              f"val={{{', '.join(f'{k}:{len(v):,}' for k, v in self.val_datasets.items())}}}")
        print(f"  {'epoch':>5} {'train':>10} {'val':>10} {'score':>10} {'lr':>9} {'sec':>6}")
        print("  " + "-" * 56)

        for epoch in range(total_epochs):
            t0 = time.time()
            progress = epoch / max(1, total_epochs - 1)
            train_loss, aux = self._train_epoch(
                self._train_loader(epoch), optimizer, scaler, use_amp, progress=progress
            )
            val_loss, val_score, per_suite = self._validate(inverse)
            scheduler.step()
            dt = time.time() - t0

            rec = EpochRecord(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_score=val_score,
                per_suite=per_suite,
                seconds=dt,
                lr=optimizer.param_groups[0]["lr"],
            )
            rec.aux = aux
            self.history.append(rec)

            marker = ""
            if val_score < self.best_score:
                self.best_score = val_score
                self.best_epoch = epoch
                self._save("best", epoch, optimizer, val_score)
                epochs_without_improvement = 0
                marker = "  *"
            else:
                epochs_without_improvement += 1

            aux_str = ""
            if aux:
                aux_str = "  " + " ".join(
                    f"{k}={v:.4f}" for k, v in aux.items() if not k.startswith("_")
                )
                if "_lambda" in aux:
                    aux_str += f" lam={aux['_lambda']:.2f}"
            print(f"  {epoch:>5} {train_loss:>10.5f} {val_loss:>10.5f} "
                  f"{val_score:>10.5f} {rec.lr:>9.2e} {dt:>6.1f}{marker}{aux_str}")

            if (
                self.cfg.early_stopping_patience is not None
                and epochs_without_improvement >= self.cfg.early_stopping_patience
            ):
                print(f"  early stop: no source-validation improvement for "
                      f"{epochs_without_improvement} epochs")
                break

        self._save("last", len(self.history) - 1, optimizer, self.history[-1].val_score)
        elapsed = time.time() - start

        summary = {
            "best_epoch": self.best_epoch,
            "best_selection_score": self.best_score,
            "epochs_run": len(self.history),
            "total_seconds": round(elapsed, 1),
            "history": [r.to_dict() for r in self.history],
        }
        self._write_run_record(summary)
        return summary

    # -- provenance ---------------------------------------------------------

    def _write_run_record(self, summary: dict[str, Any]) -> Path:
        record = {
            "run_id": self.run_dir.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_commit(Path(__file__).resolve().parents[3]),
            "seed": self.cfg.seed,
            "config": self.cfg.to_dict(),
            "protocol": self.protocol.to_dict() if self.protocol else None,
            "environment": describe_environment(),
            "targets": self.target_names,
            "train_maps": len(self.train_dataset),
            "val_maps": {k: len(v) for k, v in self.val_datasets.items()},
            "summary": {k: v for k, v in summary.items() if k != "history"},
            **self.extra_metadata,
        }
        (self.run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        (self.run_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return self.run_dir / "run.json"

    def evaluate(self, dataset, target_scaler, label: str = "eval") -> dict[str, Any]:
        """Map-level and simulation-level metrics in physical units (section 54)."""
        from ..evaluation.metrics import aggregate_by_simulation

        out = self.predict(dataset, use_amp=self.cfg.amp and self.device.type == "cuda")
        true_phys = target_scaler.inverse(out["true"])
        pred_phys = target_scaler.inverse(out["pred"])

        map_level = regression_metrics(
            true_phys, pred_phys, self.target_names, spans=self.target_spans
        )
        t_sim, p_sim, _ = aggregate_by_simulation(
            true_phys, pred_phys, out["simulation_id"], out["suite_id"]
        )
        sim_level = regression_metrics(
            t_sim, p_sim, self.target_names, spans=self.target_spans
        )

        return {
            "label": label,
            "map_level": map_level,
            "simulation_level": sim_level,
            "predictions": out,
        }
