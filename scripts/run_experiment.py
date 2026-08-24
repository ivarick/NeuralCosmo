"""Run one experiment from a config file.

Plan reference: sections 64, 65, 67, 81.

    python scripts/run_experiment.py --config configs/experiments/e01_tng_id.yaml
    python scripts/run_experiment.py --config configs/experiments/e01_tng_id.yaml --seed 1

Run directories are named deterministically (section 65):

    outputs/runs/E01_tng_id_seed00/

Never ``final2`` or ``final_really_best``.

The protocol object is constructed from the config and checked before any data
is read, so an experiment that would leak the sealed target fails immediately
rather than after an hour of training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from neuralcosmos.data.builders import build_dataset  # noqa: E402
from neuralcosmos.data.manifest import load_data_config  # noqa: E402
from neuralcosmos.data.splits import load_split_file  # noqa: E402
from neuralcosmos.data.statistics import load_normalizer  # noqa: E402
from neuralcosmos.data.targets import TargetScaler  # noqa: E402
from neuralcosmos.models.erm import ERMModel  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402
from neuralcosmos.protocol import ExperimentProtocol, ProtocolViolation  # noqa: E402
from neuralcosmos.training.seed import seed_everything  # noqa: E402
from neuralcosmos.training.trainer import TrainConfig, Trainer  # noqa: E402


class _LoadProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._suite: str | None = None
        self._last = 0.0
        self._t0 = time.time()

    def __call__(self, suite: str, done: int, total: int) -> None:
        if not self.enabled:
            return
        if suite != self._suite:
            self._suite = suite
            self._t0 = time.time()
            self._last = 0.0
        now = time.time()
        if done < total and (now - self._last) < 0.5:
            return
        self._last = now
        frac = done / total
        rate = done / max(now - self._t0, 1e-9)
        eta = (total - done) / rate if rate > 0 else 0
        bar = "#" * int(24 * frac) + "-" * (24 - int(24 * frac))
        end = "\n" if done >= total else "\r"
        sys.stdout.write(f"    {suite:<14} [{bar}] {frac*100:5.1f}%  eta {eta:4.0f}s{end}")
        sys.stdout.flush()


def load_experiment(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override the config seed")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override for a quick check")
    ap.add_argument("--max-simulations", type=int, default=None)
    ap.add_argument("--no-ram", action="store_true", help="stream from disk instead of RAM")
    ap.add_argument("--output-root", default="outputs/runs")
    ap.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    args = ap.parse_args()

    root = repo_root()

    def _p(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else root / p

    exp = load_experiment(_p(args.config))
    exp_block = exp.get("experiment", {})
    data_block = exp.get("data", {})
    train_block = exp.get("training", {})

    seed = args.seed if args.seed is not None else int(exp_block.get("seed", 0))
    run_id = f"{exp_block.get('id', 'experiment')}_seed{seed:02d}"
    run_dir = _p(args.output_root) / run_id

    if run_dir.exists() and (run_dir / "run.json").exists() and not args.force:
        print(f"Run already exists: {run_dir}\nPass --force to overwrite.", file=sys.stderr)
        return 3

    # ---- protocol first, before any data is touched ----------------------
    try:
        protocol = ExperimentProtocol.from_config(exp)
    except ValueError as exc:
        print(f"PROTOCOL ERROR: {exc}", file=sys.stderr)
        return 4

    cfg = load_data_config(_p(data_block["config"]))
    split_file = load_split_file(_p(data_block["split_file"]))
    normalizer = load_normalizer(_p(data_block["normalizer"]))

    try:
        protocol.check_normalizer(normalizer.provenance)
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION\n{exc}", file=sys.stderr)
        return 4

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(exc, file=sys.stderr)
        return 2

    sources = list(protocol.source_suites)
    max_sims = (
        args.max_simulations
        if args.max_simulations is not None
        else data_block.get("max_simulations")
    )
    use_ram = (not args.no_ram) and bool(data_block.get("ram_cache", True))

    print("=" * 74)
    print(f"  NeuralCosmos - {run_id}")
    print("=" * 74)
    print(f"  config     : {args.config}")
    print(f"  {protocol.describe()}")
    print(f"  split      : {split_file.content_hash()[:12]}")
    print(f"  normalizer : {normalizer.provenance}")
    print(f"  seed       : {seed}")
    if exp_block.get("notes"):
        print(f"  notes      : {exp_block['notes']}")
    print()

    seed_everything(seed)

    if use_ram:
        print("  loading maps into RAM (one-off quantisation pass)")
    t0 = time.time()
    common = dict(
        cfg=cfg, data_root=data_root, split_file=split_file, normalizer=normalizer,
        protocol=protocol, ram_cache=use_ram, ram_suites=sources,
        progress=_LoadProgress(enabled=use_ram),
    )
    try:
        train_ds = build_dataset(
            suites=sources, split="train", role="train",
            augment=bool(data_block.get("augment", False)), augment_seed=seed,
            max_simulations=max_sims, **common,
        )
        val_datasets = {
            s: build_dataset(suites=[s], split="val", role="val", **common)
            for s in sources
        }
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION\n{exc}", file=sys.stderr)
        return 4
    load_seconds = time.time() - t0

    if use_ram:
        from neuralcosmos.data.cache import ram_cache_bytes

        print(f"  resident   : {ram_cache_bytes() / 1024**3:.2f} GiB "
              f"in {load_seconds:.0f}s")
    print()

    # ---- model -----------------------------------------------------------
    target_scaler = TargetScaler.from_config(cfg)
    model = ERMModel.from_config(exp, n_targets=len(target_scaler.names))
    print(f"  model      : {exp.get('model', {}).get('type', 'small_cnn')}, "
          f"{model.n_parameters:,} parameters, latent {model.latent_dim}")

    workers = train_ds.safe_num_workers(int(train_block.get("num_workers", 2)))
    train_cfg = TrainConfig(
        epochs=args.epochs if args.epochs is not None else int(train_block.get("epochs", 60)),
        batch_size=int(train_block.get("batch_size", 32)),
        learning_rate=float(train_block.get("learning_rate", 3e-4)),
        weight_decay=float(train_block.get("weight_decay", 1e-4)),
        num_workers=workers,
        amp=bool(train_block.get("amp", True)),
        grad_clip=train_block.get("grad_clip", 1.0),
        early_stopping_patience=train_block.get("early_stopping_patience", 15),
        warmup_epochs=int(train_block.get("warmup_epochs", 2)),
        seed=seed,
    )
    if workers == 0 and int(train_block.get("num_workers", 2)) > 0:
        print("  workers    : forced to 0 (RAM-backed dataset)")
    print()

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(exp, sort_keys=False), encoding="utf-8"
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_datasets=val_datasets,
        config=train_cfg,
        run_dir=run_dir,
        target_names=list(target_scaler.names),
        target_spans=[hi - lo for lo, hi in zip(target_scaler.lower, target_scaler.upper)],
        protocol=protocol,
        extra_metadata={
            "experiment_id": exp_block.get("id"),
            "split_hash": split_file.content_hash(),
            "normalizer_provenance": normalizer.provenance,
            "data_config": str(data_block["config"]),
            "max_simulations": max_sims,
            "ram_cache": use_ram,
            "load_seconds": round(load_seconds, 1),
        },
    )

    summary = trainer.fit(target_scaler)

    print()
    print(f"  best epoch {summary['best_epoch']} "
          f"(source-validation score {summary['best_selection_score']:.5f})")
    print(f"  trained in {summary['total_seconds'] / 60:.1f} min")
    print()

    # ---- evaluation ------------------------------------------------------
    eval_splits = exp.get("evaluation", {}).get("splits", ["val"])
    results: dict[str, dict] = {}

    best = run_dir / "best.pt"
    if best.exists():
        import torch

        state = torch.load(best, map_location=trainer.device, weights_only=False)
        model.load_state_dict(state["model_state"])
        print(f"  evaluating checkpoint from epoch {state['epoch']}")

    for split in eval_splits:
        for suite in sources:
            ds = build_dataset(suites=[suite], split=split, role="eval", **common)
            out = trainer.evaluate(ds, target_scaler, label=f"{suite}_{split}")
            key = f"{suite}_{split}"
            results[key] = {
                "map_level": out["map_level"],
                "simulation_level": out["simulation_level"],
            }
            np.savez_compressed(
                run_dir / f"predictions_{key}.npz", **out["predictions"]
            )

    (run_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print("  " + "-" * 70)
    print(f"  {'set':<24}{'level':<14}{'MAE Om':>10}{'MAE s8':>10}{'R2 Om':>10}{'R2 s8':>10}")
    print("  " + "-" * 70)
    for key, r in results.items():
        for level in ("map_level", "simulation_level"):
            pt = r[level]["per_target"]
            names = list(pt)
            print(f"  {key:<24}{level.replace('_level',''):<14}"
                  f"{pt[names[0]]['mae']:>10.4f}{pt[names[1]]['mae']:>10.4f}"
                  f"{pt[names[0]]['r2']:>10.4f}{pt[names[1]]['r2']:>10.4f}")
    print("  " + "-" * 70)
    print()
    print(f"  run directory : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
