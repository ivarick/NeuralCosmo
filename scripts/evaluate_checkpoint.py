"""Evaluate a saved checkpoint on any suite, without retraining.

Plan reference: sections 18, 21, 54, 81.

    python scripts/evaluate_checkpoint.py --run E01_tng_id_seed00 --suite SIMBA

A cross-suite transfer experiment trains exactly the same model as the
corresponding in-domain experiment: same source suite, same seed, same
normalizer. Only the evaluation differs. Retraining to change the evaluation
would burn an hour of GPU for no new information, so this script loads the
frozen checkpoint and evaluates it wherever asked.

PROTOCOL
--------
Evaluating on a suite the model never trained on is the entire point and is
permitted (section 18). What must remain true is that the checkpoint was
produced without that suite's influence, so the script re-checks the run's
recorded normalizer provenance against the suites being evaluated and refuses
if the statistics came from data that should have been unseen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from neuralcosmos.data.builders import build_dataset  # noqa: E402
from neuralcosmos.data.manifest import load_data_config  # noqa: E402
from neuralcosmos.data.splits import load_split_file  # noqa: E402
from neuralcosmos.data.statistics import load_normalizer  # noqa: E402
from neuralcosmos.data.targets import TargetScaler  # noqa: E402
from neuralcosmos.evaluation.metrics import generalization_ratio  # noqa: E402
from neuralcosmos.models.dg_methods import build_dg_model  # noqa: E402
from neuralcosmos.models.erm import ERMModel  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402
from neuralcosmos.protocol import ExperimentProtocol, ProtocolViolation  # noqa: E402
from neuralcosmos.training.trainer import TrainConfig, Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run id or path under outputs/runs")
    ap.add_argument("--suite", action="append", required=True, help="suite to evaluate; repeatable")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--no-ram", action="store_true")
    ap.add_argument("--out", default=None, help="default: <run_dir>/transfer_<split>.json")
    args = ap.parse_args()

    root = repo_root()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / "outputs" / "runs" / args.run
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    exp = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    data_block = exp["data"]

    def _p(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else root / p

    cfg = load_data_config(_p(data_block["config"]))
    split_file = load_split_file(_p(data_block["split_file"]))
    normalizer = load_normalizer(_p(data_block["normalizer"]))

    sources = list(record["protocol"]["source_suites"])
    eval_suites = list(args.suite)

    # Declare every evaluated suite that is not a source as a target, so the
    # normalizer check below is applied against all of them.
    transfer = [s for s in eval_suites if s not in sources]
    try:
        protocol = ExperimentProtocol(
            source_suites=tuple(sources),
            target_suites=tuple(transfer),
            name="eval_only",
        )
        protocol.check_normalizer(normalizer.provenance)
    except (ValueError, ProtocolViolation) as exc:
        print(f"PROTOCOL VIOLATION\n{exc}", file=sys.stderr)
        return 4

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(exc, file=sys.stderr)
        return 2

    ckpt_path = run_dir / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_scaler = TargetScaler.from_config(cfg)
    # A DG checkpoint stores keys under "base.", so the same wrapper must be
    # rebuilt or load_state_dict fails on every parameter name.
    method = str(exp.get("method", {}).get("name", "erm"))
    if method == "erm" and "method" not in exp:
        model = ERMModel.from_config(exp, n_targets=len(target_scaler.names))
    else:
        model = build_dg_model(
            exp, n_targets=len(target_scaler.names), n_domains=len(sources)
        )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    print("=" * 74)
    print(f"  NeuralCosmos - evaluate {run_dir.name}")
    print("=" * 74)
    print(f"  checkpoint : {args.checkpoint}.pt (epoch {state['epoch']})")
    print(f"  trained on : {sources}")
    print(f"  evaluating : {eval_suites}  split={args.split}")
    print(f"  normalizer : {normalizer.provenance}")
    print(f"  git commit : {record.get('git', {}).get('commit', 'unknown')[:12]}")
    print()

    trainer = Trainer(
        model=model,
        train_dataset=None,
        val_datasets={},
        config=TrainConfig(batch_size=64, num_workers=0, seed=int(record.get("seed", 0))),
        run_dir=run_dir,
        target_names=list(target_scaler.names),
        target_spans=[hi - lo for lo, hi in zip(target_scaler.lower, target_scaler.upper)],
        protocol=None,
        device=device,
    )

    use_ram = not args.no_ram
    results: dict[str, dict] = {}
    for suite in eval_suites:
        ds = build_dataset(
            cfg=cfg, data_root=data_root, split_file=split_file,
            suites=[suite], split=args.split, normalizer=normalizer,
            protocol=protocol, role="eval", ram_cache=use_ram, ram_suites=[suite],
        )
        out = trainer.evaluate(ds, target_scaler, label=f"{suite}_{args.split}")
        results[suite] = {
            "map_level": out["map_level"],
            "simulation_level": out["simulation_level"],
        }
        np.savez_compressed(
            run_dir / f"predictions_{suite}_{args.split}.npz", **out["predictions"]
        )

    # ---- report ----------------------------------------------------------
    names = list(target_scaler.names)
    for level in ("map_level", "simulation_level"):
        print(f"  {level.replace('_level', '').upper()} LEVEL")
        print(f"    {'suite':<16}{'role':<8}"
              f"{'MAE ' + names[0]:>12}{'MAE ' + names[1]:>12}"
              f"{'R2 ' + names[0]:>11}{'R2 ' + names[1]:>11}")
        print("    " + "-" * 70)
        for suite, r in results.items():
            pt = r[level]["per_target"]
            role = "ID" if suite in sources else "OOD"
            print(f"    {suite:<16}{role:<8}"
                  f"{pt[names[0]]['mae']:>12.4f}{pt[names[1]]['mae']:>12.4f}"
                  f"{pt[names[0]]['r2']:>11.4f}{pt[names[1]]['r2']:>11.4f}")
        print()

    # ---- generalization ratio (sections 35, 56) --------------------------
    id_suites = [s for s in eval_suites if s in sources]
    ood_suites = [s for s in eval_suites if s not in sources]
    ratios: dict[str, dict] = {}

    if id_suites and ood_suites:
        id_suite = id_suites[0]
        print("  GENERALIZATION RATIO  G = OOD error / ID error  (sections 35, 56)")
        print(f"    {'comparison':<28}{'level':<14}{'target':<10}{'G (MAE)':>10}{'G (RMSE)':>11}")
        print("    " + "-" * 74)
        for ood in ood_suites:
            key = f"{id_suite}->{ood}"
            ratios[key] = {}
            for level in ("map_level", "simulation_level"):
                for n in names:
                    g_mae = generalization_ratio(
                        results[ood][level]["per_target"][n]["mae"],
                        results[id_suite][level]["per_target"][n]["mae"],
                    )
                    g_rmse = generalization_ratio(
                        results[ood][level]["per_target"][n]["rmse"],
                        results[id_suite][level]["per_target"][n]["rmse"],
                    )
                    ratios[key][f"{level}:{n}"] = {"mae": g_mae, "rmse": g_rmse}
                    print(f"    {key:<28}{level.replace('_level',''):<14}{n:<10}"
                          f"{g_mae:>10.2f}{g_rmse:>11.2f}")
        print()
        map_ratios = [
            v["mae"] for k, v in ratios[f"{id_suite}->{ood_suites[0]}"].items()
            if k.startswith("map_level")
        ]
        worst = max(map_ratios)
        print(f"    map-level G ranges {min(map_ratios):.2f} to {worst:.2f}")
        # Section 35's kill criterion is explicitly about the ratio being low
        # "consistently ... across both targets and BOTH transfer directions".
        # This script sees one direction, so it must not pronounce on the
        # criterion by itself -- a single weak direction is evidence of
        # asymmetry, not of a dead testbed.
        if worst < 1.2:
            print("    This direction alone shows little degradation. Section 35's kill")
            print("    criterion requires BOTH directions to be weak, so check the reverse")
            print("    transfer before concluding anything about the testbed.")
        else:
            print("    This direction shows a measurable shift.")
        print()

    out_path = Path(args.out) if args.out else run_dir / f"transfer_{args.split}.json"
    out_path.write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "checkpoint": args.checkpoint,
                "epoch": int(state["epoch"]),
                "split": args.split,
                "sources": sources,
                "evaluated": eval_suites,
                "normalizer_provenance": normalizer.provenance,
                "results": results,
                "generalization_ratios": ratios,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
