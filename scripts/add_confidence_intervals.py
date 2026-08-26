"""Attach simulation-level confidence intervals to saved predictions.

Plan reference: sections 56, 57, 58.

    python scripts/add_confidence_intervals.py --run E01_tng_id_seed00

Every run saves its raw predictions, so intervals can be added after the fact
without retraining. Section 57 requires resampling simulations rather than
maps, which is what makes these intervals honest rather than flattering.

Section 58 notes that seed spread and test-simulation sampling are DIFFERENT
sources of uncertainty. The seed spread already reported in
reports/baseline_report.md answers "would another training run agree?"; these
intervals answer "would another draw of test simulations agree?". Neither
substitutes for the other, and a result needs both to be trusted.
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

from neuralcosmos.evaluation.bootstrap import bootstrap_metric  # noqa: E402
from neuralcosmos.evaluation.metrics import mae, rmse  # noqa: E402
from neuralcosmos.paths import repo_root  # noqa: E402


def _per_target(fn, col: int):
    def inner(a: np.ndarray, b: np.ndarray) -> float:
        return float(fn(a[:, col], b[:, col]))
    return inner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="run id; repeatable")
    ap.add_argument("--split", default="test")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--targets", default="omega_m,sigma8")
    ap.add_argument("--out", default="reports/confidence_intervals.json")
    args = ap.parse_args()

    root = repo_root()
    names = args.targets.split(",")
    results: dict = {}

    print("=" * 78)
    print("  NeuralCosmos - simulation-level confidence intervals (plan section 57)")
    print("=" * 78)
    print(f"  replicates : {args.replicates}   confidence : {args.confidence:.0%}")
    print("  resampling : whole simulations, all 15 maps each -- never maps alone")
    print()

    for run_id in args.run:
        run_dir = root / "outputs" / "runs" / run_id
        if not run_dir.exists():
            print(f"  [skip] {run_id}: no run directory", file=sys.stderr)
            continue

        preds = sorted(run_dir.glob(f"predictions_*_{args.split}.npz"))
        if not preds:
            print(f"  [skip] {run_id}: no saved predictions for split {args.split}",
                  file=sys.stderr)
            continue

        print(f"  {run_id}")
        results[run_id] = {}

        for p in preds:
            suite = p.name[len("predictions_"):-len(f"_{args.split}.npz")]
            d = np.load(p)
            y_true, y_pred = d["true"], d["pred"]
            sims, suites = d["simulation_id"], d["suite_id"]

            # Predictions are stored in the scaled target space; convert to
            # physical units so an interval reads as "+/- 0.002 in Omega_m".
            lo = np.array([0.1, 0.6])[: y_true.shape[1]]
            span = np.array([0.4, 0.4])[: y_true.shape[1]]
            yt = y_true * span + lo
            yp = y_pred * span + lo

            entry: dict = {}
            for i, name in enumerate(names[: yt.shape[1]]):
                for label, fn in (("mae", mae), ("rmse", rmse)):
                    res = bootstrap_metric(
                        _per_target(fn, i), yt, yp, sims, suites,
                        n_replicates=args.replicates,
                        confidence=args.confidence,
                        seed=0,
                    )
                    entry[f"{name}_{label}"] = res.to_dict()
                    if label == "mae":
                        print(f"    {suite:<16}{name:<10}MAE {res.point:.4f}  "
                              f"[{res.lower:.4f}, {res.upper:.4f}]  "
                              f"({res.n_groups} sims)")
            results[run_id][suite] = entry
        print()

    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"split": args.split, "replicates": args.replicates,
         "confidence": args.confidence, "runs": results}, indent=2
    ), encoding="utf-8")
    print(f"  written: {out}")
    print()
    print("  Section 58: these intervals quantify TEST-SIMULATION sampling only.")
    print("  Seed-to-seed spread is a separate uncertainty, reported alongside in")
    print("  reports/baseline_report.md. A result needs both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
