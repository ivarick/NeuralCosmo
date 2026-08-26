"""Probe a frozen encoder for simulator identity and cosmological information.

Plan reference: sections 37, 38, 81.

    python scripts/run_domain_probe.py --run E01_tng_id_seed00

Tests hypothesis H2: a frozen representation from an ordinary source-trained
regressor should let a probe classifier predict simulator identity better than
chance.

Both probes are always run and always reported together. Section 37 is explicit
that a near-chance domain probe is not by itself evidence of a good
representation, because a collapsed representation hides the simulator too.
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
from neuralcosmos.evaluation.representations import (  # noqa: E402
    extract_embeddings,
    fit_domain_probe,
    fit_target_probe,
)
from neuralcosmos.models.dg_methods import build_dg_model  # noqa: E402
from neuralcosmos.models.erm import ERMModel  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--suite", action="append", default=None,
                    help="suites to probe between; default IllustrisTNG + SIMBA")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="best", choices=["best", "last"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = repo_root()
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = root / "outputs" / "runs" / args.run
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 2

    exp = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    data_block = exp["data"]

    def _p(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else root / p

    cfg = load_data_config(_p(data_block["config"]))
    split_file = load_split_file(_p(data_block["split_file"]))
    normalizer = load_normalizer(_p(data_block["normalizer"]))
    suites = args.suite or ["IllustrisTNG", "SIMBA"]

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(exc, file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_scaler = TargetScaler.from_config(cfg)
    # A DG checkpoint stores its keys under "base.", so the same wrapper has to
    # be rebuilt or load_state_dict fails on every parameter name.
    method = str(exp.get("method", {}).get("name", "erm"))
    srcs = list(record["protocol"]["source_suites"])
    if method == "erm" and "method" not in exp:
        model = ERMModel.from_config(exp, n_targets=len(target_scaler.names))
    else:
        from neuralcosmos.data.builders import suite_id_map

        ids = suite_id_map(cfg)
        model = build_dg_model(
            exp, n_targets=len(target_scaler.names), n_domains=len(srcs),
            domain_ids=[ids[s] for s in srcs],
        )
    state = torch.load(run_dir / f"{args.checkpoint}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    print("=" * 74)
    print(f"  NeuralCosmos - representation probes ({run_dir.name})")
    print("=" * 74)
    print(f"  checkpoint : {args.checkpoint}.pt (epoch {state['epoch']})")
    print(f"  trained on : {record['protocol']['source_suites']}")
    print(f"  probing    : {suites}  split={args.split}")
    print(f"  latent dim : {model.latent_dim}")
    print()

    ds = build_dataset(
        cfg=cfg, data_root=data_root, split_file=split_file,
        suites=suites, split=args.split, normalizer=normalizer,
        role="eval", ram_cache=True, ram_suites=suites,
    )
    emb = extract_embeddings(model, ds, device)
    print(f"  extracted  : {emb['z'].shape[0]:,} embeddings of dimension {emb['z'].shape[1]}")
    print()

    domain = fit_domain_probe(
        emb["z"], emb["suite_id"], emb["simulation_id"], seed=args.seed
    )
    target = fit_target_probe(
        emb["z"], emb["target"], emb["suite_id"], emb["simulation_id"],
        list(target_scaler.names), seed=args.seed,
    )

    # ---- report ----------------------------------------------------------
    chance = domain["chance_accuracy"]
    print("  DOMAIN PROBE (section 37) - can simulator identity be recovered?")
    print(f"    chance = {chance:.3f}   train {domain['n_train']:,} / "
          f"test {domain['n_test']:,} maps, disjoint simulations")
    print(f"    {'probe':<10}{'accuracy':>11}{'balanced':>11}{'AUROC':>9}{'above chance':>15}")
    print("    " + "-" * 56)
    for name, p in domain["probes"].items():
        auroc = f"{p['auroc']:.4f}" if "auroc" in p else "n/a"
        print(f"    {name:<10}{p['accuracy']:>11.4f}{p['balanced_accuracy']:>11.4f}"
              f"{auroc:>9}{p['above_chance']*100:>14.1f}%")
    print()

    names = list(target_scaler.names)
    print("  TARGET PROBE (section 38) - is cosmology still accessible?")
    print(f"    {'probe':<10}" + "".join(f"{'R2 ' + n:>12}" for n in names))
    print("    " + "-" * (10 + 12 * len(names)))
    for name, p in target["probes"].items():
        print(f"    {name:<10}" + "".join(f"{p[n]['r2']:>12.4f}" for n in names))
    print()

    # ---- interpretation --------------------------------------------------
    best_dom = max(v["balanced_accuracy"] for v in domain["probes"].values())
    best_tgt = max(
        np.mean([v[n]["r2"] for n in names]) for v in target["probes"].values()
    )
    print("  READING (section 37: never interpret either number alone)")
    if best_dom > chance + 0.5 * (1 - chance):
        print(f"    Simulator identity is highly recoverable ({best_dom:.3f} vs "
              f"{chance:.3f} chance).")
        print("    H2 is supported: ordinary ERM encodes simulator information.")
    elif best_dom > chance + 0.1 * (1 - chance):
        print(f"    Simulator identity is somewhat recoverable ({best_dom:.3f}).")
    else:
        print(f"    Simulator identity is near chance ({best_dom:.3f}).")
        print("    This is NOT automatically good -- check the target probe below.")
    print(f"    Cosmological information remains {'strong' if best_tgt > 0.9 else 'partial' if best_tgt > 0.5 else 'WEAK'} "
          f"(mean R2 = {best_tgt:.3f}).")
    if best_dom < chance + 0.1 * (1 - chance) and best_tgt < 0.5:
        print("    WARNING: both low. This is the collapse failure mode, not invariance.")
    print()

    out = Path(args.out) if args.out else run_dir / f"probes_{args.split}.json"
    out.write_text(
        json.dumps(
            {
                "run": run_dir.name,
                "checkpoint": args.checkpoint,
                "epoch": int(state["epoch"]),
                "split": args.split,
                "suites": suites,
                "trained_on": record["protocol"]["source_suites"],
                "latent_dim": int(model.latent_dim),
                "domain_probe": domain,
                "target_probe": target,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
