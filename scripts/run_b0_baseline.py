"""B0: can simple summary statistics do the job?

Plan reference: sections 25, 34, 35.

    python scripts/run_b0_baseline.py --source IllustrisTNG --target SIMBA

Section 25: deep learning must beat something simple. This fits linear and
gradient-boosted regressors on per-map summary features, in-domain and across
suites, under the same simulation-level split and the same DG-strict rule that
the CNN uses -- features are standardised with source-training statistics only.

Three feature sets are compared so the result is mechanistic rather than merely
a gate:

  moments   permutation-invariant; no spatial information at all
  spectrum  binned P(k); pure two-point spatial information
  both      concatenation

If the CNN barely beats ``both``, its advantage is not in higher-order spatial
structure and the framing of the project would need revisiting.
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

from neuralcosmos.data.cache import QuantSpec, load_into_ram  # noqa: E402
from neuralcosmos.data.manifest import load_data_config, resolve_suite_files  # noqa: E402
from neuralcosmos.data.splits import load_split_file, maps_for_simulations  # noqa: E402
from neuralcosmos.data.targets import TargetScaler  # noqa: E402
from neuralcosmos.evaluation.metrics import generalization_ratio, regression_metrics  # noqa: E402
from neuralcosmos.models.summary_stats import FEATURE_SETS, extract_features  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402
from neuralcosmos.protocol import ProtocolViolation, default_dg_protocol  # noqa: E402


def load_split_features(
    cfg, data_root, split_file, suite, split, feature_set, max_sims, ram, cache, progress=None
):
    """Feature matrix and physical targets for one suite and split."""
    sf = resolve_suite_files(cfg, data_root, [suite])[0]
    maps_per_sim = int(cfg["maps_per_simulation"])
    sim_ids = list(split_file.suite(suite).ids(split))
    if max_sims is not None and split == "train":
        sim_ids = sim_ids[:max_sims]
    idx = maps_for_simulations(sim_ids, maps_per_sim)

    if ram:
        key = str(sf.map_path)
        if key not in cache:
            cache[key] = load_into_ram(sf.map_path, QuantSpec())
        maps, spec = cache[key], QuantSpec()
    else:
        maps, spec = np.load(sf.map_path, mmap_mode="r"), None

    X = extract_features(maps, idx, quant_spec=spec, feature_set=feature_set, progress=progress)
    params = np.loadtxt(sf.param_path)
    y = params[idx // maps_per_sim][:, :2]
    sims = idx // maps_per_sim
    return X, y, sims


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/mtot.yaml")
    ap.add_argument("--split-file", default="configs/splits/split_v1.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--source", default="IllustrisTNG")
    ap.add_argument("--target", default="SIMBA")
    ap.add_argument("--max-simulations", type=int, default=300,
                    help="training simulations to use; features are slow to extract")
    ap.add_argument("--feature-set", action="append", default=None)
    ap.add_argument("--no-ram", action="store_true")
    ap.add_argument("--out", default="reports/b0_baseline.json")
    args = ap.parse_args()

    root = repo_root()

    def _p(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else root / p

    cfg = load_data_config(_p(args.config))
    split_file = load_split_file(_p(args.split_file))
    scaler = TargetScaler.from_config(cfg)
    names = list(scaler.names)
    spans = [hi - lo for lo, hi in zip(scaler.lower, scaler.upper)]
    feature_sets = args.feature_set or list(FEATURE_SETS)

    try:
        data_root = resolve_data_root(args.data_root)
        protocol = default_dg_protocol([args.source], [args.target])
    except (DataRootNotFound, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2

    print("=" * 74)
    print("  NeuralCosmos - B0 summary-statistic baseline (plan section 25)")
    print("=" * 74)
    print(f"  {protocol.describe()}")
    print(f"  train sims : {args.max_simulations}")
    print(f"  features   : {feature_sets}")
    print()

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    cache: dict = {}
    ram = not args.no_ram
    results: dict = {}

    for fs in feature_sets:
        print(f"  --- feature set: {fs} ---")
        t0 = time.time()
        Xtr, ytr, _ = load_split_features(
            cfg, data_root, split_file, args.source, "train", fs,
            args.max_simulations, ram, cache,
        )
        Xid, yid, sid = load_split_features(
            cfg, data_root, split_file, args.source, "test", fs, None, ram, cache
        )
        Xood, yood, sood = load_split_features(
            cfg, data_root, split_file, args.target, "test", fs, None, ram, cache
        )
        print(f"      features: {Xtr.shape[1]}  train {Xtr.shape[0]:,}  "
              f"({time.time() - t0:.0f}s)")

        # DG-strict: standardise with SOURCE-TRAIN statistics only, applied
        # unchanged to the target. Refitting on the target would be adaptation.
        fs_scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xid_s, Xood_s = (fs_scaler.transform(a) for a in (Xtr, Xid, Xood))

        models = {
            "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
            "gbt": HistGradientBoostingRegressor(max_iter=300, random_state=0),
        }
        results[fs] = {"n_features": int(Xtr.shape[1])}

        for mname, proto in models.items():
            preds_id, preds_ood = [], []
            for j in range(len(names)):
                m = (
                    HistGradientBoostingRegressor(max_iter=300, random_state=0)
                    if mname == "gbt"
                    else RidgeCV(alphas=np.logspace(-3, 3, 13))
                )
                m.fit(Xtr_s, ytr[:, j])
                preds_id.append(m.predict(Xid_s))
                preds_ood.append(m.predict(Xood_s))
            pid = np.stack(preds_id, axis=1)
            pood = np.stack(preds_ood, axis=1)

            mid = regression_metrics(yid, pid, names, spans=spans)
            mood = regression_metrics(yood, pood, names, spans=spans)
            results[fs][mname] = {
                "in_domain": mid,
                "transfer": mood,
                "g": {
                    n: generalization_ratio(
                        mood["per_target"][n]["mae"], mid["per_target"][n]["mae"]
                    )
                    for n in names
                },
            }
            print(f"      {mname:<6} ID  MAE "
                  + "  ".join(f"{n}={mid['per_target'][n]['mae']:.4f}" for n in names)
                  + "   R2 "
                  + "  ".join(f"{mid['per_target'][n]['r2']:.3f}" for n in names))
            print(f"      {'':<6} OOD MAE "
                  + "  ".join(f"{n}={mood['per_target'][n]['mae']:.4f}" for n in names)
                  + "   G "
                  + "  ".join(f"{results[fs][mname]['g'][n]:.2f}" for n in names))
        del proto, models
        print()

    # ---- comparison against the CNN ---------------------------------------
    cnn_path = root / "outputs" / "metrics" / "phase4_seeds.json"
    print("  " + "=" * 70)
    print("  B0 versus the CNN (section 25: deep learning must beat simple)")
    print("  " + "=" * 70)
    if cnn_path.exists():
        seeds = json.loads(cnn_path.read_text(encoding="utf-8"))
        key = f"{args.source} -> {args.target}"
        if key in seeds:
            per = seeds[key]["seeds"]
            cnn_id = {n: float(np.mean([s[f"map_level:{n}"]["id_mae"] for s in per.values()]))
                      for n in names}
            cnn_ood = {n: float(np.mean([s[f"map_level:{n}"]["ood_mae"] for s in per.values()]))
                       for n in names}
            print(f"    {'model':<20}" + "".join(f"{'ID ' + n:>14}" for n in names)
                  + "".join(f"{'OOD ' + n:>15}" for n in names))
            print("    " + "-" * (20 + 29 * len(names)))
            for fs in feature_sets:
                for mname in ("ridge", "gbt"):
                    r = results[fs][mname]
                    print(f"    {'B0 ' + fs + '/' + mname:<20}"
                          + "".join(f"{r['in_domain']['per_target'][n]['mae']:>14.4f}" for n in names)
                          + "".join(f"{r['transfer']['per_target'][n]['mae']:>15.4f}" for n in names))
            print(f"    {'CNN (3 seeds)':<20}"
                  + "".join(f"{cnn_id[n]:>14.4f}" for n in names)
                  + "".join(f"{cnn_ood[n]:>15.4f}" for n in names))
            print()
            best_b0_id = min(
                results[fs][m]["in_domain"]["per_target"][names[0]]["mae"]
                for fs in feature_sets for m in ("ridge", "gbt")
            )
            factor = best_b0_id / cnn_id[names[0]]
            print(f"    In-domain {names[0]}: the CNN is {factor:.1f}x better than the best B0.")
            if factor < 1.5:
                print("    WARNING: the margin is small. Section 25's premise -- that spatial")
                print("    representation learning is doing real work -- is not well supported.")
            results["cnn_reference"] = {"id_mae": cnn_id, "ood_mae": cnn_ood}
    else:
        print("    No CNN reference found; run scripts/aggregate_seeds.py first.")
    print()

    out = _p(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"source": args.source, "target": args.target,
         "max_simulations": args.max_simulations, "results": results}, indent=2
    ), encoding="utf-8")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
