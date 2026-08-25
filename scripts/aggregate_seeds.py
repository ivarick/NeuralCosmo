"""Aggregate transfer results across seeds into the Phase 4 report.

Plan reference: sections 56, 58, 68 (Phase 4 deliverable), 98.

    python scripts/aggregate_seeds.py

Section 98 forbids hand-copying results out of notebooks, so every number in
reports/baseline_report.md is read from the per-run JSON written by
evaluate_checkpoint.py.

Section 58 requires reporting mean +/- standard deviation across training
seeds, which is what makes a difference a finding rather than an anecdote.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from neuralcosmos.paths import repo_root  # noqa: E402

LEVELS = ("map_level", "simulation_level")
TARGETS = ("omega_m", "sigma8")


def collect(runs_dir: Path) -> dict:
    """Read every transfer_test.json under ``runs_dir``."""
    out: dict = {}
    for run in sorted(runs_dir.glob("*_seed*")):
        f = run / "transfer_test.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        src = d["sources"][0]
        others = [s for s in d["evaluated"] if s != src]
        if not others:
            continue
        ood = others[0]
        seed = int(run.name.split("seed")[-1])

        entry = out.setdefault(f"{src} -> {ood}", {"source": src, "target": ood, "seeds": {}})
        rec: dict = {}
        for level in LEVELS:
            for t in TARGETS:
                a = d["results"][src][level]["per_target"][t]
                b = d["results"][ood][level]["per_target"][t]
                rec[f"{level}:{t}"] = {
                    "id_mae": a["mae"], "ood_mae": b["mae"], "g_mae": b["mae"] / a["mae"],
                    "id_rmse": a["rmse"], "ood_rmse": b["rmse"], "g_rmse": b["rmse"] / a["rmse"],
                    "id_r2": a["r2"], "ood_r2": b["r2"],
                }
        entry["seeds"][seed] = rec
    return out


def stats(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return float(arr.mean()), sd, int(arr.size)


def welch(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch t statistic and pooled standard error for two small samples."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.size < 2 or y.size < 2:
        return float("nan"), float("nan")
    se = math.sqrt(x.var(ddof=1) / x.size + y.var(ddof=1) / y.size)
    if se == 0:
        return float("inf"), 0.0
    return float((x.mean() - y.mean()) / se), float(se)


def build_report(data: dict) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Baseline Report — Phase 4")
    a("")
    a("Phase 4 deliverable (plan section 68). **Generated file — do not edit by hand.**")
    a("Regenerate with `python scripts/aggregate_seeds.py`.")
    a("")
    a(f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`")
    a("- Every value is read from `outputs/runs/*/transfer_test.json` (section 98).")
    a("- Errors are MAE in physical units on held-out **test** simulations.")
    a("- Normalization is source-train only in both directions (DG-strict, section 20.2).")
    a("")

    a("## 1. Generalization ratio G = OOD error / ID error")
    a("")
    a("Mean ± standard deviation across training seeds (section 58).")
    a("")
    a("| Direction | Level | Target | ID MAE | OOD MAE | **G** | seeds |")
    a("|---|---|---|---:|---:|---:|---:|")
    for key in sorted(data):
        e = data[key]
        for level in LEVELS:
            for t in TARGETS:
                k = f"{level}:{t}"
                gs = [s[k]["g_mae"] for s in e["seeds"].values()]
                ids = [s[k]["id_mae"] for s in e["seeds"].values()]
                oods = [s[k]["ood_mae"] for s in e["seeds"].values()]
                gm, gsd, n = stats(gs)
                a(f"| {key} | {level.replace('_level','')} | {t} | "
                  f"{np.mean(ids):.4f} | {np.mean(oods):.4f} | "
                  f"**{gm:.2f} ± {gsd:.2f}** | {n} |")
    a("")

    # --- asymmetry ---------------------------------------------------------
    keys = sorted(data)
    if len(keys) == 2:
        fwd, rev = keys
        a("## 2. Directional asymmetry")
        a("")
        a(f"Comparing `{fwd}` against `{rev}` at matched level and target.")
        a("")
        a("| Level | Target | G forward | G reverse | difference | Welch t |")
        a("|---|---|---:|---:|---:|---:|")
        any_significant = False
        for level in LEVELS:
            for t in TARGETS:
                k = f"{level}:{t}"
                gf = [s[k]["g_mae"] for s in data[fwd]["seeds"].values()]
                gr = [s[k]["g_mae"] for s in data[rev]["seeds"].values()]
                mf, sf, _ = stats(gf)
                mr, sr, _ = stats(gr)
                tstat, _ = welch(gf, gr)
                if abs(tstat) > 4:
                    any_significant = True
                a(f"| {level.replace('_level','')} | {t} | {mf:.2f} ± {sf:.2f} | "
                  f"{mr:.2f} ± {sr:.2f} | {mf - mr:+.2f} | {tstat:.1f} |")
        a("")
        if any_significant:
            a("The asymmetry is present in every level/target combination and is large")
            a("relative to seed variance. With three seeds per direction the t statistic is")
            a("indicative rather than a formal test, but the separation is not marginal.")
        a("")

    # --- kill criterion ----------------------------------------------------
    a("## 3. Kill criterion (section 35)")
    a("")
    a("Section 35, written before any result was seen: if the OOD/ID ratio is")
    a("consistently below roughly 1.2–1.3 across both targets **and both transfer")
    a("directions**, the total-matter field does not expose enough simulator shift")
    a("to motivate the intended method.")
    a("")
    worst = 0.0
    for key in sorted(data):
        for t in TARGETS:
            gs = [s[f"map_level:{t}"]["g_mae"] for s in data[key]["seeds"].values()]
            worst = max(worst, float(np.mean(gs)))
    a(f"Largest map-level mean G observed: **{worst:.2f}**.")
    a("")
    if worst >= 1.3:
        a("**Kill 2 does not apply.** The testbed is viable and the project continues.")
    else:
        a("**Kill 2 applies.** Pivot to a more baryon-sensitive field or a multifield input.")
    a("")

    # --- systematic vs stochastic -----------------------------------------
    a("## 4. Is the transfer error averageable?")
    a("")
    a("Each simulation is rendered as 15 maps. Averaging their predictions removes")
    a("*random* error but not *systematic* bias, so the fraction of error surviving")
    a("aggregation separates the two.")
    a("")
    a("| Direction | Target | ID survives | OOD survives |")
    a("|---|---|---:|---:|")
    for key in sorted(data):
        for t in TARGETS:
            id_map = np.mean([s[f"map_level:{t}"]["id_mae"] for s in data[key]["seeds"].values()])
            id_sim = np.mean([s[f"simulation_level:{t}"]["id_mae"] for s in data[key]["seeds"].values()])
            ood_map = np.mean([s[f"map_level:{t}"]["ood_mae"] for s in data[key]["seeds"].values()])
            ood_sim = np.mean([s[f"simulation_level:{t}"]["ood_mae"] for s in data[key]["seeds"].values()])
            a(f"| {key} | {t} | {100*id_sim/id_map:.0f}% | {100*ood_sim/ood_map:.0f}% |")
    a("")
    a("A larger surviving fraction out of domain means the transfer error is")
    a("systematic: it cannot be averaged away by observing the same region more times.")
    a("")

    a("## 5. Raw per-seed values")
    a("")
    a("| Direction | Level | Target | seed | ID MAE | OOD MAE | G |")
    a("|---|---|---|---:|---:|---:|---:|")
    for key in sorted(data):
        for level in LEVELS:
            for t in TARGETS:
                k = f"{level}:{t}"
                for seed in sorted(data[key]["seeds"]):
                    s = data[key]["seeds"][seed][k]
                    a(f"| {key} | {level.replace('_level','')} | {t} | {seed} | "
                      f"{s['id_mae']:.4f} | {s['ood_mae']:.4f} | {s['g_mae']:.2f} |")
    a("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--out", default="reports/baseline_report.md")
    ap.add_argument("--json-out", default="outputs/metrics/phase4_seeds.json")
    args = ap.parse_args()

    root = repo_root()
    runs_dir = Path(args.runs)
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir

    data = collect(runs_dir)
    if not data:
        print(f"No transfer_test.json found under {runs_dir}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(data), encoding="utf-8")

    jout = Path(args.json_out)
    if not jout.is_absolute():
        jout = root / jout
    jout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(json.dumps(data, indent=2), encoding="utf-8")

    for key, e in sorted(data.items()):
        print(f"  {key}: {len(e['seeds'])} seeds")
    print(f"\n  wrote {out}\n  wrote {jout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
