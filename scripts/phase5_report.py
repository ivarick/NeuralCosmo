"""Aggregate the Phase 5 baselines into the section 40 trade-off.

Plan reference: sections 37, 40, 68 (Phase 5), 98.

    python scripts/phase5_report.py

Section 40 asks for the figure that makes the invariance trade-off visible:
each method as one point in

    x = simulator probe accuracy      (how much domain information survives)
    y = cosmological error            (what the task costs)

Ideal methods move down and left. A method that moves left while moving UP has
removed simulator information by destroying task information, which section 37
warns is the failure mode a domain probe cannot detect on its own.

Every number is read from the per-run JSON, never typed by hand (section 98).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from neuralcosmos.paths import repo_root  # noqa: E402

# Display order and labels. Section 29/31 note these are baselines, not
# contributions, and the report says so wherever it is read.
METHOD_LABELS = {
    "erm": "ERM (no invariance)",
    "dann": "DANN (adversarial)",
    "coral": "CORAL (2nd-order)",
    "mmd": "MMD (distribution)",
    "miest_like": "MIEST-comparable",
}


def collect(runs_dir: Path) -> dict:
    out: dict = {}
    for run in sorted(runs_dir.glob("E2*_seed*")):
        rj = run / "run.json"
        if not rj.exists():
            continue
        record = json.loads(rj.read_text(encoding="utf-8"))

        cfg_path = run / "config_snapshot.yaml"
        method = "erm"
        if cfg_path.exists():
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            method = str(cfg.get("method", {}).get("name", "erm"))

        entry: dict = {
            "run": run.name,
            "method": method,
            "best_epoch": record["summary"]["best_epoch"],
            "selection_score": record["summary"]["best_selection_score"],
            "epochs_run": record["summary"]["epochs_run"],
            "train_maps": record["train_maps"],
            "seed": record["seed"],
        }

        ev = run / "evaluation.json"
        if ev.exists():
            doc = json.loads(ev.read_text(encoding="utf-8"))
            per_suite: dict[str, dict] = {}
            for key, r in doc.items():
                if not key.endswith("_test"):
                    continue
                suite = key.rsplit("_", 1)[0]
                per_suite[suite] = {
                    t: r["map_level"]["per_target"][t]["mae"]
                    for t in r["map_level"]["per_target"]
                }
            entry["source_test_mae"] = per_suite
            if per_suite:
                targets = list(next(iter(per_suite.values())))
                entry["mean_source_mae"] = {
                    t: float(np.mean([v[t] for v in per_suite.values()])) for t in targets
                }

        pj = run / "probes_test.json"
        if pj.exists():
            probes = json.loads(pj.read_text(encoding="utf-8"))
            dom = probes["domain_probe"]
            entry["domain_probe"] = {
                "chance": dom["chance_accuracy"],
                "linear": dom["probes"]["linear"]["balanced_accuracy"],
                "mlp": dom["probes"]["mlp"]["balanced_accuracy"],
                "best": max(v["balanced_accuracy"] for v in dom["probes"].values()),
            }
            tgt = probes["target_probe"]["probes"]["linear"]
            entry["target_probe_r2"] = {t: tgt[t]["r2"] for t in tgt}

        out[run.name] = entry
    return out


def build_report(data: dict) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Phase 5 Report — Domain-Generalization Baselines")
    a("")
    a("Generated file — do not edit by hand. `python scripts/phase5_report.py`.")
    a("")
    a(f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`")
    a("- Sources: IllustrisTNG + SIMBA. Astrid sealed and untouched.")
    a("- Every method shares the same backbone, split, budget and metric (section 5).")
    a("")
    a("**None of these methods is a contribution.** MIEST already applies adversarial")
    a("de-classification to CAMELS and DA-GNN already applies MMD; sections 29 and 31")
    a("classify both as baselines. They exist to establish what the proposed method")
    a("must beat.")
    a("")

    a("## 1. What Phase 5 can and cannot show")
    a("")
    a("With only two development suites and Astrid sealed, the out-of-distribution")
    a("benefit of a multi-source method **cannot be measured before the freeze**.")
    a("What follows is therefore source-side evidence only:")
    a("")
    a("- source-test error — does the method damage the task?")
    a("- domain probe (section 37) — does it actually remove simulator information?")
    a("- the trade-off of section 40 — both together, which is the only honest reading")
    a("")
    a("Section 52 freezes the design on this evidence and evaluates Astrid once.")
    a("")

    if not data:
        a("_No Phase 5 runs found._")
        return "\n".join(lines)

    rows = sorted(data.values(), key=lambda r: list(METHOD_LABELS).index(r["method"])
                  if r["method"] in METHOD_LABELS else 99)
    targets = sorted(next((r["mean_source_mae"] for r in rows if "mean_source_mae" in r), {}))

    a("## 2. Source performance")
    a("")
    header = "| Method | best epoch | selection score |"
    sep = "|---|---:|---:|"
    for t in targets:
        header += f" source MAE {t} |"
        sep += "---:|"
    a(header)
    a(sep)
    for r in rows:
        line = (f"| {METHOD_LABELS.get(r['method'], r['method'])} | {r['best_epoch']} | "
                f"{r['selection_score']:.5f} |")
        for t in targets:
            v = r.get("mean_source_mae", {}).get(t)
            line += f" {v:.4f} |" if v is not None else " — |"
        a(line)
    a("")

    have_probe = [r for r in rows if "domain_probe" in r]
    if have_probe:
        chance = have_probe[0]["domain_probe"]["chance"]
        a("## 3. The section 40 trade-off")
        a("")
        a(f"Domain probe chance level is {chance:.3f}. Lower probe accuracy means less")
        a("simulator information survives; lower source error means less task damage.")
        a("")
        a("| Method | domain probe | vs chance | source MAE (mean) | target probe R² |")
        a("|---|---:|---:|---:|---:|")
        for r in have_probe:
            probe = r["domain_probe"]["best"]
            src = np.mean(list(r.get("mean_source_mae", {}).values())) if r.get("mean_source_mae") else float("nan")
            tp = r.get("target_probe_r2", {})
            tp_mean = np.mean(list(tp.values())) if tp else float("nan")
            a(f"| {METHOD_LABELS.get(r['method'], r['method'])} | {probe:.3f} | "
              f"{probe - chance:+.3f} | {src:.4f} | {tp_mean:.3f} |")
        a("")

        # --- reading -------------------------------------------------------
        erm = next((r for r in have_probe if r["method"] == "erm"), None)
        if erm:
            a("### Reading")
            a("")
            e_probe = erm["domain_probe"]["best"]
            e_err = np.mean(list(erm["mean_source_mae"].values()))
            for r in have_probe:
                if r["method"] == "erm":
                    continue
                p = r["domain_probe"]["best"]
                s = np.mean(list(r.get("mean_source_mae", {}).values()))
                dp = p - e_probe
                ds = s - e_err
                label = METHOD_LABELS.get(r["method"], r["method"])
                if dp < -0.01 and ds < 0.001:
                    verdict = "less simulator information at no task cost"
                elif dp < -0.01:
                    verdict = (f"removed simulator information but cost "
                               f"{100 * ds / e_err:+.1f}% source error")
                elif dp > 0.01:
                    verdict = "did NOT reduce simulator information"
                else:
                    verdict = "no meaningful change in simulator information"
                a(f"- **{label}**: probe {dp:+.3f}, source error {100 * ds / e_err:+.1f}% — {verdict}")
            a("")
            a("Section 37 is explicit that a lower probe score is not automatically")
            a("better: a collapsed representation hides the simulator perfectly while")
            a("being useless. Any method that reduced the probe while raising source")
            a("error has bought invariance with task information, and the target-probe")
            a("column is where that shows up.")
            a("")

    a("## 4. Per-run detail")
    a("")
    a("| Run | method | seed | epochs | train maps |")
    a("|---|---|---:|---:|---:|")
    for r in rows:
        a(f"| `{r['run']}` | {r['method']} | {r['seed']} | {r['epochs_run']} | "
          f"{r['train_maps']:,} |")
    a("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="outputs/runs")
    ap.add_argument("--out", default="reports/phase5_report.md")
    ap.add_argument("--json-out", default="outputs/metrics/phase5.json")
    args = ap.parse_args()

    root = repo_root()
    runs_dir = Path(args.runs)
    if not runs_dir.is_absolute():
        runs_dir = root / runs_dir

    data = collect(runs_dir)
    out = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(data), encoding="utf-8")

    jout = Path(args.json_out) if Path(args.json_out).is_absolute() else root / args.json_out
    jout.parent.mkdir(parents=True, exist_ok=True)
    jout.write_text(json.dumps(data, indent=2), encoding="utf-8")

    for name, r in data.items():
        probe = r.get("domain_probe", {}).get("best")
        print(f"  {name:<28} {r['method']:<12} "
              f"probe={probe:.3f}" if probe else f"  {name:<28} {r['method']:<12} probe=n/a")
    print(f"\n  wrote {out}\n  wrote {jout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
