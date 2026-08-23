"""Generate reports/data_report.md from the validation artefacts.

Plan reference: sections 68 (Phase 1 deliverable), 98.

Every number in the report is read from reports/data_validation.json and the
local manifest. Nothing is typed by hand, so the report cannot drift away from
what the validator actually measured.

    python scripts/make_data_report.py
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

from neuralcosmos.paths import repo_root  # noqa: E402


def _sci(x: float, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}e}"


def _gib(n: int) -> str:
    return f"{n / 1024**3:.2f}"


def build(validation: dict, manifest: dict | None) -> str:
    suites = validation["suites"]
    quick = validation.get("quick_mode", False)
    lines: list[str] = []
    a = lines.append

    a("# Data Report")
    a("")
    a("Phase 1 deliverable (plan section 68). **Generated file - do not edit by hand.**")
    a("Regenerate with `python scripts/make_data_report.py`.")
    a("")
    a(f"- Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`")
    a(f"- Validation mode: `{'QUICK (no pixel scan)' if quick else 'FULL pixel scan'}`")
    a(f"- Validation elapsed: `{validation.get('elapsed_seconds', 'n/a')} s`")
    if manifest:
        a(f"- Manifest content hash: `{manifest.get('manifest_content_hash', 'n/a')}`")
    a(f"- Overall verdict: **{'PASS' if validation['ok'] else 'FAIL'}**")
    a("")

    # --- files ------------------------------------------------------------
    a("## 1. Files")
    a("")
    a("| Suite | Map file | Bytes | GiB | Shape | dtype | Params |")
    a("|---|---|---:|---:|---|---|---|")
    by_suite = {m["suite"]: m for m in (manifest or {}).get("suites", [])}
    for s in suites:
        info = s.get("info", {})
        m = by_suite.get(s["suite"], {})
        nbytes = info.get("map_bytes", m.get("map_bytes", 0))
        a(
            f"| {s['suite']} | `{m.get('map_file', '?')}` | {nbytes:,} | {_gib(nbytes)} | "
            f"{tuple(info.get('maps_shape', []))} | {info.get('maps_dtype', '?')} | "
            f"{tuple(info.get('params_shape', []))} |"
        )
    a("")

    # --- checks -----------------------------------------------------------
    a("## 2. Validation checks")
    a("")
    total = sum(len(s["checks"]) for s in suites)
    failed = sum(1 for s in suites for c in s["checks"] if not c["passed"])
    a(f"{total - failed}/{total} checks passed across {len(suites)} suite(s).")
    a("")
    for s in suites:
        a(f"### {s['suite']} - {'OK' if s['ok'] else 'BLOCKED'}")
        a("")
        a("| Check | Result | Detail |")
        a("|---|---|---|")
        for c in s["checks"]:
            mark = "PASS" if c["passed"] else f"**FAIL ({c['severity']})**"
            a(f"| `{c['name']}` | {mark} | {c['message']} |")
        a("")

    # --- pixel statistics -------------------------------------------------
    have_stats = any(s.get("pixel_stats") for s in suites)
    if have_stats:
        a("## 3. Pixel statistics")
        a("")
        a("Computed over every pixel of every map. These are **integrity statistics only**.")
        a("They must never be used for normalization: section 20.2 requires normalization")
        a("statistics computed from source-training simulations alone.")
        a("")
        a("| Suite | N pixels | min | max | mean | std | NaN | Inf | <=0 | ==0 |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for s in suites:
            st = s.get("pixel_stats")
            if not st:
                continue
            a(
                f"| {s['suite']} | {st['count']:,} | {_sci(st['min'])} | {_sci(st['max'])} | "
                f"{_sci(st['mean'])} | {_sci(st['std'])} | {st['n_nan']:,} | {st['n_inf']:,} | "
                f"{st['n_nonpositive']:,} | {st['n_zero']:,} |"
            )
        a("")

        # --- the section 20.1 decision ------------------------------------
        a("### 3.1 Log-transform decision (section 20.1)")
        a("")
        all_positive = all(
            s["pixel_stats"]["n_nonpositive"] == 0 for s in suites if s.get("pixel_stats")
        )
        if all_positive:
            mins = [s["pixel_stats"]["min"] for s in suites if s.get("pixel_stats")]
            maxs = [s["pixel_stats"]["max"] for s in suites if s.get("pixel_stats")]
            decades = math.log10(max(maxs) / min(mins))
            a(f"All pixel values are strictly positive (smallest observed: {_sci(min(mins))}).")
            a("")
            a("Section 20.1 permits `log(x)` in this case. No epsilon is introduced and")
            a("`log1p` is **not** used: the plan warns that the `log(1 + rho*)` treatment")
            a("documented for stellar density must not be transferred to Mtot without reason.")
            a("")
            a(f"The dynamic range spans about **{decades:.1f} decades** "
              f"({_sci(min(mins))} to {_sci(max(maxs))}), which is why the transform matters.")
        else:
            a("**Non-positive values are present.** Section 20.1 requires stopping here to")
            a("inspect data semantics. Do not invent an epsilon and do not enable the log")
            a("transform until this is resolved.")
        a("")

        # --- cross-suite comparison ---------------------------------------
        a("### 3.2 Cross-suite comparison")
        a("")
        stats = [(s["suite"], s["pixel_stats"]) for s in suites if s.get("pixel_stats")]
        if len(stats) > 1:
            means = [st["mean"] for _, st in stats]
            stds = [st["std"] for _, st in stats]
            mean_spread = (max(means) - min(means)) / (sum(means) / len(means))
            std_spread = (max(stds) - min(stds)) / (sum(stds) / len(stds))
            a(f"- Relative spread of the global mean across suites: **{mean_spread:.2e}**")
            a(f"- Relative spread of the global std across suites:  **{std_spread:.2e}**")
            a("")
            a("A negligible spread in the mean is expected rather than surprising: the mean")
            a("surface density of total matter is fixed by Omega_m and the box geometry, and")
            a("all suites sample the same Latin-hypercube design over Omega_m. Baryonic")
            a("feedback redistributes matter; it does not create or destroy it.")
            a("")
            a("Two consequences follow:")
            a("")
            a("1. The DG-strict normalization of section 20.2 (source statistics applied to")
            a("   the target) costs almost nothing here, because the global statistics barely")
            a("   differ between suites. The methodologically strict choice is also cheap.")
            a("2. Any cross-suite domain shift must live in **spatial structure and tails**,")
            a("   not in global moments. The B0 summary-statistic baseline (section 25) should")
            a("   therefore be expected to show weak suite separation, and quantifying the")
            a("   shift properly requires the per-map analysis of Phase 4.")
            a("")
            a("Extreme order statistics (the per-suite maximum) are **not** interpreted here.")
            a("A maximum over ~10^9 pixels is set by a single densest cell and is far too")
            a("fragile to support a claim about feedback physics.")
        a("")
    else:
        a("## 3. Pixel statistics")
        a("")
        a("Not computed: the validator ran in `--quick` mode. Positivity is therefore")
        a("**unverified** and the log transform must not be enabled on this evidence.")
        a("")

    # --- provenance --------------------------------------------------------
    a("## 4. Integrity record")
    a("")
    if manifest:
        a("Local digests. Section 13: these are **our own** integrity record. CAMELS does")
        a("not publish matching checksums, so they must never be presented as official.")
        a("The map digest samples the file rather than reading it whole; it is designed to")
        a("catch truncation and accidental modification, not adversarial tampering.")
        a("")
        a("| Suite | sampled map sha256 | full params sha256 |")
        a("|---|---|---|")
        for m in manifest.get("suites", []):
            a(
                f"| {m['suite']} | `{m['sha256_sampled_map'][:16]}...` | "
                f"`{m['sha256_full_params'][:16]}...` |"
            )
    else:
        a("No manifest found. Regenerate with `--manifest`.")
    a("")

    a("## 5. Reproduce")
    a("")
    a("```bash")
    a("python scripts/validate_data.py --config configs/data/mtot.yaml --manifest")
    a("python scripts/make_data_report.py")
    a("```")
    a("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validation", default="reports/data_validation.json")
    ap.add_argument("--manifest", default=None, help="path to local_manifest.json")
    ap.add_argument("--out", default="reports/data_report.md")
    args = ap.parse_args()

    root = repo_root()

    vpath = Path(args.validation)
    if not vpath.is_absolute():
        vpath = root / vpath
    if not vpath.exists():
        print(f"Validation report not found: {vpath}\nRun scripts/validate_data.py first.",
              file=sys.stderr)
        return 2
    validation = json.loads(vpath.read_text(encoding="utf-8"))

    manifest = None
    mpath = Path(args.manifest) if args.manifest else None
    if mpath is None:
        # Default location written by validate_data.py --manifest.
        root_dir = validation.get("data_root")
        if root_dir:
            mpath = Path(root_dir) / "manifests" / "local_manifest.json"
    if mpath and mpath.exists():
        manifest = json.loads(mpath.read_text(encoding="utf-8"))

    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(validation, manifest), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
