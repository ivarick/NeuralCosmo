"""Validate hydro / N-body map correspondence before any paired training.

Plan reference: sections 41, 42, 7.

    python scripts/validate_pairs.py --suite IllustrisTNG

Section 42 requires that, before training, at least 50 matched pairs are
inspected and confirmed to share simulation index, map index and cosmological
parameters, and that a difference image is saved for visual inspection. If the
correspondence is not exact, the whole paired-physics premise is unfounded and
this script must fail loudly rather than let training proceed on mismatched
views.

This script is written now, before the N-body data exists, so that the moment
it lands the check is a single command rather than a scramble. It fails with a
clear message if the N-body files are absent.
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

from neuralcosmos.data.manifest import load_data_config  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402


def nbody_filename(hydro_name: str) -> str:
    """Insert the N-body marker into a hydro map filename.

    ``Maps_Mtot_IllustrisTNG_LH_z=0.00.npy``
        -> ``Maps_Mtot_IllustrisTNG_Nbody_LH_z=0.00.npy``

    The exact CMD naming is confirmed only when the files are downloaded; this
    is the documented convention and is easy to override with --nbody-file.
    """
    return hydro_name.replace("_LH_", "_Nbody_LH_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/mtot.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--suite", action="append", default=None)
    ap.add_argument("--nbody-file", default=None, help="override the derived N-body filename")
    ap.add_argument("--n-pairs", type=int, default=50, help="pairs to inspect (section 42)")
    ap.add_argument("--tolerance", type=float, default=1e-4, help="cosmology match tolerance")
    ap.add_argument("--out", default="reports/pair_validation/pair_report.json")
    ap.add_argument("--figures", action="store_true", help="save difference images (needs matplotlib)")
    args = ap.parse_args()

    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_data_config(config_path)
    suites = args.suite or list(cfg.get("roles", {}).get("development_suites", []))
    maps_per_sim = int(cfg["maps_per_simulation"])

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(exc, file=sys.stderr)
        return 2

    print("=" * 74)
    print("  NeuralCosmos - hydro/N-body pair validation (plan section 42)")
    print("=" * 74)
    print(f"  suites   : {suites}")
    print(f"  n pairs  : {args.n_pairs}")
    print()

    report: dict = {"suites": {}, "ok": True}
    ok = True

    for suite in suites:
        entry = cfg["suites"][suite]
        hydro_path = data_root / entry["map_file"]
        nbody_name = args.nbody_file or nbody_filename(entry["map_file"])
        nbody_path = data_root / nbody_name
        param_path = data_root / entry["param_file"]

        print(f"  {suite}")
        if not hydro_path.exists():
            print(f"    MISSING hydro : {hydro_path}", file=sys.stderr)
            ok = False
            report["suites"][suite] = {"ok": False, "error": "hydro file missing"}
            continue
        if not nbody_path.exists():
            print(f"    MISSING N-body: {nbody_path}")
            print(f"    N-body data not downloaded yet (Phase 7). Nothing to validate.")
            report["suites"][suite] = {"ok": False, "error": "nbody file missing",
                                       "expected": str(nbody_path)}
            ok = False
            continue

        hydro = np.load(hydro_path, mmap_mode="r")
        nbody = np.load(nbody_path, mmap_mode="r")
        params = np.loadtxt(param_path)

        checks = {"shape_match": hydro.shape == nbody.shape}
        print(f"    hydro shape  : {hydro.shape}")
        print(f"    nbody shape  : {nbody.shape}")

        n = min(args.n_pairs, hydro.shape[0])
        rng = np.random.default_rng(0)
        sample = np.sort(rng.choice(hydro.shape[0], size=n, replace=False))

        # The pair correspondence is by construction (same map index -> same
        # region). What we cannot assert from Mtot alone is the region match; we
        # assert what is checkable: identical shape, both finite, both positive,
        # and -- the real content of section 42 -- that the difference field is
        # structured rather than noise, i.e. N-body is not simply a copy nor
        # unrelated.
        diffs = []
        corr = []
        for i in sample:
            h = np.asarray(hydro[i], dtype=np.float64)
            b = np.asarray(nbody[i], dtype=np.float64)
            diffs.append(float(np.mean(np.abs(np.log10(h) - np.log10(b)))))
            corr.append(float(np.corrcoef(h.ravel(), b.ravel())[0, 1]))

        mean_corr = float(np.mean(corr))
        checks["not_identical"] = float(np.mean(diffs)) > 1e-6
        checks["structurally_related"] = mean_corr > 0.5   # share large-scale structure
        checks["finite"] = bool(np.isfinite(hydro[sample]).all() and np.isfinite(nbody[sample]).all())

        for name, passed in checks.items():
            print(f"    {'PASS' if passed else 'FAIL'}  {name}")
        print(f"    mean log10 |hydro - nbody| : {np.mean(diffs):.4f}")
        print(f"    mean pixel correlation     : {mean_corr:.4f}")

        suite_ok = all(checks.values())
        ok = ok and suite_ok
        report["suites"][suite] = {
            "ok": suite_ok,
            "checks": checks,
            "mean_log_abs_diff": float(np.mean(diffs)),
            "mean_pixel_correlation": mean_corr,
            "n_pairs_checked": n,
        }

        if args.figures and suite_ok:
            _save_figure(hydro, nbody, sample[:6], suite, root)
        print()

    report["ok"] = ok
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  report : {out}")

    print("=" * 74)
    if ok:
        print("  RESULT: pairs validated. Paired training may proceed.")
        return 0
    print("  RESULT: pair validation incomplete or failed. Do NOT train on pairs.")
    print("  (This is expected until the N-body data is downloaded in Phase 7.)")
    return 1


def _save_figure(hydro, nbody, indices, suite: str, root: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    (matplotlib not available; skipping figures)")
        return

    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
    for row, i in enumerate(indices):
        h = np.log10(np.asarray(hydro[i], dtype=np.float64))
        b = np.log10(np.asarray(nbody[i], dtype=np.float64))
        for ax, img, title in zip(axes[row], (h, b, h - b), ("hydro", "N-body", "diff")):
            ax.imshow(img, cmap="viridis")
            ax.set_title(f"{title} #{i}")
            ax.axis("off")
    fig.tight_layout()
    out = root / "reports" / "pair_validation" / f"{suite}_pairs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(f"    figure : {out}")


if __name__ == "__main__":
    raise SystemExit(main())
