"""Validate the local CAMELS archive before any model code runs.

Plan reference: sections 12, 13, 104 (step 3).

    python scripts/validate_data.py --config configs/data/mtot.yaml

Exits non-zero if any blocking check fails, so it can gate a pipeline.

The full scan reads every pixel of every configured suite (about 11.8 GB for
three suites) because positivity cannot be established by sampling, and
positivity decides whether the log transform of section 20.1 is legal. Use
--quick to skip the pixel scan during development; the report will say so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a plain script without installing the package.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neuralcosmos.data.manifest import build_manifest  # noqa: E402
from neuralcosmos.data.validate import Severity, validate_all  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402

_TICK = "PASS"
_CROSS = "FAIL"


def _fmt_bytes(n: int) -> str:
    gb = n / (1024**3)
    return f"{n:,} B ({gb:.2f} GiB)"


class _Progress:
    """Single-line progress for a long sequential read."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._last = 0.0
        self._t0 = time.time()
        self._suite: str | None = None

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
        elapsed = now - self._t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        bar_len = 28
        filled = int(bar_len * frac)
        bar = "#" * filled + "-" * (bar_len - filled)
        end = "\n" if done >= total else "\r"
        sys.stdout.write(
            f"    [{bar}] {frac*100:5.1f}%  {done:>6,}/{total:,} maps  "
            f"{rate:6.0f} maps/s  eta {eta:5.0f}s{end}"
        )
        sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--config",
        default="configs/data/mtot.yaml",
        help="data configuration YAML (default: configs/data/mtot.yaml)",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="override the archive root; otherwise CAMELS_DATA_ROOT is used",
    )
    ap.add_argument(
        "--suite",
        action="append",
        default=None,
        help="validate only this suite; repeatable. Default: every configured suite.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="skip the full pixel scan (shapes and parameters only)",
    )
    ap.add_argument(
        "--manifest",
        action="store_true",
        help="also write the local data manifest (section 13)",
    )
    ap.add_argument(
        "--manifest-out",
        default=None,
        help="manifest destination (default: <data_root>/manifests/local_manifest.json)",
    )
    ap.add_argument(
        "--json-out",
        default="reports/data_validation.json",
        help="machine-readable validation report destination",
    )
    ap.add_argument("--no-progress", action="store_true", help="suppress the progress bar")
    args = ap.parse_args()

    root_dir = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root_dir / config_path

    print("=" * 74)
    print("  NeuralCosmos - data validation (plan section 12)")
    print("=" * 74)
    print(f"  config    : {config_path}")

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(f"\n{_CROSS} {exc}", file=sys.stderr)
        return 2
    print(f"  data root : {data_root}")
    print(f"  mode      : {'QUICK (no pixel scan)' if args.quick else 'FULL pixel scan'}")
    print()

    progress = _Progress(enabled=not args.no_progress and not args.quick)
    t0 = time.time()
    report = validate_all(
        config_path=config_path,
        data_root=data_root,
        suites=args.suite,
        full_scan=not args.quick,
        progress=progress,
    )
    elapsed = time.time() - t0

    # --- human-readable output ---------------------------------------------
    for sv in report.suites:
        status = "OK" if sv.ok else "BLOCKED"
        print("-" * 74)
        print(f"  {sv.suite}  [{status}]")
        print("-" * 74)
        for c in sv.checks:
            mark = _TICK if c.passed else _CROSS
            tag = "" if c.passed else f" <{c.severity}>"
            print(f"    {mark}{tag:8s} {c.name:24s} {c.message}")
        if sv.stats is not None:
            s = sv.stats
            print()
            print(f"    pixel statistics over {s.count:,} values")
            print(f"      min  {s.minimum:.6g}")
            print(f"      max  {s.maximum:.6g}")
            print(f"      mean {s.mean:.6g}")
            print(f"      std  {s.std:.6g}")
            print(f"      NaN {s.n_nan:,}   Inf {s.n_inf:,}   "
                  f"<=0 {s.n_nonpositive:,}   ==0 {s.n_zero:,}")
        print()

    # --- machine-readable report -------------------------------------------
    json_path = Path(args.json_out)
    if not json_path.is_absolute():
        json_path = root_dir / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    doc = report.to_dict()
    doc["quick_mode"] = bool(args.quick)
    doc["elapsed_seconds"] = round(elapsed, 1)
    doc["data_root"] = str(data_root)
    doc["config"] = str(config_path)
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"  report written : {json_path}")

    # --- optional manifest --------------------------------------------------
    if args.manifest:
        manifest = build_manifest(config_path, data_root=data_root, suites=args.suite)
        if args.manifest_out:
            mpath = Path(args.manifest_out)
        else:
            mpath = data_root / "manifests" / "local_manifest.json"
        manifest.write(mpath)
        print(f"  manifest written : {mpath}")
        print(f"  manifest content hash : {manifest.content_hash()}")

    print(f"  elapsed : {elapsed:.1f}s")
    print()

    # --- verdict ------------------------------------------------------------
    warnings = [c for sv in report.suites for c in sv.checks if not c.passed and c.severity == Severity.WARN]
    if report.ok:
        print("=" * 74)
        print(f"  RESULT: PASS - {len(report.suites)} suite(s) validated"
              + (f", {len(warnings)} warning(s)" if warnings else ""))
        print("=" * 74)
        return 0

    print("=" * 74)
    print("  RESULT: FAIL - blocking checks failed. Do not train on this data.")
    print("=" * 74)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
