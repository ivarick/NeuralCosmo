"""Compute normalization statistics from source-training simulations only.

Plan reference: sections 20.2, 21, 67.

    python scripts/compute_stats.py --source IllustrisTNG --source SIMBA

Writes configs/normalizers/<name>.json, which is committed so that every run
can be traced to the exact statistics it used.

The script constructs an ExperimentProtocol from the declared sources and
targets and runs the checks before reading a single pixel, so asking it to
include a sealed target suite fails immediately rather than producing a
plausible-looking but invalid normalizer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neuralcosmos.data.builders import make_provenance, stats_sources  # noqa: E402
from neuralcosmos.data.dataset import LogNormalizer  # noqa: E402
from neuralcosmos.data.manifest import load_data_config  # noqa: E402
from neuralcosmos.data.splits import load_split_file  # noqa: E402
from neuralcosmos.data.statistics import compute_log_stats, save_normalizer  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402
from neuralcosmos.protocol import ProtocolViolation, default_dg_protocol  # noqa: E402


class _Progress:
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
            sys.stdout.write(f"    {suite}\n")
        now = time.time()
        if done < total and (now - self._last) < 0.5:
            return
        self._last = now
        frac = done / total if total else 1.0
        elapsed = now - self._t0
        rate = done / elapsed if elapsed > 0 else 0.0
        bar = "#" * int(28 * frac) + "-" * (28 - int(28 * frac))
        end = "\n" if done >= total else "\r"
        sys.stdout.write(
            f"      [{bar}] {frac*100:5.1f}%  {done:>6,}/{total:,} maps  {rate:6.0f} maps/s{end}"
        )
        sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/mtot.yaml")
    ap.add_argument("--split-file", default="configs/splits/split_v1.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--source",
        action="append",
        default=None,
        help="source suite; repeatable. Default: the config's development_suites.",
    )
    ap.add_argument(
        "--target",
        action="append",
        default=None,
        help="sealed target suite; repeatable. Default: the config's sealed_target_suites.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--max-maps-per-suite",
        type=int,
        default=None,
        help="deterministically subsample for speed; recorded in the output",
    )
    ap.add_argument(
        "--max-simulations",
        type=int,
        default=None,
        help="use only the first N training simulations per suite (smoke runs)",
    )
    ap.add_argument("--no-log", action="store_true", help="skip the log10 transform")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing normalizer")
    args = ap.parse_args()

    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_data_config(config_path)

    split_path = Path(args.split_file)
    if not split_path.is_absolute():
        split_path = root / split_path
    if not split_path.exists():
        print(f"Split file not found: {split_path}\nRun scripts/build_splits.py first.",
              file=sys.stderr)
        return 2
    split_file = load_split_file(split_path)

    roles = cfg.get("roles", {})
    sources = args.source or list(roles.get("development_suites", []))
    targets = args.target if args.target is not None else list(roles.get("sealed_target_suites", []))
    if not sources:
        print("No source suites given and none configured.", file=sys.stderr)
        return 2

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    try:
        protocol = default_dg_protocol(sources, targets)
    except ValueError as exc:
        print(f"PROTOCOL ERROR\n  {exc}", file=sys.stderr)
        print(
            "\n  If you genuinely intend to compute statistics over a suite that is "
            "\n  sealed elsewhere, pass --target explicitly to redeclare the roles for "
            "\n  this run, and be sure the experiment is reported accordingly "
            "\n  (plan sections 19, 21).",
            file=sys.stderr,
        )
        return 4

    out = Path(args.out) if args.out else (
        root / "configs" / "normalizers" / f"norm_{'-'.join(sorted(sources))}_{split_file.version}.json"
    )
    if not out.is_absolute():
        out = root / out
    if out.exists() and not args.force:
        print(f"Normalizer already exists: {out}\nPass --force to recompute.", file=sys.stderr)
        return 3

    print("=" * 74)
    print("  NeuralCosmos - normalization statistics (plan section 20.2)")
    print("=" * 74)
    print(f"  {protocol.describe()}")
    print(f"  split      : {split_path.name} ({split_file.content_hash()[:12]})")
    print(f"  transform  : {'none' if args.no_log else 'log10'}")
    print("  data used  : source TRAINING simulations only")
    print()

    try:
        triples = stats_sources(
            cfg,
            data_root,
            split_file,
            sources,
            protocol=protocol,
            max_simulations=args.max_simulations,
        )
    except ProtocolViolation as exc:
        print(f"PROTOCOL VIOLATION\n{exc}", file=sys.stderr)
        return 4

    t0 = time.time()
    acc, meta = compute_log_stats(
        triples,
        log_transform=not args.no_log,
        max_maps_per_suite=args.max_maps_per_suite,
        progress=_Progress(enabled=not args.no_progress),
    )
    elapsed = time.time() - t0

    provenance = make_provenance(
        sources,
        split_file,
        split="train",
        log_transform=not args.no_log,
        extra=f"n={acc.count}",
    )
    protocol.check_normalizer(provenance)  # belt and braces

    normalizer = LogNormalizer(mean=acc.mean, std=acc.std, provenance=provenance)
    save_normalizer(out, normalizer, acc, meta)

    print()
    print(f"  values     : {acc.count:,}")
    print(f"  mean       : {acc.mean:.6f}")
    print(f"  std        : {acc.std:.6f}")
    print(f"  min / max  : {acc.minimum:.6f} / {acc.maximum:.6f}")
    print(f"  elapsed    : {elapsed:.1f}s")
    print()
    for s in meta["suites"]:
        note = " (subsampled)" if s["subsampled"] else ""
        print(f"    {s['suite']:<16} {s['n_maps_used']:>6,} maps in "
              f"{s['n_runs']:>4} contiguous runs{note}")
    print()
    print(f"  provenance : {provenance}")
    print(f"  written    : {out}")
    print()
    print("  COMMIT THIS FILE. Applying these statistics unchanged to the sealed")
    print("  target suite is the DG-strict condition (section 20.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
