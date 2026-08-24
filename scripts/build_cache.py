"""Build the uint16 log-quantised map cache on fast local storage.

Plan reference: section 83.

    python scripts/build_cache.py --cache-root C:\\camels_cache

Justified by measurement, not assumption: benchmark_io.py showed the loader
saturating a USB-attached disk at ~33 MB/s while the GPU could consume 70%
more. Section 83 requires profiling before rewriting the loader, and that
profile is recorded in reports/io_benchmark.json.

Defaults to the SOURCE suites only. The sealed target is read a handful of
times at the very end of the project, so caching it buys little and costs
1.83 GiB of a system drive.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neuralcosmos.data.cache import QuantSpec, cache_paths, write_cache  # noqa: E402
from neuralcosmos.data.manifest import load_data_config, resolve_suite_files  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402


def _gib(n: int) -> float:
    return n / (1024**3)


class _Progress:
    def __init__(self, suite: str, enabled: bool = True) -> None:
        self.suite = suite
        self.enabled = enabled
        self._t0 = time.time()
        self._last = 0.0

    def __call__(self, done: int, total: int) -> None:
        if not self.enabled:
            return
        now = time.time()
        if done < total and (now - self._last) < 0.5:
            return
        self._last = now
        frac = done / total
        elapsed = now - self._t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        bar = "#" * int(28 * frac) + "-" * (28 - int(28 * frac))
        end = "\n" if done >= total else "\r"
        sys.stdout.write(
            f"    [{bar}] {frac*100:5.1f}%  {done:>6,}/{total:,}  "
            f"{rate:6.0f} maps/s  eta {eta:4.0f}s{end}"
        )
        sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/mtot.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--cache-root",
        default=None,
        help="destination; default NEURALCOSMOS_CACHE_ROOT, else <repo>/../camels_cache",
    )
    ap.add_argument(
        "--suite",
        action="append",
        default=None,
        help="suite to cache; repeatable. Default: the config's development_suites.",
    )
    ap.add_argument("--quant-lo", type=float, default=9.0)
    ap.add_argument("--quant-hi", type=float, default=16.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args()

    import os

    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_data_config(config_path)

    cache_root = Path(
        args.cache_root
        or os.environ.get("NEURALCOSMOS_CACHE_ROOT")
        or (root.parent / "camels_cache")
    ).resolve()

    suites = args.suite or list(cfg.get("roles", {}).get("development_suites", []))
    if not suites:
        print("No suites requested and none configured.", file=sys.stderr)
        return 2

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(exc, file=sys.stderr)
        return 2

    spec = QuantSpec(lo=args.quant_lo, hi=args.quant_hi)
    n_maps = int(cfg["expected"]["maps_shape"][0])
    px = int(cfg["expected"]["maps_shape"][1])
    per_suite_bytes = n_maps * px * px * 2

    print("=" * 74)
    print("  NeuralCosmos - build log-quantised cache (plan section 83)")
    print("=" * 74)
    print(f"  source     : {data_root}")
    print(f"  cache root : {cache_root}")
    print(f"  suites     : {suites}")
    print(f"  window     : log10 in [{spec.lo}, {spec.hi}], step {spec.step:.3e} dex")
    print(f"  size       : {_gib(per_suite_bytes):.2f} GiB per suite, "
          f"{_gib(per_suite_bytes * len(suites)):.2f} GiB total")

    cache_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_root).free
    needed = per_suite_bytes * len(suites)
    print(f"  free space : {_gib(free):.2f} GiB")
    if free < needed * 1.05:
        print(f"\n  REFUSING: need ~{_gib(needed):.2f} GiB plus headroom, "
              f"only {_gib(free):.2f} GiB free.", file=sys.stderr)
        return 3
    print()

    total_t0 = time.time()
    for suite in suites:
        arr_path, _ = cache_paths(cache_root, suite, str(cfg.get("field", "Mtot")))
        if arr_path.exists() and not args.overwrite:
            print(f"  {suite}: already cached at {arr_path} (use --overwrite)")
            continue

        sf = resolve_suite_files(cfg, data_root, [suite])[0]
        print(f"  {suite}")
        t0 = time.time()
        cached = write_cache(
            suite=suite,
            source_path=sf.map_path,
            cache_root=cache_root,
            spec=spec,
            progress=_Progress(suite, enabled=not args.no_progress),
            overwrite=args.overwrite,
        )
        dt = time.time() - t0
        out_bytes = cached.path.stat().st_size
        print(f"    wrote {cached.path.name}  {_gib(out_bytes):.2f} GiB  "
              f"({_gib(cached.source_bytes):.2f} GiB source, "
              f"{100 * out_bytes / cached.source_bytes:.0f}%)  in {dt:.0f}s")
        print()

    print("=" * 74)
    print(f"  done in {time.time() - total_t0:.0f}s")
    print()
    print("  Point training at the cache with:")
    print(f"    $env:NEURALCOSMOS_CACHE_ROOT = \"{cache_root}\"")
    print()
    print("  The cache holds UN-NORMALISED log10 values, so it stays valid when")
    print("  the source-suite set changes for the leave-one-suite-out runs of")
    print("  section 53. Re-run scripts/benchmark_io.py to confirm the loader is")
    print("  no longer the bottleneck.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
