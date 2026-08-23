"""Generate the permanent simulation-level split file.

Plan reference: sections 16, 17, 104 (step 4).

    python scripts/build_splits.py --seed 42

Section 16: "The specific simulation IDs should be generated once with a fixed
seed and stored permanently. Do not regenerate splits every run."

This script therefore REFUSES to overwrite an existing split file unless
--force is passed, and prints a warning explaining what overwriting would
invalidate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from neuralcosmos.data.manifest import load_data_config  # noqa: E402
from neuralcosmos.data.splits import SPLIT_NAMES, build_split_file, load_split_file  # noqa: E402
from neuralcosmos.paths import repo_root  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/mtot.yaml")
    ap.add_argument("--seed", type=int, default=42, help="master seed (default: 42)")
    ap.add_argument("--n-val", type=int, default=50, help="validation simulations per suite")
    ap.add_argument("--n-test", type=int, default=50, help="test simulations per suite")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out", default=None, help="default: configs/splits/split_<version>.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = ap.parse_args()

    root = repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_data_config(config_path)

    out = Path(args.out) if args.out else root / "configs" / "splits" / f"split_{args.version}.json"
    if not out.is_absolute():
        out = root / out

    if out.exists() and not args.force:
        existing = load_split_file(out)
        print("=" * 74)
        print("  Split file already exists. Refusing to regenerate.")
        print("=" * 74)
        print(f"  path : {out}")
        print(f"  hash : {existing.content_hash()}")
        print(f"  seed : {existing.master_seed}")
        print()
        print("  Section 16 requires splits to be generated once and kept. Regenerating")
        print("  would silently invalidate every result, checkpoint and metric already")
        print("  computed against the current partition.")
        print()
        print("  Pass --force only if you intend exactly that.")
        return 3

    n_sims = int(cfg["expected"]["n_simulations"])
    maps_per_sim = int(cfg["maps_per_simulation"])
    suites = {name: n_sims for name in cfg["suites"]}

    sf = build_split_file(
        suites=suites,
        master_seed=args.seed,
        n_val=args.n_val,
        n_test=args.n_test,
        maps_per_simulation=maps_per_sim,
        version=args.version,
    )
    sf.write(out)

    print("=" * 74)
    print("  NeuralCosmos - simulation-level splits (plan section 16)")
    print("=" * 74)
    print(f"  master seed : {args.seed}")
    print(f"  written to  : {out}")
    print(f"  hash        : {sf.content_hash()}")
    print()
    print(f"  {'suite':<16}{'train':>8}{'val':>7}{'test':>7}   |   "
          f"{'train':>8}{'val':>7}{'test':>7}  (maps)")
    print("  " + "-" * 70)
    for name in sorted(sf.splits):
        sp = sf.suite(name)
        sims = [len(sp.ids(s)) for s in SPLIT_NAMES]
        maps = [n * maps_per_sim for n in sims]
        print(f"  {name:<16}{sims[0]:>8}{sims[1]:>7}{sims[2]:>7}   |   "
              f"{maps[0]:>8}{maps[1]:>7}{maps[2]:>7}")
    print()
    print("  All split-integrity invariants (section 17) verified:")
    print("    - train / val / test simulation sets are pairwise disjoint")
    print("    - the three sets partition 0..N-1 exactly, with no duplicates")
    print(f"    - all {maps_per_sim} maps of a simulation belong to exactly one split")
    print()
    print("  COMMIT THIS FILE. Do not regenerate it per run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
