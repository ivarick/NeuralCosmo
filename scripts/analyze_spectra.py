"""Compare power spectra across simulation suites.

Plan reference: sections 4 (Phase 4), 35, 69 (Figure 1).

    python scripts/analyze_spectra.py --n-maps 400

Tests the hypothesis raised after the Phase 4 asymmetry measurement: that
SIMBA's feedback suppresses small-scale structure, so a TNG-trained model comes
to rely on features SIMBA does not have, while a SIMBA-trained model is forced
onto coarser features present in both.

The prediction is falsifiable from the data alone, with no network involved:
the SIMBA/TNG power ratio should DECREASE with increasing k. If the ratio is
flat, or rises, the hypothesis is wrong and the asymmetry needs another
explanation.

To avoid confounding the comparison with cosmology, maps are drawn from
simulations in a restricted Omega_m window, since Omega_m sets the overall
amplitude of clustering and the suites sample it independently.
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

from neuralcosmos.data.manifest import load_data_config, resolve_suite_files  # noqa: E402
from neuralcosmos.data.splits import load_split_file  # noqa: E402
from neuralcosmos.evaluation.spectra import mean_power_spectrum  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402


def select_maps(
    params: np.ndarray,
    sim_ids: list[int],
    maps_per_sim: int,
    n_maps: int,
    omega_lo: float,
    omega_hi: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Pick map indices from simulations inside an Omega_m window."""
    eligible = [s for s in sim_ids if omega_lo <= params[s, 0] <= omega_hi]
    if not eligible:
        raise ValueError(f"no simulations with Omega_m in [{omega_lo}, {omega_hi}]")

    rng = np.random.default_rng(seed)
    all_maps = np.concatenate(
        [np.arange(s * maps_per_sim, (s + 1) * maps_per_sim) for s in eligible]
    )
    if all_maps.size > n_maps:
        chosen = np.sort(rng.choice(all_maps, size=n_maps, replace=False))
    else:
        chosen = np.sort(all_maps)
    return chosen, len(eligible)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/mtot.yaml")
    ap.add_argument("--split-file", default="configs/splits/split_v1.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--suite", action="append", default=None)
    ap.add_argument("--n-maps", type=int, default=400, help="maps per suite")
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--omega-lo", type=float, default=0.25)
    ap.add_argument("--omega-hi", type=float, default=0.35)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/power_spectra.json")
    args = ap.parse_args()

    root = repo_root()

    def _p(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else root / p

    cfg = load_data_config(_p(args.config))
    split_file = load_split_file(_p(args.split_file))
    suites = args.suite or list(cfg["suites"].keys())
    maps_per_sim = int(cfg["maps_per_simulation"])

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(exc, file=sys.stderr)
        return 2

    print("=" * 74)
    print("  NeuralCosmos - cross-suite power spectra (Phase 4 diagnostic)")
    print("=" * 74)
    print(f"  suites     : {suites}")
    print(f"  split      : {args.split}")
    print(f"  Omega_m    : [{args.omega_lo}, {args.omega_hi}]  (controls amplitude)")
    print(f"  maps/suite : {args.n_maps}")
    print()

    results: dict[str, dict] = {}
    for suite in suites:
        sf = resolve_suite_files(cfg, data_root, [suite])[0]
        params = np.loadtxt(sf.param_path)
        sim_ids = list(split_file.suite(suite).ids(args.split))
        chosen, n_sims = select_maps(
            params, sim_ids, maps_per_sim, args.n_maps,
            args.omega_lo, args.omega_hi, args.seed,
        )

        t0 = time.time()
        maps = np.load(sf.map_path, mmap_mode="r")
        res = mean_power_spectrum(maps, chosen, n_bins=args.n_bins)
        del maps

        results[suite] = res.to_dict()
        results[suite]["n_simulations"] = n_sims
        print(f"  {suite:<14} {res.n_maps:>4} maps from {n_sims:>3} simulations "
              f"({time.time() - t0:.0f}s)")

    print()

    # ---- comparison ------------------------------------------------------
    if len(suites) >= 2:
        ref = suites[0]
        k = np.array(results[ref]["k"])
        p_ref = np.array(results[ref]["power"])

        print("  POWER RATIO relative to " + ref)
        header = f"    {'k [h/Mpc]':>11}"
        for s in suites[1:]:
            header += f"{s + '/' + ref:>22}"
        print(header)
        print("    " + "-" * (11 + 22 * (len(suites) - 1)))

        ratios: dict[str, list[float]] = {s: [] for s in suites[1:]}
        for i in range(len(k)):
            if not np.isfinite(k[i]) or p_ref[i] <= 0:
                continue
            line = f"    {k[i]:>11.3f}"
            for s in suites[1:]:
                r = float(np.array(results[s]["power"])[i] / p_ref[i])
                ratios[s].append(r)
                line += f"{r:>22.4f}"
            print(line)
        print()

        # ---- verdict on the hypothesis -----------------------------------
        finite = np.isfinite(k) & (p_ref > 0)
        k_valid = k[finite]
        large_scale = k_valid < 3.0        # a few h/Mpc: quasi-linear
        small_scale = k_valid > 10.0       # deeply non-linear, feedback-sensitive

        print("  HYPOTHESIS TEST")
        print("    Prediction: if SIMBA feedback suppresses small-scale structure,")
        print("    the SIMBA/TNG power ratio should FALL with increasing k.")
        print()
        for s in suites[1:]:
            arr = np.array(ratios[s])
            if large_scale.sum() and small_scale.sum():
                lo = float(np.mean(arr[large_scale]))
                hi = float(np.mean(arr[small_scale]))
                trend = "SUPPRESSED" if hi < lo * 0.95 else (
                    "ENHANCED" if hi > lo * 1.05 else "FLAT"
                )
                print(f"    {s}/{ref}:  k<3 -> {lo:.3f}   k>10 -> {hi:.3f}   "
                      f"small-scale {trend}")
        print()

    out = _p(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "split": args.split,
                "omega_m_window": [args.omega_lo, args.omega_hi],
                "n_maps_requested": args.n_maps,
                "box_size_mpc_h": 25.0,
                "suites": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
