"""Measure data throughput against model throughput.

Plan reference: sections 83, 84.

    python scripts/benchmark_io.py --workers 0 --workers 2 --workers 4

Section 83 is explicit that the loader must not be rewritten on a hunch:
profile first, and do not assume the GPU is the bottleneck. This script
measures both sides of the question separately.

  data throughput  : maps/second the DataLoader can deliver, per worker count
  model throughput : maps/second a training step can consume on the GPU

If model throughput exceeds data throughput, the GPU starves and the loader is
the problem. If not, the loader is already fast enough and optimising it would
buy nothing.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from neuralcosmos.data.builders import build_dataset  # noqa: E402
from neuralcosmos.data.manifest import load_data_config  # noqa: E402
from neuralcosmos.data.splits import load_split_file  # noqa: E402
from neuralcosmos.data.statistics import load_normalizer  # noqa: E402
from neuralcosmos.models.erm import ERMModel, build_backbone  # noqa: E402
from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root  # noqa: E402


def bench_loader(dataset, batch_size: int, workers: int, n_batches: int) -> dict:
    """Time pure data delivery, with no model attached."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=True,
    )
    it = iter(loader)

    # One warm-up batch: worker startup and the first mmap page faults are
    # setup cost, not steady-state throughput.
    try:
        next(it)
    except StopIteration:
        return {"workers": workers, "error": "dataset too small"}

    t0 = time.perf_counter()
    seen = 0
    for _ in range(n_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        seen += batch["image"].shape[0]
    elapsed = time.perf_counter() - t0

    del it, loader
    return {
        "workers": workers,
        "maps": seen,
        "seconds": round(elapsed, 3),
        "maps_per_second": round(seen / elapsed, 1) if elapsed > 0 else 0.0,
        "mb_per_second": round(seen * 256 * 256 * 4 / (1024**2) / elapsed, 1) if elapsed > 0 else 0.0,
    }


def bench_model(model, device, batch_size: int, n_steps: int, amp: bool) -> dict:
    """Time forward + backward on synthetic data of the right shape."""
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(batch_size, 1, 256, 256, device=device)
    y = torch.rand(batch_size, 2, device=device)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = torch.nn.functional.mse_loss(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):        # warm-up: cuDNN autotune and allocator warm-up
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(n_steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out = {
        "batch_size": batch_size,
        "steps": n_steps,
        "seconds": round(elapsed, 3),
        "maps_per_second": round(n_steps * batch_size / elapsed, 1),
        "amp": bool(use_amp),
    }
    if device.type == "cuda":
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data/mtot.yaml")
    ap.add_argument("--split-file", default="configs/splits/split_v1.json")
    ap.add_argument("--normalizer", default="configs/normalizers/norm_IllustrisTNG-SIMBA_v1.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--suite", action="append", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=30, help="batches to time per worker count")
    ap.add_argument("--workers", type=int, action="append", default=None,
                    help="worker counts to test; repeatable. Default: 0, 2, 4")
    ap.add_argument("--model-steps", type=int, default=30)
    ap.add_argument("--max-simulations", type=int, default=200)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="reports/io_benchmark.json")
    args = ap.parse_args()

    root = repo_root()

    def _p(x: str) -> Path:
        p = Path(x)
        return p if p.is_absolute() else root / p

    cfg = load_data_config(_p(args.config))
    split_file = load_split_file(_p(args.split_file))
    normalizer = load_normalizer(_p(args.normalizer))
    suites = args.suite or list(cfg.get("roles", {}).get("development_suites", []))
    worker_counts = args.workers or [0, 2, 4]

    try:
        data_root = resolve_data_root(args.data_root)
    except DataRootNotFound as exc:
        print(exc, file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 74)
    print("  NeuralCosmos - throughput benchmark (plan section 83)")
    print("=" * 74)
    print(f"  device     : {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else platform.processor()})")
    print(f"  data root  : {data_root}")
    print(f"  suites     : {suites}")
    print(f"  batch size : {args.batch_size}")
    print()

    dataset = build_dataset(
        cfg, data_root, split_file, suites, "train", normalizer,
        max_simulations=args.max_simulations,
    )
    print(f"  dataset    : {len(dataset):,} maps "
          f"({args.max_simulations} sims/suite)")
    print()

    # ---- data side ------------------------------------------------------
    print("  DATA THROUGHPUT")
    print(f"    {'workers':>8}{'maps/s':>12}{'MB/s':>10}{'seconds':>10}")
    print("    " + "-" * 40)
    data_results = []
    for w in worker_counts:
        r = bench_loader(dataset, args.batch_size, w, args.batches)
        data_results.append(r)
        if "error" in r:
            print(f"    {w:>8}  {r['error']}")
        else:
            print(f"    {w:>8}{r['maps_per_second']:>12.1f}{r['mb_per_second']:>10.1f}"
                  f"{r['seconds']:>10.2f}")
    print()

    # ---- model side -----------------------------------------------------
    print("  MODEL THROUGHPUT")
    model = ERMModel(backbone=build_backbone({"type": "small_cnn", "latent_dim": 256}))
    print(f"    parameters : {model.n_parameters:,}")
    m = bench_model(model, device, args.batch_size, args.model_steps, amp=not args.no_amp)
    print(f"    maps/s     : {m['maps_per_second']:.1f}  (amp={m['amp']})")
    if "peak_vram_mb" in m:
        print(f"    peak VRAM  : {m['peak_vram_mb']:.0f} MB")
    print()

    # ---- verdict --------------------------------------------------------
    usable = [r for r in data_results if "maps_per_second" in r]
    best = max(usable, key=lambda r: r["maps_per_second"]) if usable else None
    verdict = "indeterminate"
    if best:
        ratio = best["maps_per_second"] / m["maps_per_second"]
        print("=" * 74)
        print(f"  best data throughput : {best['maps_per_second']:.1f} maps/s "
              f"(workers={best['workers']})")
        print(f"  model can consume    : {m['maps_per_second']:.1f} maps/s")
        print(f"  data/model ratio     : {ratio:.2f}")
        print()
        if ratio < 0.9:
            starve = 1.0 - ratio
            verdict = "io_bound"
            print(f"  VERDICT: I/O BOUND. The GPU idles roughly {starve*100:.0f}% of the time.")
            print("  Section 83 remedies, in order of cost:")
            print("    - larger contiguous reads / shuffle within contiguous blocks")
            print("    - cache a working subset on SSD")
            print("    - a uint16 log-quantised cache (~2.0 GB/suite instead of 3.9 GB)")
        elif ratio < 1.5:
            verdict = "balanced"
            print("  VERDICT: BALANCED. Data and compute are comparable; a modest")
            print("  loader improvement would still help, but the GPU is not idle.")
        else:
            verdict = "compute_bound"
            print("  VERDICT: COMPUTE BOUND. The loader keeps the GPU fed.")
            print("  Do not rewrite the loader (section 83).")
        print("=" * 74)

    out = _p(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__,
                "platform": platform.platform(),
                "data_root": str(data_root),
                "suites": suites,
                "batch_size": args.batch_size,
                "dataset_maps": len(dataset),
                "data": data_results,
                "model": m,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
