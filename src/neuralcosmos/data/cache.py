"""uint16 log-quantised map cache on fast local storage.

Plan reference: section 83.

WHY
---
Measured on this machine: the archive lives on a USB-attached disk delivering
~33 MB/s, the loader tops out at 130.8 maps/s, and the model can consume 222.7
maps/s. The GPU therefore idles ~41% of the time. Section 83 requires profiling
before rewriting the loader, and the profile says the disk is saturated.

WHAT
----
Each map is stored as ``log10(value)`` linearly quantised into uint16 over a
fixed window. Two bytes per pixel instead of four, on an SSD instead of USB.

WHY uint16 RATHER THAN float16
------------------------------
Same storage cost, far better precision for this data. float16 spacing near
log10 = 15.6 is 2^-10 x 2^3 = 0.0078 dex. uint16 across a 7-dex window gives
7/65535 = 1.07e-4 dex, roughly 70x finer. Measured pixel standard deviation in
log space is 0.4836, so quantisation noise (uniform, sd = step/sqrt(12)) is
about 6e-5 of one standard deviation. In linear terms one step is a 0.025%
change. This is irrelevant next to the physical signal.

WHAT IS DELIBERATELY *NOT* STORED
---------------------------------
Normalization. The cache holds raw log values, so it does not bake in a choice
of source suites. The leave-one-suite-out experiments of section 53 change
which suites are sources, and therefore change the normalizer; a cache that had
already applied normalization would silently become wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = ["QuantSpec", "CachedSuite", "write_cache", "open_cache", "cache_paths"]

_UINT16_MAX = 65535
_MAGIC = "neuralcosmos-log-uint16-v1"


@dataclass(frozen=True)
class QuantSpec:
    """The quantisation window.

    Fixed and documented rather than derived from the data, so two suites
    cached at different times are guaranteed to share an encoding. Deriving the
    window per suite from that suite's own min/max would make the cached values
    suite-dependent, which is a subtle way of leaking suite statistics into the
    input representation.
    """

    lo: float = 9.0
    hi: float = 16.0

    def __post_init__(self) -> None:
        if not self.hi > self.lo:
            raise ValueError(f"invalid quantisation window [{self.lo}, {self.hi}]")

    @property
    def step(self) -> float:
        return (self.hi - self.lo) / _UINT16_MAX

    def encode(self, log_values: np.ndarray) -> np.ndarray:
        """log10 values -> uint16. Raises if anything falls outside the window."""
        lo_hit = float(np.min(log_values))
        hi_hit = float(np.max(log_values))
        if lo_hit < self.lo or hi_hit > self.hi:
            raise ValueError(
                f"log10 values [{lo_hit:.6f}, {hi_hit:.6f}] fall outside the "
                f"quantisation window [{self.lo}, {self.hi}]. Widen the window "
                f"rather than clipping: clipping would silently destroy the "
                f"extreme densities that carry small-scale information."
            )
        scaled = (log_values - self.lo) / (self.hi - self.lo) * _UINT16_MAX
        return np.rint(scaled).astype(np.uint16)

    def decode(self, codes: np.ndarray, dtype=np.float32) -> np.ndarray:
        """uint16 -> log10 values."""
        out = codes.astype(dtype)
        out *= np.asarray((self.hi - self.lo) / _UINT16_MAX, dtype=dtype)
        out += np.asarray(self.lo, dtype=dtype)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi, "step": self.step, "codes": _UINT16_MAX + 1}


@dataclass
class CachedSuite:
    """Metadata describing one cached suite."""

    suite: str
    path: Path
    shape: tuple[int, ...]
    spec: QuantSpec
    source_file: str
    source_bytes: int
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "magic": _MAGIC,
            "suite": self.suite,
            "file": self.path.name,
            "shape": list(self.shape),
            "dtype": "uint16",
            "quantisation": self.spec.to_dict(),
            "transform": "log10",
            "normalised": False,
            "source_file": self.source_file,
            "source_bytes": self.source_bytes,
            "created_utc": self.created_utc,
            "note": (
                "Un-normalised log10 values. Normalization is applied at load "
                "time from source-training statistics (plan section 20.2), so "
                "this cache stays valid when the source-suite set changes."
            ),
        }


def cache_paths(cache_root: Path, suite: str, field: str = "Mtot") -> tuple[Path, Path]:
    """Return the (array, sidecar-metadata) paths for a suite."""
    stem = f"{field}_{suite}_log_uint16"
    return cache_root / f"{stem}.npy", cache_root / f"{stem}.json"


def write_cache(
    suite: str,
    source_path: Path,
    cache_root: Path,
    spec: QuantSpec | None = None,
    chunk_maps: int = 250,
    progress: Callable[[int, int], None] | None = None,
    overwrite: bool = False,
) -> CachedSuite:
    """Build the quantised cache for one suite.

    Written chunk-by-chunk through a writable memmap so neither the source nor
    the destination is ever fully resident.
    """
    spec = spec or QuantSpec()
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path, meta_path = cache_paths(cache_root, suite)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Cache already exists: {out_path}. Pass overwrite=True.")

    src = np.load(source_path, mmap_mode="r")
    shape = tuple(int(v) for v in src.shape)
    n = shape[0]

    dst = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=shape)
    try:
        for start in range(0, n, chunk_maps):
            stop = min(start + chunk_maps, n)
            block = np.asarray(src[start:stop], dtype=np.float64)
            if np.any(block <= 0):
                raise ValueError(
                    f"[{suite}] non-positive pixel in maps [{start}, {stop}); "
                    f"log10 is undefined. Re-run scripts/validate_data.py."
                )
            dst[start:stop] = spec.encode(np.log10(block))
            if progress is not None:
                progress(stop, n)
        dst.flush()
    finally:
        del dst
        del src

    cached = CachedSuite(
        suite=suite,
        path=out_path,
        shape=shape,
        spec=spec,
        source_file=source_path.name,
        source_bytes=source_path.stat().st_size,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    meta_path.write_text(json.dumps(cached.to_dict(), indent=2), encoding="utf-8")
    return cached


def open_cache(cache_root: Path, suite: str, field: str = "Mtot") -> tuple[Path, QuantSpec]:
    """Resolve a cached suite, validating its sidecar metadata.

    Returns the array path and the quantisation spec needed to decode it.
    """
    arr_path, meta_path = cache_paths(cache_root, suite, field)
    if not arr_path.exists():
        raise FileNotFoundError(f"No cache for {suite!r} at {arr_path}")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Cache array {arr_path} has no metadata sidecar. Refusing to guess "
            f"its quantisation window; rebuild the cache."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("magic") != _MAGIC:
        raise ValueError(f"Unrecognised cache format in {meta_path}: {meta.get('magic')!r}")
    if meta.get("normalised", False):
        raise ValueError(
            f"{meta_path} claims to hold normalised values. This cache format "
            f"must store raw log10 so it stays valid across source-suite changes."
        )

    q = meta["quantisation"]
    return arr_path, QuantSpec(lo=float(q["lo"]), hi=float(q["hi"]))
