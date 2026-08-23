"""Data validation: reject bad data before any model exists.

Plan reference: sections 12, 17, 20.1.

The only purpose of this module is to say no. It has no other side effects on
the pipeline, and nothing it computes may be reused as a modelling statistic.

IMPORTANT SCOPE NOTE
--------------------
This module computes whole-file pixel statistics for every configured suite,
including the sealed target suite. That is permitted: section 19 explicitly
allows integrity checks on Astrid. What is NOT permitted is feeding any of
those numbers into normalization, architecture choice, or hyperparameters.
Normalization statistics are computed separately, from source-training
simulations only (section 20.2), by ``statistics.py``. The two must never be
confused, which is why they live in different modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..paths import resolve_data_root
from .manifest import load_data_config, resolve_suite_files

__all__ = [
    "Severity",
    "Check",
    "PixelStats",
    "SuiteValidation",
    "ValidationReport",
    "scan_pixel_stats",
    "validate_suite",
    "validate_all",
]


class Severity:
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"


@dataclass
class Check:
    """A single validation assertion and its outcome."""

    name: str
    passed: bool
    severity: str
    message: str

    @property
    def is_blocking(self) -> bool:
        return (not self.passed) and self.severity == Severity.ERROR


@dataclass
class PixelStats:
    """Streaming statistics over an entire map array.

    Accumulated in chunks so a 3.9 GB file never lands in RAM at once
    (section 14). ``sum`` and ``sum_sq`` are float64 to keep the variance
    numerically sane across ~10^9 float32 values.
    """

    count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    total: float = 0.0
    total_sq: float = 0.0
    n_nan: int = 0
    n_inf: int = 0
    n_nonpositive: int = 0
    n_zero: int = 0

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        if self.count < 2:
            return float("nan")
        var = self.total_sq / self.count - self.mean**2
        return float(np.sqrt(max(var, 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
            "n_nan": self.n_nan,
            "n_inf": self.n_inf,
            "n_nonpositive": self.n_nonpositive,
            "n_zero": self.n_zero,
        }


@dataclass
class SuiteValidation:
    suite: str
    checks: list[Check] = field(default_factory=list)
    stats: PixelStats | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, severity: str, message: str) -> Check:
        c = Check(name=name, passed=passed, severity=severity, message=message)
        self.checks.append(c)
        return c

    @property
    def ok(self) -> bool:
        return not any(c.is_blocking for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "ok": self.ok,
            "info": self.info,
            "pixel_stats": self.stats.to_dict() if self.stats else None,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


@dataclass
class ValidationReport:
    suites: list[SuiteValidation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.suites)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "suites": [s.to_dict() for s in self.suites]}


def scan_pixel_stats(
    map_path: Path,
    chunk_maps: int = 250,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[PixelStats, tuple[int, ...], str]:
    """Stream the whole map array and accumulate exact statistics.

    A sampled scan cannot prove positivity, and positivity decides whether the
    log transform in section 20.1 is legal. So this reads every value.

    Parameters
    ----------
    chunk_maps:
        Maps per chunk. 250 maps of 256x256 float32 is about 65 MB, which keeps
        the read sequential without stressing RAM.

    Returns
    -------
    (stats, shape, dtype)
    """
    arr = np.load(map_path, mmap_mode="r")
    shape = tuple(int(v) for v in arr.shape)
    dtype = str(arr.dtype)
    n = shape[0]

    st = PixelStats()
    for start in range(0, n, chunk_maps):
        stop = min(start + chunk_maps, n)
        # np.asarray on the mmap slice performs the actual read.
        block = np.asarray(arr[start:stop], dtype=np.float64)

        finite = np.isfinite(block)
        n_nan = int(np.isnan(block).sum())
        n_inf = int(np.isinf(block).sum())
        st.n_nan += n_nan
        st.n_inf += n_inf

        vals = block[finite] if (n_nan or n_inf) else block.ravel()
        if vals.size:
            st.count += int(vals.size)
            st.minimum = min(st.minimum, float(vals.min()))
            st.maximum = max(st.maximum, float(vals.max()))
            st.total += float(vals.sum())
            st.total_sq += float(np.square(vals).sum())
            st.n_nonpositive += int((vals <= 0).sum())
            st.n_zero += int((vals == 0).sum())

        if progress is not None:
            progress(stop, n)

    del arr
    return st, shape, dtype


def validate_suite(
    suite: str,
    map_path: Path,
    param_path: Path,
    cfg: dict[str, Any],
    full_scan: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> SuiteValidation:
    """Run every section-12 check against one suite."""
    v = SuiteValidation(suite=suite)
    expected = cfg["expected"]
    param_cols = cfg["param_columns"]
    maps_per_sim = int(cfg["maps_per_simulation"])
    tol = float(cfg.get("targets", {}).get("range_tolerance", 0.01))

    # --- existence ----------------------------------------------------------
    map_exists = map_path.exists()
    v.add(
        "map_file_exists",
        map_exists,
        Severity.ERROR,
        str(map_path) if map_exists else f"missing: {map_path}",
    )
    param_exists = param_path.exists()
    v.add(
        "param_file_exists",
        param_exists,
        Severity.ERROR,
        str(param_path) if param_exists else f"missing: {param_path}",
    )
    if not (map_exists and param_exists):
        return v

    # --- byte size ----------------------------------------------------------
    map_bytes = map_path.stat().st_size
    v.info["map_bytes"] = map_bytes
    if "map_file_bytes" in expected:
        want = int(expected["map_file_bytes"])
        v.add(
            "map_byte_size",
            map_bytes == want,
            Severity.ERROR,
            f"{map_bytes:,} bytes (expected {want:,}) "
            f"{'- file is truncated or incomplete' if map_bytes != want else ''}".strip(),
        )

    # --- header shape and dtype, without reading the payload ----------------
    header = np.load(map_path, mmap_mode="r")
    shape = tuple(int(x) for x in header.shape)
    dtype = str(header.dtype)
    del header
    v.info["maps_shape"] = list(shape)
    v.info["maps_dtype"] = dtype

    want_shape = tuple(int(x) for x in expected["maps_shape"])
    v.add(
        "maps_shape",
        shape == want_shape,
        Severity.ERROR,
        f"{shape} (expected {want_shape})",
    )
    if "maps_dtype" in expected:
        v.add(
            "maps_dtype",
            dtype == str(expected["maps_dtype"]),
            Severity.WARN,
            f"{dtype} (expected {expected['maps_dtype']})",
        )

    # --- parameters ---------------------------------------------------------
    params = np.loadtxt(param_path)
    if params.ndim == 1:
        params = params.reshape(1, -1)
    p_shape = tuple(int(x) for x in params.shape)
    v.info["params_shape"] = list(p_shape)

    want_p = tuple(int(x) for x in expected["params_shape"])
    v.add(
        "params_shape",
        p_shape == want_p,
        Severity.ERROR,
        f"{p_shape} (expected {want_p})",
    )
    v.add(
        "params_finite",
        bool(np.isfinite(params).all()),
        Severity.ERROR,
        "all parameter values finite"
        if np.isfinite(params).all()
        else f"{int((~np.isfinite(params)).sum())} non-finite parameter values",
    )

    # --- the map/simulation contract ---------------------------------------
    # Every scientific claim depends on this identity. If it fails, every
    # simulation-level split and every bootstrap interval is wrong.
    n_sims = p_shape[0]
    n_maps = shape[0]
    consistent = n_maps == n_sims * maps_per_sim
    v.add(
        "map_simulation_mapping",
        consistent,
        Severity.ERROR,
        f"{n_maps} maps == {n_sims} simulations x {maps_per_sim} maps/sim"
        if consistent
        else f"{n_maps} maps != {n_sims} x {maps_per_sim} = {n_sims * maps_per_sim}",
    )

    # --- target ranges ------------------------------------------------------
    ranges_cfg = cfg.get("targets", {}).get("ranges", {})
    observed: dict[str, list[float]] = {}
    for tname, (lo, hi) in ranges_cfg.items():
        idx = param_cols[tname]
        col = params[:, idx]
        obs_lo, obs_hi = float(col.min()), float(col.max())
        observed[tname] = [obs_lo, obs_hi]
        within = (obs_lo >= lo - tol) and (obs_hi <= hi + tol)
        v.add(
            f"range_{tname}",
            within,
            Severity.ERROR,
            f"[{obs_lo:.5f}, {obs_hi:.5f}] within [{lo}, {hi}] +/- {tol}"
            if within
            else f"[{obs_lo:.5f}, {obs_hi:.5f}] escapes [{lo}, {hi}] +/- {tol}",
        )
    v.info["observed_target_ranges"] = observed

    # --- pixel scan ---------------------------------------------------------
    if not full_scan:
        v.add(
            "pixel_scan",
            True,
            Severity.WARN,
            "SKIPPED (--quick). Positivity is unverified, so the log transform "
            "must not be enabled on the strength of this run.",
        )
        return v

    st, _, _ = scan_pixel_stats(map_path, progress=progress)
    v.stats = st

    v.add(
        "no_nan",
        st.n_nan == 0,
        Severity.ERROR,
        "0 NaN" if st.n_nan == 0 else f"{st.n_nan:,} NaN values",
    )
    v.add(
        "no_inf",
        st.n_inf == 0,
        Severity.ERROR,
        "0 Inf" if st.n_inf == 0 else f"{st.n_inf:,} Inf values",
    )

    # Section 20.1: the log transform is legal only if every value is strictly
    # positive. If not, STOP and inspect the data semantics. Do not silently
    # invent an epsilon, and do not reach for log1p without a reason.
    strictly_positive = st.n_nonpositive == 0
    v.add(
        "strictly_positive",
        strictly_positive,
        Severity.WARN,
        f"min = {st.minimum:.6g}; log(x) is safe"
        if strictly_positive
        else (
            f"{st.n_nonpositive:,} values <= 0 ({st.n_zero:,} exactly zero), "
            f"min = {st.minimum:.6g}. Section 20.1: STOP and inspect data "
            f"semantics before choosing a log transform."
        ),
    )

    return v


def validate_all(
    config_path: str | Path,
    data_root: str | Path | None = None,
    suites: list[str] | None = None,
    full_scan: bool = True,
    progress: Callable[[str, int, int], None] | None = None,
) -> ValidationReport:
    """Validate every configured suite and return a combined report."""
    cfg = load_data_config(config_path)
    root = resolve_data_root(data_root)

    report = ValidationReport()
    for sf in resolve_suite_files(cfg, root, suites):
        cb = None
        if progress is not None:
            cb = lambda done, total, _s=sf.suite: progress(_s, done, total)  # noqa: E731
        report.suites.append(
            validate_suite(
                suite=sf.suite,
                map_path=sf.map_path,
                param_path=sf.param_path,
                cfg=cfg,
                full_scan=full_scan,
                progress=cb,
            )
        )
    return report
