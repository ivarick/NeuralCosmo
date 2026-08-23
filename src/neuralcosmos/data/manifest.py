"""Data manifest: the authoritative record of what is actually on disk.

Plan reference: sections 7, 12, 13.

The manifest is deliberately separate from validation. This module answers
"what files exist, how big are they, what do they contain" and records that as
a machine-readable artefact. ``validate.py`` answers "is that acceptable".

The SHA-256 digests recorded here are a LOCAL integrity record. CAMELS does not
publish matching checksums, so they must never be presented as official ones
(section 13). Their purpose is to detect a file that changed underneath us
between runs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..paths import resolve_data_root

__all__ = [
    "SuiteFiles",
    "SuiteManifest",
    "DataManifest",
    "load_data_config",
    "resolve_suite_files",
    "build_manifest",
]

# Hashing 3.9 GB byte-by-byte is slow and, for our purpose, unnecessary. The
# digest exists to notice that a file changed, not to defend against an
# adversary. We hash a deterministic sample: the file size plus evenly spaced
# 1 MiB blocks, which catches truncation, partial downloads and overwrites.
_HASH_BLOCK = 1 << 20  # 1 MiB
_HASH_SAMPLES = 32


@dataclass
class SuiteFiles:
    """Resolved absolute paths for one simulation suite."""

    suite: str
    map_path: Path
    param_path: Path

    def exists(self) -> tuple[bool, bool]:
        return self.map_path.exists(), self.param_path.exists()


@dataclass
class SuiteManifest:
    """Everything we know about the files of one suite on this machine."""

    suite: str
    map_file: str
    param_file: str
    map_bytes: int
    param_bytes: int
    maps_shape: list[int]
    maps_dtype: str
    params_shape: list[int]
    n_simulations: int
    maps_per_simulation: int
    sha256_sampled_map: str
    sha256_full_params: str
    param_ranges: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataManifest:
    """The full manifest across all configured suites."""

    dataset: str
    set_name: str
    field: str
    redshift: float
    data_root: str
    generated_utc: str
    suites: list[SuiteManifest] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "set": self.set_name,
            "field": self.field,
            "redshift": self.redshift,
            "data_root": self.data_root,
            "generated_utc": self.generated_utc,
            "note": (
                "sha256 values are a local integrity record only. CAMELS does not "
                "publish matching checksums; do not present these as official."
            ),
            "suites": [s.to_dict() for s in self.suites],
        }

    def content_hash(self) -> str:
        """A stable hash of the data content of the manifest.

        Excludes the generation timestamp and the machine-specific data root, so
        the same archive on two machines yields the same hash. This value is
        recorded in every run's metadata (section 64).
        """
        payload = {
            "dataset": self.dataset,
            "set": self.set_name,
            "field": self.field,
            "redshift": self.redshift,
            "suites": [s.to_dict() for s in self.suites],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = self.to_dict()
        doc["manifest_content_hash"] = self.content_hash()
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return path


def load_data_config(config_path: str | Path) -> dict[str, Any]:
    """Load and lightly sanity-check a data configuration YAML."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    for key in ("suites", "expected", "param_columns", "maps_per_simulation"):
        if key not in cfg:
            raise ValueError(f"Data config {config_path} is missing required key: {key!r}")
    return cfg


def resolve_suite_files(
    cfg: dict[str, Any],
    data_root: Path,
    suites: list[str] | None = None,
) -> list[SuiteFiles]:
    """Map configured suite names onto absolute paths under ``data_root``."""
    wanted = suites if suites is not None else list(cfg["suites"].keys())
    resolved: list[SuiteFiles] = []
    for name in wanted:
        if name not in cfg["suites"]:
            known = ", ".join(cfg["suites"])
            raise KeyError(f"Suite {name!r} is not in the data config. Known suites: {known}")
        entry = cfg["suites"][name]
        resolved.append(
            SuiteFiles(
                suite=name,
                map_path=data_root / entry["map_file"],
                param_path=data_root / entry["param_file"],
            )
        )
    return resolved


def sampled_sha256(path: Path, n_samples: int = _HASH_SAMPLES) -> str:
    """Digest of a deterministic sample of a large file, plus its size."""
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())

    if size <= _HASH_BLOCK * n_samples:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_BLOCK), b""):
                h.update(chunk)
        return h.hexdigest()

    stride = size // n_samples
    with path.open("rb") as fh:
        for i in range(n_samples):
            fh.seek(i * stride)
            h.update(fh.read(_HASH_BLOCK))
        fh.seek(max(0, size - _HASH_BLOCK))
        h.update(fh.read(_HASH_BLOCK))
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    """Full digest. Used only for the small parameter text files."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_BLOCK), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    config_path: str | Path,
    data_root: str | Path | None = None,
    suites: list[str] | None = None,
) -> DataManifest:
    """Inspect the archive on disk and produce a manifest.

    Maps are opened with ``mmap_mode="r"`` so that shape and dtype are read from
    the ``.npy`` header without loading gigabytes into RAM (section 14).
    """
    cfg = load_data_config(config_path)
    root = resolve_data_root(data_root)
    param_cols = cfg["param_columns"]
    maps_per_sim = int(cfg["maps_per_simulation"])

    manifest = DataManifest(
        dataset=cfg.get("dataset", "CAMELS_CMD"),
        set_name=str(cfg.get("set", "LH")),
        field=str(cfg.get("field", "Mtot")),
        redshift=float(cfg.get("redshift", 0.0)),
        data_root=str(root),
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    for sf in resolve_suite_files(cfg, root, suites):
        map_ok, param_ok = sf.exists()
        if not map_ok:
            raise FileNotFoundError(f"[{sf.suite}] map file not found: {sf.map_path}")
        if not param_ok:
            raise FileNotFoundError(f"[{sf.suite}] parameter file not found: {sf.param_path}")

        maps = np.load(sf.map_path, mmap_mode="r")
        params = np.loadtxt(sf.param_path)
        if params.ndim == 1:
            params = params.reshape(1, -1)

        ranges = {
            name: [float(params[:, idx].min()), float(params[:, idx].max())]
            for name, idx in param_cols.items()
            if idx < params.shape[1]
        }

        manifest.suites.append(
            SuiteManifest(
                suite=sf.suite,
                map_file=sf.map_path.name,
                param_file=sf.param_path.name,
                map_bytes=sf.map_path.stat().st_size,
                param_bytes=sf.param_path.stat().st_size,
                maps_shape=[int(v) for v in maps.shape],
                maps_dtype=str(maps.dtype),
                params_shape=[int(v) for v in params.shape],
                n_simulations=int(params.shape[0]),
                maps_per_simulation=maps_per_sim,
                sha256_sampled_map=sampled_sha256(sf.map_path),
                sha256_full_params=file_sha256(sf.param_path),
                param_ranges=ranges,
            )
        )
        del maps

    return manifest
