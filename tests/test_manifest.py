"""Manifest construction and integrity hashing.

Plan reference: sections 13, 64.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neuralcosmos.data.manifest import (
    build_manifest,
    file_sha256,
    load_data_config,
    resolve_suite_files,
    sampled_sha256,
)


def test_load_config_rejects_incomplete_config(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("dataset: X\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_data_config(bad)


def test_resolve_suite_files_reports_unknown_suite(synthetic_config: Path, synthetic_archive: Path):
    cfg = load_data_config(synthetic_config)
    with pytest.raises(KeyError, match="NotASuite"):
        resolve_suite_files(cfg, synthetic_archive, suites=["NotASuite"])


def test_build_manifest_records_shapes_and_ranges(synthetic_config: Path, synthetic_archive: Path):
    m = build_manifest(synthetic_config, data_root=synthetic_archive)
    assert len(m.suites) == 2

    for s in m.suites:
        assert s.maps_shape == [90, 8, 8]        # 6 sims x 15 maps
        assert s.params_shape == [6, 6]
        assert s.maps_dtype == "float32"
        assert s.n_simulations * s.maps_per_simulation == s.maps_shape[0]
        # Ranges must reflect the documented CAMELS design bounds.
        assert s.param_ranges["omega_m"][0] == pytest.approx(0.1002, abs=1e-4)
        assert s.param_ranges["sigma8"][1] == pytest.approx(0.9998, abs=1e-4)


def test_content_hash_is_stable_across_rebuilds(synthetic_config: Path, synthetic_archive: Path):
    a = build_manifest(synthetic_config, data_root=synthetic_archive)
    b = build_manifest(synthetic_config, data_root=synthetic_archive)
    # Timestamps differ between the two, so a naive hash of the whole document
    # would differ. The content hash must ignore them.
    assert a.generated_utc is not None
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_when_data_changes(synthetic_config: Path, synthetic_archive: Path):
    before = build_manifest(synthetic_config, data_root=synthetic_archive).content_hash()

    target = synthetic_archive / "Maps_Mtot_SuiteA_LH_z=0.00.npy"
    maps = np.load(target)
    maps[0, 0, 0] *= 2.0
    np.save(target, maps)

    after = build_manifest(synthetic_config, data_root=synthetic_archive).content_hash()
    assert before != after, "a modified map must change the manifest content hash"


def test_manifest_write_round_trips(synthetic_config: Path, synthetic_archive: Path, tmp_path: Path):
    m = build_manifest(synthetic_config, data_root=synthetic_archive)
    out = tmp_path / "manifests" / "local_manifest.json"
    m.write(out)

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["manifest_content_hash"] == m.content_hash()
    assert len(doc["suites"]) == 2
    # Section 13: never present these digests as official CAMELS checksums.
    assert "do not present these as official" in doc["note"]


def test_missing_file_raises_with_suite_name(synthetic_config: Path, synthetic_archive: Path):
    (synthetic_archive / "Maps_Mtot_SuiteB_LH_z=0.00.npy").unlink()
    with pytest.raises(FileNotFoundError, match="SuiteB"):
        build_manifest(synthetic_config, data_root=synthetic_archive)


def test_sampled_hash_detects_truncation(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x01" * 5000)
    original = sampled_sha256(p, n_samples=4)

    p.write_bytes(b"\x01" * 4999)
    assert sampled_sha256(p, n_samples=4) != original, "truncation must change the digest"


def test_sampled_hash_is_deterministic(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(bytes(range(256)) * 40)
    assert sampled_sha256(p, n_samples=4) == sampled_sha256(p, n_samples=4)


def test_full_hash_matches_known_value(tmp_path: Path):
    p = tmp_path / "known.txt"
    p.write_bytes(b"abc")
    # sha256("abc"), a published constant.
    assert file_sha256(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
