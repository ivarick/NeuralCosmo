"""Path resolution must fail loudly, never guess.

Plan reference: section 10.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuralcosmos.paths import DataRootNotFound, repo_root, resolve_data_root


def test_repo_root_contains_project_marker():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "neuralcosmos").is_dir()


def test_missing_env_var_raises_rather_than_guessing(monkeypatch):
    monkeypatch.delenv("CAMELS_DATA_ROOT", raising=False)
    with pytest.raises(DataRootNotFound) as exc:
        resolve_data_root()
    # The message must name the variable so the failure is self-diagnosing.
    assert "CAMELS_DATA_ROOT" in str(exc.value)


def test_empty_env_var_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("CAMELS_DATA_ROOT", "")
    with pytest.raises(DataRootNotFound):
        resolve_data_root()


def test_env_var_is_used_when_set(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CAMELS_DATA_ROOT", str(tmp_path))
    assert resolve_data_root() == tmp_path.resolve()


def test_override_beats_env_var(monkeypatch, tmp_path: Path):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("CAMELS_DATA_ROOT", str(tmp_path))
    assert resolve_data_root(other) == other.resolve()


def test_nonexistent_path_raises(monkeypatch, tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setenv("CAMELS_DATA_ROOT", str(missing))
    with pytest.raises(DataRootNotFound, match="does not exist"):
        resolve_data_root()


def test_file_instead_of_directory_raises(tmp_path: Path):
    a_file = tmp_path / "not_a_dir.txt"
    a_file.write_text("x", encoding="utf-8")
    with pytest.raises(DataRootNotFound, match="not a directory"):
        resolve_data_root(a_file)
