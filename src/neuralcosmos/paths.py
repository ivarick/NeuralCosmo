"""Filesystem path resolution for NeuralCosmos.

Plan reference: section 10.

The location of the CAMELS archive is machine-specific and must never be baked
into the source tree. Resolution order is fixed and fails loudly:

    1. an explicit override (CLI flag or config field)
    2. the CAMELS_DATA_ROOT environment variable
    3. raise DataRootNotFound -- never guess

The repository root is derived from this file's own location, so the package
works the same whether it is run from the repo, from an editable install, or
from a different working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DataRootNotFound",
    "repo_root",
    "resolve_data_root",
    "outputs_dir",
    "reports_dir",
    "configs_dir",
]

_ENV_VAR = "CAMELS_DATA_ROOT"


class DataRootNotFound(RuntimeError):
    """Raised when the CAMELS archive location cannot be determined."""


def repo_root() -> Path:
    """Return the repository root directory.

    This file lives at ``<repo>/src/neuralcosmos/paths.py``, so the root is
    three levels up.
    """
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"


def outputs_dir() -> Path:
    return repo_root() / "outputs"


def reports_dir() -> Path:
    return repo_root() / "reports"


def resolve_data_root(override: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the root directory of the local CAMELS archive.

    Parameters
    ----------
    override:
        An explicit path from a CLI flag or config file. Takes precedence over
        the environment variable.

    Returns
    -------
    Path
        An existing directory.

    Raises
    ------
    DataRootNotFound
        If no source supplies a path, or the supplied path does not exist or is
        not a directory. The error message names the source that was tried so
        the failure is diagnosable without reading this code.
    """
    if override is not None:
        candidate = Path(override).expanduser()
        source = "explicit override"
    else:
        env_value = os.environ.get(_ENV_VAR)
        if not env_value:
            raise DataRootNotFound(
                f"No CAMELS data root configured. Set the {_ENV_VAR} environment "
                f"variable or pass an explicit path.\n"
                f"  PowerShell:  $env:{_ENV_VAR} = " + r"'E:\CAMELS_CMD'" + "\n"
                f"  bash:        export {_ENV_VAR}=/mnt/e/CAMELS_CMD"
            )
        candidate = Path(env_value).expanduser()
        source = f"${_ENV_VAR}"

    candidate = candidate.resolve()

    if not candidate.exists():
        raise DataRootNotFound(f"CAMELS data root from {source} does not exist: {candidate}")
    if not candidate.is_dir():
        raise DataRootNotFound(f"CAMELS data root from {source} is not a directory: {candidate}")

    return candidate
