"""Procedencia reproducible: versión de código, de dataset y de librerías."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRACKED_LIBRARIES = (
    "numpy", "pandas", "scikit-learn", "scipy", "xgboost", "lightgbm", "catboost",
    "optuna", "shap", "mlflow", "fastapi", "streamlit",
)


def file_sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> dict[str, Any]:
    """Commit actual. Si no hay repo git, usa ``GIT_COMMIT``/``GIT_DIRTY`` del entorno."""
    env_commit = os.getenv("GIT_COMMIT")
    if env_commit:
        return {"commit": env_commit, "dirty": os.getenv("GIT_DIRTY", "").lower() == "true",
                "source": "environment"}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True,
            timeout=10,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty, "source": "git"}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None, "source": "not a git checkout"}


def library_versions(names: tuple[str, ...] = TRACKED_LIBRARIES) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def collect_provenance(dataset_path: Path | str, config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_commit(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": file_sha256(dataset_path),
            "size_bytes": dataset_path.stat().st_size,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": library_versions(),
        "configuration": config,
    }
