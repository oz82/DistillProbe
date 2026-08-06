from __future__ import annotations

import datetime
import hashlib
import json
import platform
import socket
import subprocess
import sys
from typing import Any

_TRACKED_DEPS = [
    "model2vec",
    "gensim",
    "numpy",
    "scipy",
    "pydantic",
    "pyyaml",
    "matplotlib",
    "pandas",
    "sentence-transformers",
]


def _git_rev() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        rev = r.stdout.strip()
        return rev or None
    except Exception:
        return None


def _frozen_deps() -> dict[str, str]:
    import importlib.metadata

    out: dict[str, str] = {}
    for p in _TRACKED_DEPS:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            continue
    return out


def make_run_id(config: dict[str, Any]) -> str:
    """Run identifier: <UTC-timestamp>_<6-char config hash>."""
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{ts}_{h}"


def make_run_manifest(
    config: dict[str, Any], seed: int, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "seed": seed,
        "config": config,
        "git_rev": _git_rev(),
        "deps": _frozen_deps(),
    }
    if extra:
        manifest.update(extra)
    return manifest
