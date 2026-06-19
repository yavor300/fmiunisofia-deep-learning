"""Small project-local `.env` loader for secrets such as HF_TOKEN."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_env_file(env_path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from `.env` into `os.environ`."""
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ[key]
    return loaded


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    key, value = stripped.split("=", maxsplit=1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value
