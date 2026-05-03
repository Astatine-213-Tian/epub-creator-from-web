from __future__ import annotations

import os
import shlex
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs from the project .env file.

    Existing environment variables are not overwritten.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _parse_env_value(value.strip())


def _parse_env_value(value: str) -> str:
    if not value:
        return ""
    try:
        parts = shlex.split(value, comments=False, posix=True)
    except ValueError:
        return value.strip("\"'")
    if len(parts) == 1:
        return parts[0]
    return value.strip("\"'")
