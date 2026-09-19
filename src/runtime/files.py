from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_BACKUP_DIR = Path("/private/tmp/epub-creator-from-web-codex-backups")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
