from __future__ import annotations

import re
from pathlib import Path


INVALID_PATH_CHARS_RE = re.compile(r'[\\/:*?"<>|\0]+')
SPACE_RE = re.compile(r"\s+")


def safe_path_name(value: str, fallback: str) -> str:
    cleaned = INVALID_PATH_CHARS_RE.sub("_", value or "")
    cleaned = SPACE_RE.sub(" ", cleaned).strip(" .")
    return cleaned or fallback


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_path(title: str, author: str = "") -> Path:
    out_dir = repo_root() / "epub"
    if author.strip():
        out_dir = out_dir / safe_path_name(author, "Unknown Author")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe_path_name(title, 'book')}.epub"


def resolve_output_path(output: str | Path | None, title: str, author: str = "") -> Path:
    out_path = Path(output) if output else default_output_path(title, author)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path
