from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

CRAWL_SCHEMA_VERSION = 1
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def deduplicate_comments(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for comment in comments:
        comment_id = clean_text(str(comment.get("id") or ""))
        if comment_id and comment_id in seen_ids:
            continue
        if comment_id:
            seen_ids.add(comment_id)
        unique.append(comment)
    return unique


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"crawl snapshot manifest not found: {manifest_path}")
    data = load_json(manifest_path)
    version = int(data.get("schema_version") or 0)
    if version != CRAWL_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported crawl snapshot schema {version}: {manifest_path}"
        )
    return data


def snapshot_chapter_ids(manifest: dict[str, Any]) -> list[str]:
    return [str(item["id"]) for item in manifest.get("chapters", [])]


def chapter_path(snapshot_dir: Path, manifest: dict[str, Any], chapter_id: str) -> Path:
    for item in manifest.get("chapters", []):
        if str(item.get("id")) == chapter_id:
            return snapshot_dir / str(item.get("path") or f"chapters/{chapter_id}.json")
    raise KeyError(f"chapter id not in manifest: {chapter_id}")


def comment_path(snapshot_dir: Path, manifest: dict[str, Any], chapter_id: str) -> Path:
    for item in manifest.get("chapters", []):
        if str(item.get("id")) == chapter_id:
            return snapshot_dir / str(
                item.get("comments_path") or f"comments/{chapter_id}.json"
            )
    raise KeyError(f"chapter id not in manifest: {chapter_id}")


def load_chapter(
    snapshot_dir: Path, manifest: dict[str, Any], chapter_id: str
) -> dict[str, Any]:
    return load_json(chapter_path(snapshot_dir, manifest, chapter_id))


def load_comments(
    snapshot_dir: Path, manifest: dict[str, Any], chapter_id: str
) -> list[dict[str, Any]]:
    path = comment_path(snapshot_dir, manifest, chapter_id)
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return data
    return list(data.get("comments") or [])


def block_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    return clean_text(soup.get_text(" ", strip=True))
