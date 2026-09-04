from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.crawl.snapshot import (
    clean_text,
    load_comments,
    load_manifest,
    snapshot_chapter_ids,
    write_json,
)


AUTHORITATIVE_AUTHORS = {"risk", "via lactea press inc.", "via lactea press"}
VERIFIED_TRANSLATION_STAFF_AUTHORS = {"pengiesama"}
COMMENT_EVIDENCE_SCHEMA_VERSION = 1
CJK_RE = re.compile(r"[\u3400-\u9fff]")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def is_authoritative_comment(comment: dict[str, Any]) -> bool:
    author = clean_text(str(comment.get("author") or ""))
    return (
        author.lower()
        in AUTHORITATIVE_AUTHORS | VERIFIED_TRANSLATION_STAFF_AUTHORS
        or bool(comment.get("is_by_creator"))
    )


def _evidence_comment(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": clean_text(str(comment.get("id") or "")),
        "parent_id": clean_text(str(comment.get("parent_id") or "")) or None,
        "author": clean_text(str(comment.get("author") or "")) or "unknown",
        "body": clean_text(str(comment.get("body") or "")),
        "created": clean_text(str(comment.get("created") or "")) or None,
    }


def authoritative_reply_threads(
    comments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {
        str(comment.get("id")): comment
        for comment in comments
        if comment.get("id")
    }
    replies_by_parent: dict[str, list[dict[str, Any]]] = {}
    for comment in comments:
        parent_id = str(comment.get("parent_id") or "")
        if (
            parent_id
            and has_cjk(clean_text(str(comment.get("body") or "")))
            and is_authoritative_comment(comment)
        ):
            replies_by_parent.setdefault(parent_id, []).append(comment)

    threads: list[dict[str, Any]] = []
    for parent_id, replies in replies_by_parent.items():
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        ordered_replies = sorted(
            replies,
            key=lambda item: str(item.get("created") or ""),
        )
        threads.append(
            {
                "parent": _evidence_comment(parent),
                "authoritative_replies": [
                    _evidence_comment(reply) for reply in ordered_replies
                ],
            }
        )
    threads.sort(
        key=lambda thread: str(
            thread["authoritative_replies"][-1].get("created") or ""
        ),
        reverse=True,
    )
    return threads


def write_glossary_comment_evidence(
    *,
    snapshot_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    snapshot_dir = snapshot_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    manifest = load_manifest(snapshot_dir)
    chapters: list[dict[str, Any]] = []
    thread_count = 0
    reply_count = 0

    chapter_metadata = {
        str(chapter["id"]): chapter for chapter in manifest.get("chapters") or []
    }
    for chapter_id in snapshot_chapter_ids(manifest):
        threads = authoritative_reply_threads(
            load_comments(snapshot_dir, manifest, chapter_id)
        )
        if not threads:
            continue
        metadata = chapter_metadata[chapter_id]
        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_title": str(metadata.get("title") or ""),
                "source_id": str(metadata.get("source_id") or ""),
                "source_url": str(metadata.get("source_url") or ""),
                "threads": threads,
            }
        )
        thread_count += len(threads)
        reply_count += sum(
            len(thread["authoritative_replies"]) for thread in threads
        )

    evidence = {
        "schema_version": COMMENT_EVIDENCE_SCHEMA_VERSION,
        "purpose": "review_for_glossary",
        "snapshot_dir": str(snapshot_dir),
        "review_required": True,
        "consumed_by_translation_prompts": False,
        "thread_count": thread_count,
        "authoritative_reply_count": reply_count,
        "chapters": chapters,
    }
    write_json(output_path, evidence)
    return evidence
