from __future__ import annotations

import re
from typing import Any

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SPACE_RE = re.compile(r"\s+")
AUTHORITATIVE_AUTHORS = {"risk", "via lactea press inc.", "via lactea press"}


def clean_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def comment_priority(comment: dict[str, Any], glossary_terms: dict[str, str]) -> tuple[int, str]:
    author = clean_text(str(comment.get("author") or ""))
    body = clean_text(str(comment.get("body") or ""))
    text = author + "\n" + body
    priority = 0
    if author.lower() in AUTHORITATIVE_AUTHORS:
        priority += 100
    if comment.get("is_by_creator"):
        priority += 90
    if any(token in body for token in ("翻译", "中译", "原文", "译", "名字", "诗", "设定")):
        priority += 30
    if any(term and term in text for term in glossary_terms.values()):
        priority += 15
    if comment.get("is_liked_by_creator"):
        priority += 5
    return priority, str(comment.get("created") or "")


def selected_comment_notes(
    comments: list[dict[str, Any]],
    glossary_terms: dict[str, str],
    *,
    char_budget: int,
) -> list[str]:
    chinese_comments = [item for item in comments if has_cjk(str(item.get("body") or ""))]
    ranked = sorted(chinese_comments, key=lambda item: comment_priority(item, glossary_terms), reverse=True)
    picked: list[str] = []
    used = 0
    for comment in ranked:
        body = clean_text(str(comment.get("body") or ""))
        if not body:
            continue
        author = clean_text(str(comment.get("author") or "")) or "unknown"
        parent = f" reply_to={comment['parent_id']}" if comment.get("parent_id") else ""
        note = f"{author}{parent}: {body}"
        priority, _created = comment_priority(comment, glossary_terms)
        if used + len(note) > char_budget and priority < 80:
            break
        picked.append(note)
        used += len(note)
    return picked
