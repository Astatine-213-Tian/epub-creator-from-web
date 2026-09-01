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


def is_authoritative_comment(comment: dict[str, Any]) -> bool:
    author = clean_text(str(comment.get("author") or ""))
    return author.lower() in AUTHORITATIVE_AUTHORS or bool(
        comment.get("is_by_creator")
    )


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
    chinese_comments = [
        item for item in comments if has_cjk(str(item.get("body") or ""))
    ]
    by_id = {
        str(comment.get("id")): comment
        for comment in chinese_comments
        if comment.get("id")
    }
    replies_by_parent: dict[str, list[dict[str, Any]]] = {}
    for comment in chinese_comments:
        parent_id = str(comment.get("parent_id") or "")
        if parent_id and is_authoritative_comment(comment):
            replies_by_parent.setdefault(parent_id, []).append(comment)

    threads: list[tuple[tuple[int, str], list[dict[str, Any]]]] = []
    for parent_id, replies in replies_by_parent.items():
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        ordered_replies = sorted(replies, key=lambda item: str(item.get("created") or ""))
        priority = max(
            (comment_priority(reply, glossary_terms) for reply in ordered_replies),
            default=(0, ""),
        )
        threads.append((priority, [parent, *ordered_replies]))
    threads.sort(key=lambda item: item[0], reverse=True)

    picked: list[str] = []
    used = 0
    for _priority, thread in threads:
        notes: list[str] = []
        for comment in thread:
            body = clean_text(str(comment.get("body") or ""))
            if not body:
                continue
            author = clean_text(str(comment.get("author") or "")) or "unknown"
            parent = f" reply_to={comment['parent_id']}" if comment.get("parent_id") else ""
            notes.append(f"{author}{parent}: {body}")
        thread_size = sum(len(note) for note in notes)
        if picked and used + thread_size > char_budget:
            break
        picked.extend(notes)
        used += thread_size
    return picked
