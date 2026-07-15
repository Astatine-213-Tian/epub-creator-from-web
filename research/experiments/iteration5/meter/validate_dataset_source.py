#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CHAPTER_HEADING_RE = re.compile(
    r"^第[0-9零一二三四五六七八九十百千万两〇]+章(?:\s|、|$)"
)
EURO_RUN_RE = re.compile(r"€{2,}")
MALFORMED_COMMA_QUESTION_RE = re.compile(r",\?")
OBFUSCATED_AD_RE = re.compile(r"\[\(\)\].{0,120}(?:ｃｏｍ|com)", re.I)
AUTHOR_NOTE_RE = re.compile(r"作者(?:有话要说|的话)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text))


def chapter_sections(lines: list[str]) -> list[tuple[str, str]]:
    headings = [index for index, line in enumerate(lines) if CHAPTER_HEADING_RE.match(line.strip())]
    sections: list[tuple[str, str]] = []
    for position, start in enumerate(headings):
        end = headings[position + 1] if position + 1 < len(headings) else len(lines)
        sections.append((lines[start].strip(), "\n".join(lines[start + 1 : end])))
    return sections


def analyze_source(
    path: Path,
    *,
    expected_author: str,
    accepted_titles: set[str],
    expected_chapters: int | None,
    min_cjk: int,
) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    nonblank = [line.strip() for line in lines if line.strip()]
    source_title = nonblank[0] if nonblank else ""
    author_header = next((line for line in nonblank[:12] if line.startswith("作者：")), "")
    source_author = author_header.removeprefix("作者：").strip()
    sections = chapter_sections(lines)
    heading_counts = Counter(heading for heading, _ in sections)
    duplicate_headings = sorted(heading for heading, count in heading_counts.items() if count > 1)
    short_chapters = [
        {"heading": heading, "cjk": cjk_len(body)}
        for heading, body in sections
        if cjk_len(body) < 100
    ]
    metrics = {
        "bytes": path.stat().st_size,
        "lines": len(lines),
        "cjk": cjk_len(text),
        "chapters": len(sections),
        "duplicate_chapter_headings": len(duplicate_headings),
        "short_chapters_under_100_cjk": len(short_chapters),
        "euro_runs": len(EURO_RUN_RE.findall(text)),
        "unicode_replacement_chars": text.count("�"),
        "malformed_comma_question": len(MALFORMED_COMMA_QUESTION_RE.findall(text)),
        "obfuscated_ads": len(OBFUSCATED_AD_RE.findall(text)),
        "author_note_markers": len(AUTHOR_NOTE_RE.findall(text)),
    }
    fatal: list[str] = []
    if source_title not in accepted_titles:
        fatal.append(f"source title {source_title!r} is not accepted")
    if source_author != expected_author:
        fatal.append(f"source author {source_author!r} != {expected_author!r}")
    if metrics["cjk"] < min_cjk:
        fatal.append(f"cleanable CJK volume {metrics['cjk']} < {min_cjk}")
    if expected_chapters is not None and len(sections) != expected_chapters:
        fatal.append(f"chapter count {len(sections)} != {expected_chapters}")
    for key in ("euro_runs", "unicode_replacement_chars", "malformed_comma_question", "obfuscated_ads"):
        if metrics[key]:
            fatal.append(f"{key}={metrics[key]}")
    if duplicate_headings:
        fatal.append(f"duplicate chapter headings={len(duplicate_headings)}")
    if len(short_chapters) > max(2, len(sections) // 50):
        fatal.append(f"excess short chapters={len(short_chapters)}")
    return {
        "schema_version": 1,
        "path": str(path),
        "sha256": sha256_file(path),
        "source_title": source_title,
        "source_author": source_author,
        "accepted_titles": sorted(accepted_titles),
        "expected_author": expected_author,
        "expected_chapters": expected_chapters,
        "metrics": metrics,
        "duplicate_headings": duplicate_headings[:20],
        "short_chapters": short_chapters[:20],
        "fatal_reasons": fatal,
        "status": "pass" if not fatal else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a TXT source before dataset promotion.")
    parser.add_argument("txt", type=Path)
    parser.add_argument("--author", required=True)
    parser.add_argument("--accepted-title", action="append", required=True)
    parser.add_argument("--expected-chapters", type=int)
    parser.add_argument("--min-cjk", type=int, default=50_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_source(
        args.txt,
        expected_author=args.author,
        accepted_titles=set(args.accepted_title),
        expected_chapters=args.expected_chapters,
        min_cjk=args.min_cjk,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
