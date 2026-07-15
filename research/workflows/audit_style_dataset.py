#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")
CHAPTER_RE = re.compile(
    r"^\s*(?:"
    r"第[0-9零一二三四五六七八九十百千万两〇]+[章节卷部篇回]"
    r"|正文"
    r"|楔子"
    r"|引子"
    r"|序章"
    r"|序"
    r"|番外"
    r"|尾声"
    r"|后记"
    r")\b"
)
TITLE_PREFIX_RE = re.compile(r"^\s*(?:书名|作品|小说|文名|标题)[:：]")
AUTHOR_RE = re.compile(r"^\s*作者[:：]")
AUTHOR_HEADER_RE = re.compile(r"^\s*作者\s*[:：]\s*(?P<author>.+?)\s*$")
INTRO_RE = re.compile(r"^\s*(?:简介|文案|内容简介|作品简介)[:：]?\s*$")
TAG_RE = re.compile(r"^\s*(?:标签|关键字|关键词)[:：]")
URL_RE = re.compile(r"https?://|www\.|\.com|\.net|\.org", re.I)
BOILERPLATE_RE = re.compile(
    r"晋江文学城|jjwxc|请收藏|收藏此文章|霸王票|手机阅读|"
    r"最新网址|返回目录|书友群|盗文|防盗|点击下一章|上一章|下一章|"
    r"本书由|整理制作|TXT下载|电子书|更多精彩|营养液加更|感谢.*营养液"
)
INLINE_AD_RE = re.compile(
    r"您下载的文件由.*?(?:更多好看小说哦！|更多好看小说哦!|$)"
)
OBFUSCATED_INLINE_AD_RE = re.compile(
    r"[^\u3400-\u9fffA-Za-z0-9\s，。！？；：、“”‘’《》【】()\[\]（）]{1,8}"
    r"[^\n]{0,160}?"
    r"(?:[（(][ｃcＣ][ｏoＯ][ｍmＭ][）)]|[ｃcＣ][ｏoＯ][ｍmＭ][）)]?)",
    re.I,
)
SPLIT_DOMAIN_AD_RE = re.compile(
    r"(?:[（(][A-Za-zＡ-Ｚａ-ｚ0-9０-９._-]{1,40}[）)]){1,3}"
    r"[（(][ｃcＣ][ｏoＯ](?:[ｍmＭ])?[）)]",
    re.I,
)
AUTHOR_NOTE_RE = re.compile(r"^\s*[【\[]?\s*作者(?:有话要说|的话)\s*[】\]]?[:：]?")
REPLACEMENT_ARTIFACT_RE = re.compile(r"€{2,}")
UNICODE_REPLACEMENT_CHAR = "�"
SYSTEMIC_REPLACEMENT_MIN_HITS = 10
SYSTEMIC_REPLACEMENT_MIN_LINES = 5


AUTHOR_ALIASES = {
    "顾雪柔": "非天夜翔",
}

CHUNK_TARGET_CJK = 1_500
CHUNK_MIN_CJK = 800
MASK_NGRAM_MIN_COUNT = 4
MAX_ENTITY_TERMS_PER_BOOK = 240
MAX_ENTITY_V2_TERMS_PER_BOOK = 800
MAX_TOPIC_TERMS_PER_BOOK = 640
MASK_NGRAM_MIN_N = 2
MASK_NGRAM_MAX_N = 5

MASK_TERM_STOP_CHARS = set(
    "的一是在不了有和人这中大为上个我以要他时来用们生到作地于出就分对成会可主"
    "发年动同工也能下过子说产种面而方后多定行学法所民得经之进着等部度家"
    "电力里如水化高自理起小物现实加都两体制机当使点从业本去把性好应开它"
    "合还因由其些然前外天那与关各重新线内正心反你看又么但向道此只没给被"
    "很最才并已让"
)
FUNCTION_STYLE_CHARS = set(
    "的一是在不了有和人这中为上个我以要他时来用们到地于出就对成会可也能下过"
    "而后定行得经之着等里如自起把性好应开还因由其些然前那与关各并已又但"
    "只没给被很最才让吗呢啊吧呀么着了过"
)



def cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text))


def clean_author(author: str) -> str:
    return AUTHOR_ALIASES.get(author, author)


def safe_id(text: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text.strip())
    text = re.sub(r"\s+", "_", text)
    return text or "untitled"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def normalize_lines(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank:
                collapsed.append("")
            blank = True
            continue
        collapsed.append(line.strip())
        blank = False
    while collapsed and not collapsed[0]:
        collapsed.pop(0)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return collapsed


def find_body_start(lines: list[str], title: str) -> int:
    for index, line in enumerate(lines[:240]):
        if CHAPTER_RE.search(line):
            return index
    skip = 0
    for index, line in enumerate(lines[:40]):
        stripped = line.strip()
        if not stripped:
            skip = index + 1
            continue
        if stripped == title or TITLE_PREFIX_RE.search(stripped) or AUTHOR_RE.search(stripped):
            skip = index + 1
            continue
        if INTRO_RE.search(stripped) or TAG_RE.search(stripped):
            skip = index + 1
            continue
        break
    return skip


def source_author_headers(raw_text: str, *, title: str) -> list[str]:
    """Return plausible author metadata found before the first body chapter."""
    lines = normalize_lines(raw_text)
    body_start = find_body_start(lines, title)
    headers: list[str] = []
    for line in lines[:body_start]:
        match = AUTHOR_HEADER_RE.match(line)
        if not match:
            continue
        value = match.group("author").strip()
        if not re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9_.·-]{1,30}", value):
            continue
        normalized = clean_author(value)
        if normalized not in headers:
            headers.append(normalized)
    return headers


def filter_body_lines(lines: list[str]) -> list[str]:
    filtered: list[str] = []
    skipping_author_note = False
    previous_blank = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            skipping_author_note = False
            if not previous_blank:
                filtered.append("")
            previous_blank = True
            continue
        if CHAPTER_RE.search(stripped):
            skipping_author_note = False
            if filtered and not previous_blank:
                filtered.append("")
                previous_blank = True
            continue
        if AUTHOR_NOTE_RE.search(stripped):
            skipping_author_note = True
            continue
        if skipping_author_note:
            continue
        if URL_RE.search(stripped):
            continue
        if BOILERPLATE_RE.search(stripped):
            continue
        stripped = INLINE_AD_RE.sub("", stripped).strip()
        stripped = SPLIT_DOMAIN_AD_RE.sub("", stripped).strip()
        stripped = OBFUSCATED_INLINE_AD_RE.sub("", stripped).strip()
        stripped = stripped.replace(UNICODE_REPLACEMENT_CHAR, "")
        if not stripped:
            continue
        filtered.append(stripped)
        previous_blank = False
    while filtered and not filtered[0]:
        filtered.pop(0)
    while filtered and not filtered[-1]:
        filtered.pop()
    return filtered


def cleaned_text(raw_text: str, *, title: str) -> tuple[str, dict[str, Any]]:
    lines = normalize_lines(raw_text)
    body_start = find_body_start(lines, title)
    body_lines = filter_body_lines(lines[body_start:])
    cleaned = "\n".join(body_lines).strip() + "\n" if body_lines else ""
    return cleaned, {
        "raw_line_count": len(lines),
        "body_start_line": body_start + 1 if lines else 0,
        "dropped_leading_lines": body_start,
    }


def chunk_text(text: str, *, target_cjk: int = CHUNK_TARGET_CJK, min_cjk: int = CHUNK_MIN_CJK) -> list[str]:
    paragraphs = [line for line in normalize_lines(text) if line.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_cjk = 0
    for paragraph in paragraphs:
        paragraph_cjk = cjk_len(paragraph)
        if current and current_cjk >= min_cjk and current_cjk + paragraph_cjk > target_cjk:
            chunks.append("\n".join(current).strip())
            current = []
            current_cjk = 0
        current.append(paragraph)
        current_cjk += paragraph_cjk
        if current_cjk >= target_cjk:
            chunks.append("\n".join(current).strip())
            current = []
            current_cjk = 0
    if current:
        tail = "\n".join(current).strip()
        if chunks and cjk_len(tail) < min_cjk:
            chunks[-1] = f"{chunks[-1]}\n{tail}".strip()
        else:
            chunks.append(tail)
    return [chunk for chunk in chunks if chunk]


def cjk_ngrams(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for match in CJK_RUN_RE.finditer(text):
        run = match.group(0)
        for size in range(MASK_NGRAM_MIN_N, MASK_NGRAM_MAX_N + 1):
            if len(run) < size:
                continue
            for index in range(0, len(run) - size + 1):
                term = run[index:index + size]
                if is_mask_candidate(term):
                    counts[term] += 1
    return counts


def is_mask_candidate(term: str) -> bool:
    return (
        MASK_NGRAM_MIN_N <= len(term) <= MASK_NGRAM_MAX_N
        and bool(CJK_RUN_RE.fullmatch(term))
        and len(set(term)) > 1
        and not any(char in MASK_TERM_STOP_CHARS for char in term)
    )


def title_mask_terms(title: str) -> set[str]:
    terms: set[str] = set()
    for run in CJK_RUN_RE.findall(title):
        if len(run) >= MASK_NGRAM_MIN_N:
            terms.add(run)
        for size in range(MASK_NGRAM_MIN_N, min(MASK_NGRAM_MAX_N, len(run)) + 1):
            for index in range(0, len(run) - size + 1):
                term = run[index:index + size]
                if len(set(term)) > 1:
                    terms.add(term)
    return terms


def record_key(record: dict[str, Any]) -> str:
    return f"{record['author']}::{record['title']}"


def select_mask_terms(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_book_counts: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    term_authors: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if not record.get("exists") or not record.get("clean_txt_path"):
            continue
        text_path = Path(str(record["clean_txt_path"]))
        if not text_path.exists():
            continue
        counts = Counter({
            term: count
            for term, count in cjk_ngrams(read_text(text_path)).items()
            if count >= MASK_NGRAM_MIN_COUNT
        })
        key = record_key(record)
        per_book_counts[key] = counts
        document_frequency.update(counts.keys())
        for term in counts:
            term_authors[term].add(str(record["author"]))

    book_count = max(len(per_book_counts), 1)
    entity_max_df = max(3, book_count // 20)
    topic_max_df = max(6, book_count // 10)
    author_frequency = {term: len(authors) for term, authors in term_authors.items()}
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        counts = per_book_counts.get(key, Counter())
        title_terms = title_mask_terms(str(record["title"]))

        def ranked_terms(max_df: int, limit: int, *, allow_author_concentrated: bool = False) -> list[str]:
            ranked = sorted(
                (
                    (term, count, document_frequency[term])
                    for term, count in counts.items()
                    if document_frequency[term] <= max_df
                    or (allow_author_concentrated and author_frequency.get(term, 0) <= 1)
                ),
                key=lambda item: (
                    -(item[1] * len(item[0]) / max(author_frequency.get(item[0], item[2]), 1)),
                    -len(item[0]),
                    item[0],
                ),
            )
            terms = list(title_terms)
            seen = set(terms)
            for term, _count, _df in ranked:
                if term in seen:
                    continue
                terms.append(term)
                seen.add(term)
                if len(terms) >= limit:
                    break
            return sorted(terms, key=lambda item: (-len(item), item))

        selected[key] = {
            "author": record["author"],
            "title": record["title"],
            "candidate_term_count": len(counts),
            "entity_terms": ranked_terms(entity_max_df, MAX_ENTITY_TERMS_PER_BOOK),
            "entity_terms_v2": ranked_terms(
                entity_max_df,
                MAX_ENTITY_V2_TERMS_PER_BOOK,
                allow_author_concentrated=True,
            ),
            "topic_terms": ranked_terms(topic_max_df, MAX_TOPIC_TERMS_PER_BOOK),
        }
    return selected


def normalize_non_cjk_content(text: str) -> str:
    text = LATIN_RE.sub("<LATIN>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    return re.sub(r"<+(CONTENT|NUM|LATIN)>+", r"<\1>", text)


def compile_term_pattern(terms: list[str]) -> re.Pattern[str] | None:
    if not terms:
        return None
    return re.compile("|".join(re.escape(term) for term in sorted(terms, key=lambda item: (-len(item), item))))


def mask_terms(text: str, pattern: re.Pattern[str] | None, *, placeholder: str, preserve_length: bool = False) -> str:
    if pattern is None:
        return text
    if preserve_length:
        return pattern.sub(lambda match: placeholder * cjk_len(match.group(0)), text)
    return pattern.sub(placeholder, text)


def entity_masked_text(text: str, pattern: re.Pattern[str] | None) -> str:
    text = normalize_non_cjk_content(text)
    return mask_terms(text, pattern, placeholder="<CONTENT>")


def entity_masked_v2_text(text: str, pattern: re.Pattern[str] | None) -> str:
    text = normalize_non_cjk_content(text)
    return mask_terms(text, pattern, placeholder="<TERM>")


def entity_masked_v3_text(text: str, pattern: re.Pattern[str] | None) -> str:
    text = normalize_non_cjk_content(text)
    return mask_terms(text, pattern, placeholder="某", preserve_length=True)


def topic_distorted_text(text: str, pattern: re.Pattern[str] | None) -> str:
    text = normalize_non_cjk_content(text)
    text = mask_terms(text, pattern, placeholder="文", preserve_length=True)
    return CJK_RE.sub(lambda match: match.group(0) if match.group(0) in FUNCTION_STYLE_CHARS else "文", text)


def structure_only_text(text: str) -> str:
    text = normalize_non_cjk_content(text)
    return CJK_RE.sub("文", text)


def duplicate_line_count(lines: list[str]) -> int:
    previous = ""
    duplicates = 0
    for line in lines:
        if line and line == previous:
            duplicates += 1
        previous = line
    return duplicates


def text_flags(raw: str, cleaned: str) -> list[str]:
    flags: list[str] = []
    clean_len = cjk_len(cleaned)
    raw_len = cjk_len(raw)
    if clean_len < 50_000:
        flags.append("too_short_for_primary_benchmark")
    elif clean_len < 120_000:
        flags.append("short_text")
    if raw_len and clean_len / raw_len < 0.75:
        flags.append("large_header_or_nonbody_drop")
    if URL_RE.search(raw):
        flags.append("raw_url_or_domain_text")
    if URL_RE.search(cleaned):
        flags.append("clean_url_or_domain_text")
    if BOILERPLATE_RE.search(raw):
        flags.append("raw_boilerplate_signal")
    if BOILERPLATE_RE.search(cleaned):
        flags.append("clean_boilerplate_signal")
    lines = normalize_lines(cleaned)
    if duplicate_line_count(lines) > 0:
        flags.append("adjacent_duplicate_lines")
    if "作者有话要说" in raw or "作者的话" in raw:
        flags.append("raw_author_notes_present")
    if "作者有话要说" in cleaned or "作者的话" in cleaned:
        flags.append("clean_author_notes_present")
    if re.search(r"[A-Za-z]{30,}", raw):
        flags.append("long_latin_span")
    if UNICODE_REPLACEMENT_CHAR in raw:
        flags.append("raw_unicode_replacement_char")
    replacement_hits = len(REPLACEMENT_ARTIFACT_RE.findall(cleaned))
    replacement_lines = sum(
        1 for line in cleaned.splitlines() if REPLACEMENT_ARTIFACT_RE.search(line)
    )
    if replacement_hits:
        flags.append("clean_replacement_artifacts")
    if (
        replacement_hits >= SYSTEMIC_REPLACEMENT_MIN_HITS
        or replacement_lines >= SYSTEMIC_REPLACEMENT_MIN_LINES
    ):
        flags.append("systemic_encoding_corruption")
    return flags


def book_record(row: dict[str, Any], dataset_root: Path, output_text_root: Path) -> dict[str, Any]:
    txt_path = Path(str(row.get("txt_path") or ""))
    if txt_path.is_absolute() and not txt_path.exists() and "datasets" in txt_path.parts:
        marker = len(txt_path.parts) - 1 - txt_path.parts[::-1].index("datasets")
        txt_path = dataset_root.joinpath(*txt_path.parts[marker + 1 :])
    elif not txt_path.is_absolute():
        candidates = (
            dataset_root / txt_path,
            dataset_root.parent / txt_path,
            dataset_root.parent.parent / txt_path,
            txt_path,
        )
        txt_path = next(
            (candidate for candidate in candidates if candidate.exists()),
            candidates[0],
        )
    author = clean_author(str(row.get("author") or ""))
    title = str(row.get("title") or txt_path.stem)
    exists = txt_path.exists()
    raw = read_text(txt_path) if exists else ""
    cleaned, clean_meta = cleaned_text(raw, title=title) if exists else ("", {})
    detected_authors = source_author_headers(raw, title=title) if exists else []
    author_conflicts = [value for value in detected_authors if value != author]
    raw_cjk = cjk_len(raw)
    clean_cjk = cjk_len(cleaned)
    sha = hashlib.sha256(cleaned.encode("utf-8")).hexdigest() if cleaned else ""
    out_path = output_text_root / safe_id(author) / f"{safe_id(title)}.clean.txt"
    quality_flags = text_flags(raw, cleaned) if exists else ["missing_txt"]
    replacement_artifact_hits = len(REPLACEMENT_ARTIFACT_RE.findall(cleaned))
    replacement_artifact_lines = sum(
        1 for line in cleaned.splitlines() if REPLACEMENT_ARTIFACT_RE.search(line)
    )
    if author_conflicts:
        quality_flags.append("manifest_author_header_conflict")
    record = {
        "title": title,
        "author": author,
        "original_author": str(row.get("author") or ""),
        "txt_path": str(txt_path),
        "clean_txt_path": str(out_path),
        "source_epub": row.get("source_epub") or "",
        "metadata_source": row.get("metadata_source") or "",
        "time_area": row.get("time_area") or "",
        "genre": row.get("genre") or "",
        "article_type": row.get("article_type") or "",
        "jjwxc_url": row.get("jjwxc_url") or "",
        "status": row.get("status") or "",
        "manifest_char_count": int(row.get("char_count") or 0),
        "raw_cjk_count": raw_cjk,
        "clean_cjk_count": clean_cjk,
        "clean_sha256": sha,
        "exists": exists,
        "source_author_headers": detected_authors,
        "author_header_conflicts": author_conflicts,
        "quality_flags": quality_flags,
        "replacement_artifact_hits": replacement_artifact_hits,
        "replacement_artifact_lines": replacement_artifact_lines,
        "raw_unicode_replacement_char_count": raw.count(UNICODE_REPLACEMENT_CHAR),
        **clean_meta,
    }
    if exists:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(cleaned, encoding="utf-8")
    return record


def is_primary_eligible(item: dict[str, Any]) -> bool:
    return bool(
        item["exists"]
        and item["clean_cjk_count"] >= 50_000
        and "manifest_author_header_conflict" not in item["quality_flags"]
        and "systemic_encoding_corruption" not in item["quality_flags"]
    )


def split_books(records: list[dict[str, Any]], target_author: str) -> dict[str, Any]:
    eligible = [item for item in records if is_primary_eligible(item)]
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_author[item["author"]].append(item)

    splits = {"train": [], "dev": [], "test": [], "proxy_transfer": [], "excluded": []}
    for author, books in sorted(by_author.items()):
        ordered = sorted(
            books,
            key=lambda item: (-int(item["clean_cjk_count"]), str(item["title"])),
        )
        if author == target_author:
            proxy_n = min(4, max(2, len(ordered) // 10)) if len(ordered) >= 12 else 0
            dev_n = min(4, max(1, len(ordered) // 10)) if len(ordered) >= 5 else 1
            test_n = min(4, max(1, len(ordered) // 10)) if len(ordered) >= 5 else 1
            proxy = ordered[:proxy_n]
            test = ordered[proxy_n:proxy_n + test_n]
            dev = ordered[proxy_n + test_n:proxy_n + test_n + dev_n]
            train = ordered[proxy_n + test_n + dev_n:]
        elif len(ordered) >= 5:
            test = ordered[:1]
            dev = ordered[1:2]
            proxy = []
            train = ordered[2:]
        elif len(ordered) >= 3:
            test = ordered[:1]
            dev = ordered[1:2]
            proxy = []
            train = ordered[2:]
        else:
            continue
        splits["train"].extend(split_entry(item) for item in train)
        splits["dev"].extend(split_entry(item) for item in dev)
        splits["test"].extend(split_entry(item) for item in test)
        splits["proxy_transfer"].extend(split_entry(item) for item in proxy)

    included_ids = {
        (entry["author"], entry["title"])
        for split_name in ("train", "dev", "test", "proxy_transfer")
        for entry in splits[split_name]
    }
    for item in records:
        key = (item["author"], item["title"])
        if key not in included_ids and item["exists"]:
            reasons = []
            if item["clean_cjk_count"] < 50_000:
                reasons.append("too_short_for_primary_benchmark")
            if "manifest_author_header_conflict" in item["quality_flags"]:
                reasons.append("manifest_author_header_conflict")
            if "systemic_encoding_corruption" in item["quality_flags"]:
                reasons.append("systemic_encoding_corruption")
            author_eligible = len(by_author.get(item["author"], []))
            if author_eligible < 3:
                reasons.append("insufficient_books_for_book_level_split")
            if reasons:
                splits["excluded"].append(split_entry(item, ",".join(sorted(set(reasons)))))
    return splits


def split_entry(item: dict[str, Any], reason: str = "") -> dict[str, Any]:
    return {
        "author": item["author"],
        "title": item["title"],
        "clean_txt_path": item["clean_txt_path"],
        "clean_cjk_count": item["clean_cjk_count"],
        "quality_flags": item["quality_flags"],
        "reason": reason,
    }


def summarize(records: list[dict[str, Any]], splits: dict[str, Any]) -> dict[str, Any]:
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        by_author[item["author"]].append(item)
    author_summary = []
    for author, books in sorted(by_author.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        usable = [item for item in books if is_primary_eligible(item)]
        primary = [
            item for item in books
            if is_primary_eligible(item) and item["clean_cjk_count"] >= 120_000
        ]
        author_summary.append(
            {
                "author": author,
                "book_count": len(books),
                "usable_book_count_50k": len(usable),
                "primary_book_count_120k": len(primary),
                "clean_cjk_total": sum(int(item["clean_cjk_count"]) for item in books),
                "flags": dict(Counter(flag for item in books for flag in item["quality_flags"])),
            }
        )
    return {
        "book_count": len(records),
        "author_count": len(by_author),
        "txt_missing_count": sum(1 for item in records if not item["exists"]),
        "usable_book_count_50k": sum(1 for item in records if is_primary_eligible(item)),
        "primary_book_count_120k": sum(
            1
            for item in records
            if is_primary_eligible(item) and item["clean_cjk_count"] >= 120_000
        ),
        "quality_flag_counts": dict(Counter(flag for item in records for flag in item["quality_flags"])),
        "authors_with_3plus_usable_books": sum(1 for item in author_summary if item["usable_book_count_50k"] >= 3),
        "authors_with_5plus_usable_books": sum(1 for item in author_summary if item["usable_book_count_50k"] >= 5),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "author_summary": author_summary,
    }


def duplicate_report(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        if item.get("clean_sha256"):
            by_sha[str(item["clean_sha256"])].append(item)
    duplicates = []
    for sha, items in by_sha.items():
        if len(items) > 1:
            duplicates.append(
                {
                    "sha256": sha,
                    "books": [
                        {
                            "author": item["author"],
                            "title": item["title"],
                            "clean_txt_path": item["clean_txt_path"],
                        }
                        for item in items
                    ],
                }
            )
    return duplicates


def prune_stale_cleaned_texts(
    output_text_root: Path, records: list[dict[str, Any]]
) -> list[str]:
    expected = {
        Path(str(record["clean_txt_path"])).resolve()
        for record in records
        if record.get("exists") and record.get("clean_txt_path")
    }
    removed: list[str] = []
    for path in sorted(output_text_root.glob("**/*.clean.txt")):
        if path.resolve() in expected:
            continue
        path.unlink()
        removed.append(str(path))
    for directory in sorted(output_text_root.glob("**/*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return removed


def split_lookup(splits: dict[str, Any]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for split_name, entries in splits.items():
        for entry in entries:
            lookup[(str(entry["author"]), str(entry["title"]))] = split_name
    return lookup


def chunk_record(
    record: dict[str, Any],
    *,
    chunk_id: str,
    split_name: str,
    chunk_index: int,
    clean_cjk_count: int,
    view: str,
    view_text: str,
    view_cjk_count: int,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "view": view,
        "split": split_name,
        "author": record["author"],
        "title": record["title"],
        "chunk_index": chunk_index,
        "book_clean_cjk_count": record["clean_cjk_count"],
        "chunk_clean_cjk_count": clean_cjk_count,
        "chunk_view_cjk_count": view_cjk_count,
        "time_area": record.get("time_area") or "",
        "genre": record.get("genre") or "",
        "article_type": record.get("article_type") or "",
        "quality_flags": record.get("quality_flags") or [],
        "text": view_text,
    }


def generate_chunk_views(
    records: list[dict[str, Any]],
    splits: dict[str, Any],
    mask_plan: dict[str, dict[str, Any]],
    dataset_root: Path,
    *,
    chunk_target_cjk: int = CHUNK_TARGET_CJK,
    chunk_min_cjk: int = CHUNK_MIN_CJK,
    include_excluded: bool = False,
) -> dict[str, Any]:
    unmasked_dir = dataset_root / "unmasked"
    masked_dir = dataset_root / "masked"
    unmasked_dir.mkdir(parents=True, exist_ok=True)
    masked_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "clean": unmasked_dir / "chunks.clean.jsonl",
        "entity_masked": masked_dir / "chunks.entity_masked.jsonl",
        "entity_masked_v2": masked_dir / "chunks.entity_masked_v2.jsonl",
        "entity_masked_v3": masked_dir / "chunks.entity_masked_v3.jsonl",
        "topic_distorted": masked_dir / "chunks.topic_distorted.jsonl",
        "structure_only": masked_dir / "chunks.structure_only.jsonl",
    }
    handles = {
        view: path.open("w", encoding="utf-8")
        for view, path in output_paths.items()
    }
    split_names = split_lookup(splits)
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "chunk_target_cjk": chunk_target_cjk,
        "chunk_min_cjk": chunk_min_cjk,
        "include_excluded": include_excluded,
        "books_chunked": 0,
        "chunks_by_view": Counter(),
        "chunks_by_split": Counter(),
        "output_paths": {view: str(path) for view, path in output_paths.items()},
    }

    try:
        for record in records:
            if not record.get("exists") or not record.get("clean_txt_path"):
                continue
            text_path = Path(str(record["clean_txt_path"]))
            if not text_path.exists():
                continue
            clean_text = read_text(text_path)
            split_name = split_names.get((str(record["author"]), str(record["title"])), "excluded")
            if split_name == "excluded" and not include_excluded:
                continue
            clean_chunks = chunk_text(clean_text, target_cjk=chunk_target_cjk, min_cjk=chunk_min_cjk)
            if not clean_chunks:
                continue
            summary["books_chunked"] += 1
            plan = mask_plan.get(record_key(record), {})
            entity_pattern = compile_term_pattern(list(plan.get("entity_terms") or []))
            entity_v2_pattern = compile_term_pattern(list(plan.get("entity_terms_v2") or []))
            topic_pattern = compile_term_pattern(list(plan.get("topic_terms") or []))

            for index, clean_chunk in enumerate(clean_chunks, start=1):
                chunk_id = f"{safe_id(str(record['author']))}__{safe_id(str(record['title']))}__{index:04d}"
                clean_cjk_count = cjk_len(clean_chunk)
                entity_text = entity_masked_text(clean_chunk, entity_pattern)
                entity_v2_text = entity_masked_v2_text(clean_chunk, entity_v2_pattern)
                entity_v3_text = entity_masked_v3_text(clean_chunk, entity_v2_pattern)
                views = {
                    "clean": clean_chunk,
                    "entity_masked": entity_text,
                    "entity_masked_v2": entity_v2_text,
                    "entity_masked_v3": entity_v3_text,
                    "topic_distorted": topic_distorted_text(clean_chunk, topic_pattern),
                    "structure_only": structure_only_text(clean_chunk),
                }
                view_cjk_counts = {
                    "clean": clean_cjk_count,
                    "entity_masked": cjk_len(entity_text),
                    "entity_masked_v2": cjk_len(entity_v2_text),
                    "entity_masked_v3": cjk_len(entity_v3_text),
                    "topic_distorted": clean_cjk_count,
                    "structure_only": clean_cjk_count,
                }
                for view, view_text in views.items():
                    payload = chunk_record(
                        record,
                        chunk_id=chunk_id,
                        split_name=split_name,
                        chunk_index=index,
                        clean_cjk_count=clean_cjk_count,
                        view=view,
                        view_text=view_text,
                        view_cjk_count=view_cjk_counts[view],
                    )
                    handles[view].write(json.dumps(payload, ensure_ascii=False) + "\n")
                    summary["chunks_by_view"][view] += 1
                summary["chunks_by_split"][split_name] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary["chunks_by_view"] = dict(summary["chunks_by_view"])
    summary["chunks_by_split"] = dict(summary["chunks_by_split"])
    return summary


def write_mask_plan(path: Path, mask_plan: dict[str, dict[str, Any]]) -> None:
    payload = {
        "parameters": {
            "ngram_min_count": MASK_NGRAM_MIN_COUNT,
            "ngram_min_n": MASK_NGRAM_MIN_N,
            "ngram_max_n": MASK_NGRAM_MAX_N,
            "max_entity_terms_per_book": MAX_ENTITY_TERMS_PER_BOOK,
            "max_entity_v2_terms_per_book": MAX_ENTITY_V2_TERMS_PER_BOOK,
            "max_topic_terms_per_book": MAX_TOPIC_TERMS_PER_BOOK,
        },
        "books": list(mask_plan.values()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_chunk_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Style Research Chunk Report",
        "",
        "## Summary",
        "",
        f"- Dataset root: `{summary['dataset_root']}`",
        f"- Books chunked: {summary['books_chunked']}",
        f"- Chunk target CJK: {summary['chunk_target_cjk']}",
        f"- Minimum tail CJK before merge: {summary['chunk_min_cjk']}",
        "",
        "## Chunks By View",
        "",
    ]
    for view, count in sorted(summary["chunks_by_view"].items()):
        lines.append(f"- {view}: {count}")
    lines.extend(["", "## Clean Chunks By Split", ""])
    for split_name, count in sorted(summary["chunks_by_split"].items()):
        lines.append(f"- {split_name}: {count}")
    lines.extend(["", "## Output Files", ""])
    for view, output_path in sorted(summary["output_paths"].items()):
        lines.append(f"- {view}: `{output_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recommendation(summary: dict[str, Any]) -> list[str]:
    authors_3 = int(summary["authors_with_3plus_usable_books"])
    authors_5 = int(summary["authors_with_5plus_usable_books"])
    notes = [
        "The dataset is usable for a first exploratory author-identification benchmark.",
        "Use book-level train/dev/test splits; do not split chunks from the same book across splits.",
        "Use 50k CJK as the first usable-book threshold and keep <50k texts out of the primary benchmark.",
    ]
    if authors_3 >= 20 and authors_5 < authors_3:
        notes.append(
            "For stronger results, add more books for existing 3-book comparison authors; this is more valuable than adding many new one-book authors."
        )
    if authors_5 < 15:
        notes.append(
            "Aim for at least 5 usable books per core comparison author so train/dev/test has more than one training book."
        )
    notes.append(
        "Keep one-book and two-book authors as out-of-distribution stress tests, not primary classifier labels."
    )
    notes.append(
        "Hold out several target-author books as proxy_transfer data for style-transfer method selection."
    )
    return notes


def write_markdown(path: Path, summary: dict[str, Any], duplicates: list[dict[str, Any]]) -> None:
    lines = [
        "# Style Research Dataset Cleaning Report",
        "",
        "## Summary",
        "",
        f"- Books in manifest: {summary['book_count']}",
        f"- Authors: {summary['author_count']}",
        f"- Missing TXT files: {summary['txt_missing_count']}",
        f"- Usable books >=50k CJK: {summary['usable_book_count_50k']}",
        f"- Primary books >=120k CJK: {summary['primary_book_count_120k']}",
        f"- Authors with >=3 usable books: {summary['authors_with_3plus_usable_books']}",
        f"- Authors with >=5 usable books: {summary['authors_with_5plus_usable_books']}",
        f"- Exact duplicate cleaned texts: {len(duplicates)}",
        "",
        "## Split Counts",
        "",
    ]
    for name, count in summary["split_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Quality Flags", ""])
    for flag, count in sorted(summary["quality_flag_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {flag}: {count}")
    lines.extend(["", "## Recommendations", ""])
    for note in recommendation(summary):
        lines.append(f"- {note}")
    lines.extend(["", "## Author Coverage", ""])
    lines.append("| Author | Books | Usable >=50k | Primary >=120k | Clean CJK | Main Flags |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for item in summary["author_summary"]:
        flags = ", ".join(f"{key}:{value}" for key, value in sorted(item["flags"].items()))
        lines.append(
            f"| {item['author']} | {item['book_count']} | {item['usable_book_count_50k']} | "
            f"{item['primary_book_count_120k']} | {item['clean_cjk_total']} | {flags} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and prepare cleaned corpus artifacts for author-style research.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/dataset_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("generated/style_research/corpus"))
    parser.add_argument("--target-author", default="非天夜翔")
    parser.add_argument(
        "--stage",
        choices=["clean", "all"],
        default="clean",
        help="Run only corpus cleanup, or cleanup followed by chunk and mask generation.",
    )
    parser.add_argument("--chunk-target-cjk", type=int, default=CHUNK_TARGET_CJK)
    parser.add_argument("--chunk-min-cjk", type=int, default=CHUNK_MIN_CJK)
    parser.add_argument(
        "--include-excluded-chunks",
        action="store_true",
        help="Also write chunks for books excluded from the primary book-level benchmark.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest
    dataset_root = manifest_path.parent
    output_dir = args.output_dir
    text_root = output_dir / "texts"
    output_dir.mkdir(parents=True, exist_ok=True)
    text_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [book_record(row, dataset_root, text_root) for row in manifest]
    stale_cleaned_files_removed = prune_stale_cleaned_texts(text_root, records)
    splits = split_books(records, clean_author(args.target_author))
    summary = summarize(records, splits)
    duplicates = duplicate_report(records)

    (output_dir / "raw_manifest_snapshot.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cleaned_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "duplicate_report.json").write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cleaning_report.json").write_text(
        json.dumps({**summary, "recommendations": recommendation(summary)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "cleaning_report.md", summary, duplicates)
    output = {
        "stage": args.stage,
        "output_dir": str(output_dir),
        "dataset_root": str(dataset_root),
        "book_count": summary["book_count"],
        "author_count": summary["author_count"],
        "usable_book_count_50k": summary["usable_book_count_50k"],
        "authors_with_3plus_usable_books": summary["authors_with_3plus_usable_books"],
        "authors_with_5plus_usable_books": summary["authors_with_5plus_usable_books"],
        "split_counts": summary["split_counts"],
        "duplicate_count": len(duplicates),
        "stale_cleaned_files_removed": stale_cleaned_files_removed,
    }

    if args.stage == "all":
        split_names = split_lookup(splits)
        if args.include_excluded_chunks:
            chunk_records = records
        else:
            chunk_records = [
                record for record in records
                if split_names.get((str(record["author"]), str(record["title"])), "excluded") != "excluded"
            ]
        mask_plan = select_mask_terms(chunk_records)
        chunk_summary = generate_chunk_views(
            chunk_records,
            splits,
            mask_plan,
            dataset_root,
            chunk_target_cjk=args.chunk_target_cjk,
            chunk_min_cjk=args.chunk_min_cjk,
            include_excluded=args.include_excluded_chunks,
        )
        (output_dir / "chunk_report.json").write_text(
            json.dumps(chunk_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_mask_plan(dataset_root / "masked" / "mask_terms.json", mask_plan)
        write_chunk_report(dataset_root / "masked" / "masking_report.md", chunk_summary)
        output["chunks_by_view"] = chunk_summary["chunks_by_view"]
        output["chunk_outputs"] = chunk_summary["output_paths"]

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
