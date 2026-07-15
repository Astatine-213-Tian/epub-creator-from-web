#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import ahocorasick


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
    r"本书由|整理制作|TXT下载|电子书|更多精彩|营养液加更|感谢.*营养液|"
    r"这个段落是图片段落|请访问正确的网站|原版未篡改内容请移至|"
    r"关闭广告拦截功能|退出浏览器阅读模式"
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
PUNCTUATION_NORMALIZATION_VERSION = "canonical_zh_v1"
CROSS_BOOK_DECONTAMINATION_VERSION = "exact_passage_shingles_v1"
MASKING_POLICY_VERSION = "train_fit_global_label_blind_local_v2"

ASCII_ELLIPSIS_RE = re.compile(r"\.{2,}")
UNICODE_ELLIPSIS_RE = re.compile(r"…+")
DASH_RUN_RE = re.compile(r"(?:-{2,}|[—–―─]{2,})")
ASCII_COMMA_RE = re.compile(r"(?<!\d),|,(?!\d)")
ASCII_PERIOD_RE = re.compile(r"(?<!\d)\.|\.(?!\d)")
ASCII_COLON_RE = re.compile(r"(?<!\d):|:(?!\d)")
PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "﹐": "，",
        "﹑": "、",
        "､": "、",
        "﹒": "。",
        "．": "。",
        "｡": "。",
        "!": "！",
        "﹗": "！",
        "?": "？",
        "﹖": "？",
        ";": "；",
        "﹔": "；",
        "﹕": "：",
        "(": "（",
        ")": "）",
        "﹙": "（",
        "﹚": "）",
        "[": "【",
        "]": "】",
        "﹝": "【",
        "﹞": "】",
        "「": "“",
        "」": "”",
        "﹁": "“",
        "﹂": "”",
        "『": "‘",
        "』": "’",
        "〝": "“",
        "〞": "”",
        "–": "—",
        "―": "—",
        "─": "—",
        "~": "～",
        "＂": '"',
    }
)


AUTHOR_ALIASES = {
    "顾雪柔": "非天夜翔",
}

CHUNK_TARGET_CJK = 1_500
CHUNK_MIN_CJK = 800
MASK_NGRAM_MIN_COUNT = 4
MAX_ENTITY_TERMS_PER_BOOK = 240
MAX_ENTITY_V2_TERMS_PER_BOOK = 800
MAX_TOPIC_TERMS_PER_BOOK = 640
MAX_GLOBAL_ENTITY_V2_TERMS = 12_000
MAX_GLOBAL_CONCENTRATION_RESCUE_TERMS = 4_000
MASK_NGRAM_MIN_N = 2
MASK_NGRAM_MAX_N = 5
MASK_RESCUE_MAX_N = 3
MASK_RESCUE_MIN_BOOK_COUNT = 20
MASK_RESCUE_MAX_DOCUMENT_FREQUENCY = 4
MASK_RESCUE_MAX_AUTHOR_FREQUENCY = 2
MASK_RESCUE_MIN_DOMINANT_BOOK_SHARE = 0.80
MASK_RESCUE_HIGH_CONCENTRATION_SHARE = 0.95
MASK_RESCUE_HIGH_CONCENTRATION_MIN_BOOK_COUNT = 100
MASK_RESCUE_MIN_COMPONENT_CONDITIONAL_SHARE = 0.50
MASK_NESTED_EXTENSION_MIN_COVERAGE = 0.95
CROSS_BOOK_DUPLICATE_RULES = ((1, 80), (2, 100), (3, 120))

MASK_TERM_STOP_CHARS = set(
    "的一是在不了有和人这中大为上个我以要他时来用们生到作地于出就分对成会可主"
    "发年动同工也能下过子说产种面而方后多定行学法所民得经之进着等部度家"
    "电力里如水化高自理起小物现实加都两体制机当使点从业本去把性好应开它"
    "合还因由其些然前外天那与关各重新线内正心反你看又么但向道此只没给被"
    "很最才并已让"
)
MASK_RESCUE_BOUNDARY_CHARS = set(
    "的地得了着过说道问答喊叫想看见听向和与在把被给从到为是有将让又也都就才还却而"
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


def resolve_manifest_path(value: str, dataset_root: Path) -> Path:
    dataset_root = dataset_root.resolve()
    path = Path(value)
    if path.is_absolute():
        if path.exists():
            return path
        for anchor in ("datasets", "books"):
            if anchor not in path.parts:
                continue
            marker = len(path.parts) - 1 - path.parts[::-1].index(anchor)
            base = dataset_root if anchor == "datasets" else dataset_root.parent.parent
            candidate = base.joinpath(*path.parts[marker + (1 if anchor == "datasets" else 0):])
            if candidate.exists():
                return candidate
        return path
    candidates = (
        dataset_root / path,
        dataset_root.parent / path,
        dataset_root.parent.parent / path,
        path,
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def normalize_manifest_paths(
    manifest: list[dict[str, Any]], dataset_root: Path
) -> int:
    changed = 0
    for row in manifest:
        for field in ("txt_path", "source_epub"):
            value = str(row.get(field) or "")
            if not value:
                continue
            resolved = resolve_manifest_path(value, dataset_root).resolve()
            portable = Path(os.path.relpath(resolved, dataset_root.resolve())).as_posix()
            if portable != value:
                row[field] = portable
                changed += 1
    return changed


def normalize_balanced_straight_quotes(text: str) -> str:
    normalized: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.count('"') % 2:
            normalized.append(line)
            continue
        opening = True
        chars: list[str] = []
        for char in line:
            if char == '"':
                chars.append("“" if opening else "”")
                opening = not opening
            else:
                chars.append(char)
        normalized.append("".join(chars))
    return "".join(normalized)


def normalize_punctuation(text: str) -> str:
    """Canonicalize typographic variants while preserving punctuation roles."""
    text = ASCII_ELLIPSIS_RE.sub("……", text)
    text = UNICODE_ELLIPSIS_RE.sub("……", text)
    text = DASH_RUN_RE.sub("——", text)
    text = ASCII_COMMA_RE.sub("，", text)
    text = ASCII_PERIOD_RE.sub("。", text)
    text = ASCII_COLON_RE.sub("：", text)
    text = text.translate(PUNCTUATION_TRANSLATION)
    return normalize_balanced_straight_quotes(text)


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
    unnormalized = "\n".join(body_lines).strip()
    normalized = normalize_punctuation(unnormalized)
    cleaned = normalized + "\n" if normalized else ""
    return cleaned, {
        "raw_line_count": len(lines),
        "body_start_line": body_start + 1 if lines else 0,
        "dropped_leading_lines": body_start,
        "punctuation_normalization": PUNCTUATION_NORMALIZATION_VERSION,
        "punctuation_normalized": normalized != unnormalized,
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


def mask_candidate_ngram_counts(
    text: str,
) -> tuple[Counter[str], Counter[str]]:
    standard: Counter[str] = Counter()
    rescue: Counter[str] = Counter()
    for match in CJK_RUN_RE.finditer(text):
        run = match.group(0)
        for size in range(MASK_NGRAM_MIN_N, MASK_NGRAM_MAX_N + 1):
            if len(run) < size:
                continue
            for index in range(0, len(run) - size + 1):
                term = run[index:index + size]
                if is_mask_candidate(term):
                    standard[term] += 1
                elif is_concentration_rescue_candidate(term):
                    rescue[term] += 1
    return standard, rescue


def cjk_ngrams(text: str) -> Counter[str]:
    standard, _ = mask_candidate_ngram_counts(text)
    return standard


def concentration_rescue_ngrams(text: str) -> Counter[str]:
    """Count high-frequency content candidates rejected by the broad stop list."""
    _, rescue = mask_candidate_ngram_counts(text)
    return rescue


def count_selected_terms(
    text: str,
    candidate_terms: set[str],
) -> Counter[str]:
    """Count every occurrence of a bounded candidate vocabulary in one book."""
    counts: Counter[str] = Counter()
    candidate_lengths = sorted({len(term) for term in candidate_terms})
    if not candidate_lengths:
        return counts
    for match in CJK_RUN_RE.finditer(text):
        run = match.group(0)
        for size in candidate_lengths:
            if len(run) < size:
                continue
            for index in range(0, len(run) - size + 1):
                term = run[index:index + size]
                if term in candidate_terms:
                    counts[term] += 1
    return counts


def is_mask_candidate(term: str) -> bool:
    return (
        MASK_NGRAM_MIN_N <= len(term) <= MASK_NGRAM_MAX_N
        and bool(CJK_RUN_RE.fullmatch(term))
        and len(set(term)) > 1
        and not any(char in MASK_TERM_STOP_CHARS for char in term)
    )


def is_concentration_rescue_candidate(term: str) -> bool:
    return (
        MASK_NGRAM_MIN_N <= len(term) <= MASK_RESCUE_MAX_N
        and bool(CJK_RUN_RE.fullmatch(term))
        and len(set(term)) > 1
        and any(char in MASK_TERM_STOP_CHARS for char in term)
        and term[0] not in MASK_RESCUE_BOUNDARY_CHARS
        and term[-1] not in MASK_RESCUE_BOUNDARY_CHARS
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


def canonicalize_nested_mask_terms(
    terms: Iterable[str],
    counts: Mapping[str, int],
    *,
    extension_min_coverage: float = MASK_NESTED_EXTENSION_MIN_COVERAGE,
) -> tuple[list[str], dict[str, int]]:
    """Keep entity cores while dropping low-coverage name-plus-context terms."""
    active = set(terms)
    input_term_count = len(active)
    removed_extensions: set[str] = set()
    removed_subterms: set[str] = set()
    for term in sorted(
        active,
        key=lambda item: (-len(item), -int(counts.get(item, 0)), item),
    ):
        if term not in active or int(counts.get(term, 0)) <= 0:
            continue
        observed_nested = {
            term[index:index + size]
            for size in range(MASK_NGRAM_MIN_N, len(term))
            for index in range(0, len(term) - size + 1)
            if int(counts.get(term[index:index + size], 0)) > 0
        }
        if not observed_nested:
            continue
        term_count = int(counts[term])
        max_subterm_count = max(
            int(counts[candidate]) for candidate in observed_nested
        )
        if term_count / max_subterm_count < extension_min_coverage:
            active.remove(term)
            removed_extensions.add(term)
            continue
        for candidate in observed_nested:
            if (
                candidate in active
                and term_count / int(counts[candidate]) >= extension_min_coverage
            ):
                active.remove(candidate)
                removed_subterms.add(candidate)

    return sorted(active, key=lambda item: (-len(item), item)), {
        "input_term_count": input_term_count,
        "output_term_count": len(active),
        "removed_context_extension_count": len(removed_extensions),
        "removed_near_equivalent_subterm_count": len(removed_subterms),
    }


def record_key(record: dict[str, Any]) -> str:
    return f"{record['author']}::{record['title']}"


def book_id_sha256(book_ids: list[str]) -> str:
    payload = "\n".join(sorted(book_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ranked_local_terms(
    counts: Counter[str],
    title: str,
    *,
    limit: int,
) -> list[str]:
    terms = list(title_mask_terms(title))
    seen = set(terms)
    ranked = sorted(
        counts.items(),
        key=lambda item: (-(item[1] * len(item[0])), -len(item[0]), item[0]),
    )
    for term, _count in ranked:
        if term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= limit:
            break
    canonical_terms, _ = canonicalize_nested_mask_terms(terms, counts)
    return canonical_terms


def select_mask_terms(
    records: list[dict[str, Any]],
    splits: dict[str, Any],
) -> dict[str, Any]:
    split_names = split_lookup(splits)
    fit_records = [
        record
        for record in records
        if split_names.get((str(record["author"]), str(record["title"]))) == "train"
    ]
    fit_keys = {record_key(record) for record in fit_records}
    fit_book_ids = sorted(fit_keys)

    fit_book_counts: dict[str, Counter[str]] = {}
    document_frequency: Counter[str] = Counter()
    term_authors: dict[str, set[str]] = defaultdict(set)
    total_frequency: Counter[str] = Counter()
    rescue_seed_terms: set[str] = set()
    rescue_max_book_frequency: Counter[str] = Counter()
    rescue_dominant_book: dict[str, str] = {}
    for record in fit_records:
        text_path = Path(str(record["clean_txt_path"]))
        if not text_path.exists():
            continue
        text = read_text(text_path)
        standard_counts, raw_rescue_counts = mask_candidate_ngram_counts(text)
        counts = Counter({
            term: count
            for term, count in standard_counts.items()
            if count >= MASK_NGRAM_MIN_COUNT
        })
        rescue_seed_counts = Counter({
            term: count
            for term, count in raw_rescue_counts.items()
            if count >= MASK_RESCUE_MIN_BOOK_COUNT
        })
        key = record_key(record)
        fit_book_counts[key] = counts
        document_frequency.update(counts.keys())
        total_frequency.update(counts)
        for term in counts:
            term_authors[term].add(str(record["author"]))
        rescue_seed_terms.update(rescue_seed_counts)
        for term, count in rescue_seed_counts.items():
            if count > rescue_max_book_frequency[term]:
                rescue_max_book_frequency[term] = count
                rescue_dominant_book[term] = key

    # The support threshold only seeds plausible rescue terms. Their spread and
    # concentration must include every occurrence in every fit book; otherwise
    # 19 occurrences in many books would be silently ignored while 20 in one
    # book could make an ordinary phrase look book-specific.
    rescue_document_frequency: Counter[str] = Counter()
    rescue_term_authors: dict[str, set[str]] = defaultdict(set)
    rescue_total_frequency: Counter[str] = Counter()
    for record in fit_records:
        text_path = Path(str(record["clean_txt_path"]))
        if not text_path.exists():
            continue
        rescue_counts = count_selected_terms(
            read_text(text_path), rescue_seed_terms
        )
        rescue_document_frequency.update(rescue_counts.keys())
        rescue_total_frequency.update(rescue_counts)
        for term in rescue_counts:
            rescue_term_authors[term].add(str(record["author"]))

    book_count = max(len(fit_book_counts), 1)
    entity_max_df = max(3, book_count // 20)
    author_frequency = {term: len(authors) for term, authors in term_authors.items()}
    global_candidates = [
        term
        for term in total_frequency
        if document_frequency[term] <= entity_max_df
        or author_frequency.get(term, 0) <= 1
    ]
    global_entity_terms = sorted(
        global_candidates,
        key=lambda term: (
            -(total_frequency[term] * len(term) / max(author_frequency.get(term, 1), 1)),
            -len(term),
            term,
        ),
    )[:MAX_GLOBAL_ENTITY_V2_TERMS]
    concentration_candidates = []
    for term, total in rescue_total_frequency.items():
        dominant_share = rescue_max_book_frequency[term] / total
        limited_spread = (
            rescue_document_frequency[term] <= MASK_RESCUE_MAX_DOCUMENT_FREQUENCY
            and len(rescue_term_authors[term]) <= MASK_RESCUE_MAX_AUTHOR_FREQUENCY
        )
        collision_tolerant_concentration = (
            dominant_share >= MASK_RESCUE_HIGH_CONCENTRATION_SHARE
            and rescue_max_book_frequency[term]
            >= MASK_RESCUE_HIGH_CONCENTRATION_MIN_BOOK_COUNT
        )
        if (
            dominant_share >= MASK_RESCUE_MIN_DOMINANT_BOOK_SHARE
            and (limited_spread or collision_tolerant_concentration)
        ):
            concentration_candidates.append(term)

    fit_records_by_key = {record_key(record): record for record in fit_records}
    dominant_book_char_counts: dict[str, Counter[str]] = {}
    component_conditional_rejections: list[str] = []
    rescue_candidates: list[str] = []
    for term in concentration_candidates:
        if len(term) != 2:
            rescue_candidates.append(term)
            continue
        dominant_key = rescue_dominant_book[term]
        if dominant_key not in dominant_book_char_counts:
            dominant_record = fit_records_by_key[dominant_key]
            dominant_text = read_text(Path(str(dominant_record["clean_txt_path"])))
            dominant_book_char_counts[dominant_key] = Counter(
                CJK_RE.findall(dominant_text)
            )
        char_counts = dominant_book_char_counts[dominant_key]
        dominant_count = rescue_max_book_frequency[term]
        component_conditional_share = max(
            dominant_count / char_counts[char]
            for char in term
            if char_counts[char]
        )
        if (
            component_conditional_share
            >= MASK_RESCUE_MIN_COMPONENT_CONDITIONAL_SHARE
        ):
            rescue_candidates.append(term)
        else:
            component_conditional_rejections.append(term)
    concentration_rescue_terms = sorted(
        rescue_candidates,
        key=lambda term: (
            -(rescue_total_frequency[term] * len(term)),
            -len(term),
            term,
        ),
    )[:MAX_GLOBAL_CONCENTRATION_RESCUE_TERMS]
    raw_global_entity_terms = set(global_entity_terms) | set(
        concentration_rescue_terms
    )
    global_count_terms = set(raw_global_entity_terms)
    for term in raw_global_entity_terms:
        global_count_terms.update(
            term[index:index + size]
            for size in range(MASK_NGRAM_MIN_N, len(term))
            for index in range(0, len(term) - size + 1)
        )
    global_term_counts: Counter[str] = Counter()
    for record in fit_records:
        text_path = Path(str(record["clean_txt_path"]))
        if text_path.exists():
            global_term_counts.update(
                count_selected_terms(read_text(text_path), global_count_terms)
            )
    global_entity_terms, global_canonicalization = canonicalize_nested_mask_terms(
        raw_global_entity_terms,
        global_term_counts,
    )
    global_entity_term_set = set(global_entity_terms)
    concentration_rescue_terms = [
        term for term in concentration_rescue_terms if term in global_entity_term_set
    ]

    selected: list[dict[str, Any]] = []
    for record in records:
        key = record_key(record)
        text_path = Path(str(record["clean_txt_path"]))
        counts = fit_book_counts.get(key)
        if counts is None:
            counts = Counter({
                term: count
                for term, count in cjk_ngrams(read_text(text_path)).items()
                if count >= MASK_NGRAM_MIN_COUNT
            })
        title = str(record["title"])
        selected.append({
            "author": record["author"],
            "title": title,
            "split": split_names.get((str(record["author"]), title), "excluded"),
            "selection_uses_author_label": False,
            "candidate_term_count": len(counts),
            "entity_terms": ranked_local_terms(
                counts, title, limit=MAX_ENTITY_TERMS_PER_BOOK
            ),
            "entity_terms_v2": ranked_local_terms(
                counts, title, limit=MAX_ENTITY_V2_TERMS_PER_BOOK
            ),
            "topic_terms": ranked_local_terms(
                counts, title, limit=MAX_TOPIC_TERMS_PER_BOOK
            ),
        })
    return {
        "schema_version": 3,
        "masking_policy": MASKING_POLICY_VERSION,
        "provenance": {
            "fit_split": "train",
            "fit_book_count": len(fit_book_ids),
            "fit_book_ids": fit_book_ids,
            "fit_book_ids_sha256": book_id_sha256(fit_book_ids),
            "global_term_selection_uses_fit_labels": True,
            "concentration_rescue_uses_fit_labels": True,
            "concentration_rescue_uses_held_out_statistics": False,
            "concentration_rescue_counts_all_fit_occurrences": True,
            "nested_canonicalization_counts_all_fit_occurrences": True,
            "nested_term_canonicalization": global_canonicalization,
            "rescue_component_conditional_rejection_count": len(
                component_conditional_rejections
            ),
            "transform_uses_author_label": False,
            "held_out_corpus_statistics_used_for_global_terms": False,
            "local_term_selection": "same_book_text_and_title_without_author_label",
        },
        "parameters": {
            "ngram_min_count": MASK_NGRAM_MIN_COUNT,
            "ngram_min_n": MASK_NGRAM_MIN_N,
            "ngram_max_n": MASK_NGRAM_MAX_N,
            "rescue_max_n": MASK_RESCUE_MAX_N,
            "max_entity_terms_per_book": MAX_ENTITY_TERMS_PER_BOOK,
            "max_entity_v2_terms_per_book": MAX_ENTITY_V2_TERMS_PER_BOOK,
            "max_topic_terms_per_book": MAX_TOPIC_TERMS_PER_BOOK,
            "max_global_entity_v2_terms": MAX_GLOBAL_ENTITY_V2_TERMS,
            "max_global_concentration_rescue_terms": (
                MAX_GLOBAL_CONCENTRATION_RESCUE_TERMS
            ),
            "rescue_min_book_count": MASK_RESCUE_MIN_BOOK_COUNT,
            "rescue_max_document_frequency": MASK_RESCUE_MAX_DOCUMENT_FREQUENCY,
            "rescue_max_author_frequency": MASK_RESCUE_MAX_AUTHOR_FREQUENCY,
            "rescue_min_dominant_book_share": (
                MASK_RESCUE_MIN_DOMINANT_BOOK_SHARE
            ),
            "rescue_high_concentration_share": (
                MASK_RESCUE_HIGH_CONCENTRATION_SHARE
            ),
            "rescue_high_concentration_min_book_count": (
                MASK_RESCUE_HIGH_CONCENTRATION_MIN_BOOK_COUNT
            ),
            "rescue_min_component_conditional_share": (
                MASK_RESCUE_MIN_COMPONENT_CONDITIONAL_SHARE
            ),
            "nested_extension_min_coverage": (
                MASK_NESTED_EXTENSION_MIN_COVERAGE
            ),
        },
        "global_terms": {
            "entity_terms_v2": global_entity_terms,
            "concentration_rescue_terms": sorted(
                concentration_rescue_terms, key=lambda item: (-len(item), item)
            ),
        },
        "books": selected,
    }


def normalize_non_cjk_content(text: str) -> str:
    text = LATIN_RE.sub("<LATIN>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    return re.sub(r"<+(CONTENT|NUM|LATIN)>+", r"<\1>", text)


class TermMatcher:
    """Exact leftmost-longest multi-term matcher backed by Aho-Corasick."""

    def __init__(self, terms: list[str]):
        automaton = ahocorasick.Automaton()
        for term in sorted(set(terms), key=lambda item: (-len(item), item)):
            if term:
                automaton.add_word(term, term)
        automaton.make_automaton()
        self._automaton = automaton

    def substitute(
        self,
        text: str,
        *,
        placeholder: str,
        preserve_length: bool,
    ) -> str:
        output: list[str] = []
        cursor = 0
        for end_index, term in self._automaton.iter_long(text):
            start_index = end_index - len(term) + 1
            output.append(text[cursor:start_index])
            output.append(
                placeholder * cjk_len(term) if preserve_length else placeholder
            )
            cursor = end_index + 1
        output.append(text[cursor:])
        return "".join(output)


def compile_term_matcher(terms: list[str]) -> TermMatcher | None:
    if not terms:
        return None
    return TermMatcher(terms)


def mask_terms(
    text: str,
    matcher: TermMatcher | None,
    *,
    placeholder: str,
    preserve_length: bool = False,
) -> str:
    if matcher is None:
        return text
    return matcher.substitute(
        text,
        placeholder=placeholder,
        preserve_length=preserve_length,
    )


def entity_masked_text(text: str, matcher: TermMatcher | None) -> str:
    text = normalize_non_cjk_content(text)
    return mask_terms(text, matcher, placeholder="<CONTENT>")


def entity_masked_v2_text(text: str, matcher: TermMatcher | None) -> str:
    text = normalize_non_cjk_content(text)
    return mask_terms(text, matcher, placeholder="<TERM>")


def entity_masked_v3_text(text: str, matcher: TermMatcher | None) -> str:
    text = normalize_non_cjk_content(text)
    return mask_terms(text, matcher, placeholder="某", preserve_length=True)


def topic_distorted_text(text: str, matcher: TermMatcher | None) -> str:
    text = normalize_non_cjk_content(text)
    text = mask_terms(text, matcher, placeholder="文", preserve_length=True)
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
    txt_path = resolve_manifest_path(str(row.get("txt_path") or ""), dataset_root)
    recorded_txt_path = Path(
        os.path.relpath(txt_path.resolve(), Path.cwd().resolve())
    ).as_posix()
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
        "txt_path": recorded_txt_path,
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


def passage_fingerprint(lines: list[str]) -> str:
    normalized = "\n".join(re.sub(r"\s+", "", line) for line in lines)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cross_book_duplicate_passages(
    books: dict[str, list[str]],
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    marked_lines: dict[str, set[int]] = defaultdict(set)
    findings: list[dict[str, Any]] = []
    for window_size, minimum_cjk in CROSS_BOOK_DUPLICATE_RULES:
        occurrences: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for key, lines in books.items():
            for index in range(0, len(lines) - window_size + 1):
                window = lines[index:index + window_size]
                window_cjk = sum(cjk_len(line) for line in window)
                if window_cjk < minimum_cjk:
                    continue
                occurrences[passage_fingerprint(window)].append(
                    (key, index, window_cjk)
                )
        for fingerprint, matches in occurrences.items():
            matched_books = {key for key, _index, _cjk in matches}
            if len(matched_books) < 2:
                continue
            for key, index, _window_cjk in matches:
                marked_lines[key].update(range(index, index + window_size))
            findings.append(
                {
                    "sha256": fingerprint,
                    "window_lines": window_size,
                    "minimum_cjk": minimum_cjk,
                    "occurrence_count": len(matches),
                    "book_count": len(matched_books),
                    "occurrences": [
                        {
                            "book_id": key,
                            "paragraph_index": index + 1,
                            "cjk_count": window_cjk,
                        }
                        for key, index, window_cjk in matches
                    ],
                }
            )
    findings.sort(
        key=lambda item: (
            -int(item["book_count"]),
            -int(item["window_lines"]),
            str(item["sha256"]),
        )
    )
    return marked_lines, findings


def decontaminate_cross_book_passages(records: list[dict[str, Any]]) -> dict[str, Any]:
    books: dict[str, list[str]] = {}
    records_by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        if not record.get("exists") or not record.get("clean_txt_path"):
            continue
        path = Path(str(record["clean_txt_path"]))
        if not path.exists():
            continue
        key = record_key(record)
        books[key] = [line for line in normalize_lines(read_text(path)) if line]
        records_by_key[key] = record

    marked_lines, findings = cross_book_duplicate_passages(books)
    removed_line_count = 0
    removed_cjk_count = 0
    affected_books = 0
    for key, lines in books.items():
        record = records_by_key[key]
        removed = marked_lines.get(key, set())
        removed_cjk = sum(cjk_len(lines[index]) for index in removed)
        remaining = [line for index, line in enumerate(lines) if index not in removed]
        cleaned = "\n".join(remaining).strip()
        cleaned = cleaned + "\n" if cleaned else ""
        path = Path(str(record["clean_txt_path"]))
        path.write_text(cleaned, encoding="utf-8")

        record["cross_book_decontamination"] = CROSS_BOOK_DECONTAMINATION_VERSION
        record["cross_book_duplicate_lines_removed"] = len(removed)
        record["cross_book_duplicate_cjk_removed"] = removed_cjk
        record["clean_cjk_count"] = cjk_len(cleaned)
        record["clean_sha256"] = (
            hashlib.sha256(cleaned.encode("utf-8")).hexdigest() if cleaned else ""
        )
        raw = read_text(Path(str(record["txt_path"])))
        record["quality_flags"] = text_flags(raw, cleaned)
        if record.get("author_header_conflicts"):
            record["quality_flags"].append("manifest_author_header_conflict")
        if removed:
            record["quality_flags"].append("cross_book_duplicate_passages_removed")
            affected_books += 1
        removed_line_count += len(removed)
        removed_cjk_count += removed_cjk

    remaining_books = {
        key: [line for line in normalize_lines(read_text(Path(str(record["clean_txt_path"])))) if line]
        for key, record in records_by_key.items()
    }
    _remaining_marks, remaining_findings = cross_book_duplicate_passages(remaining_books)
    if remaining_findings:
        raise ValueError(
            "Cross-book passage decontamination left repeated fingerprints; "
            "the corpus cannot be split safely"
        )
    return {
        "schema_version": 1,
        "policy": CROSS_BOOK_DECONTAMINATION_VERSION,
        "rules": [
            {"window_lines": window_size, "minimum_cjk": minimum_cjk}
            for window_size, minimum_cjk in CROSS_BOOK_DUPLICATE_RULES
        ],
        "affected_books": affected_books,
        "removed_line_count": removed_line_count,
        "removed_cjk_count": removed_cjk_count,
        "detected_fingerprint_count": len(findings),
        "remaining_fingerprint_count": 0,
        "fingerprints": findings,
    }


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
        "punctuation_normalization": PUNCTUATION_NORMALIZATION_VERSION,
        "books_with_punctuation_normalization": sum(
            1 for item in records if item.get("punctuation_normalized")
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
        "punctuation_normalization": record.get("punctuation_normalization") or "",
        "cross_book_decontamination": record.get("cross_book_decontamination") or "",
        "masking_policy": MASKING_POLICY_VERSION,
        "text": view_text,
    }


def generate_chunk_views(
    records: list[dict[str, Any]],
    splits: dict[str, Any],
    mask_plan: dict[str, Any],
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
        "train_global_masked": masked_dir / "chunks.train_global_masked.jsonl",
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
    plan_by_book = {
        record_key(row): row
        for row in mask_plan.get("books", [])
        if isinstance(row, dict)
    }
    global_entity_v2_matcher = compile_term_matcher(
        list(mask_plan.get("global_terms", {}).get("entity_terms_v2") or [])
    )
    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "chunk_target_cjk": chunk_target_cjk,
        "chunk_min_cjk": chunk_min_cjk,
        "include_excluded": include_excluded,
        "masking_policy": mask_plan.get("masking_policy", ""),
        "mask_fit_book_count": int(mask_plan.get("provenance", {}).get("fit_book_count", 0)),
        "mask_fit_book_ids_sha256": str(
            mask_plan.get("provenance", {}).get("fit_book_ids_sha256", "")
        ),
        "cross_book_decontamination": CROSS_BOOK_DECONTAMINATION_VERSION,
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
            plan = plan_by_book.get(record_key(record), {})
            entity_matcher = compile_term_matcher(list(plan.get("entity_terms") or []))
            entity_v2_matcher = compile_term_matcher(list(plan.get("entity_terms_v2") or []))
            topic_matcher = compile_term_matcher(list(plan.get("topic_terms") or []))

            for index, clean_chunk in enumerate(clean_chunks, start=1):
                chunk_id = f"{safe_id(str(record['author']))}__{safe_id(str(record['title']))}__{index:04d}"
                clean_cjk_count = cjk_len(clean_chunk)
                entity_text = entity_masked_text(clean_chunk, entity_matcher)
                train_global_masked_text = entity_masked_v3_text(
                    clean_chunk, global_entity_v2_matcher
                )
                entity_v2_text = entity_masked_v2_text(clean_chunk, global_entity_v2_matcher)
                entity_v2_text = mask_terms(
                    entity_v2_text, entity_v2_matcher, placeholder="<TERM>"
                )
                entity_v3_text = entity_masked_v3_text(clean_chunk, global_entity_v2_matcher)
                entity_v3_text = mask_terms(
                    entity_v3_text,
                    entity_v2_matcher,
                    placeholder="某",
                    preserve_length=True,
                )
                views = {
                    "clean": clean_chunk,
                    "train_global_masked": train_global_masked_text,
                    "entity_masked": entity_text,
                    "entity_masked_v2": entity_v2_text,
                    "entity_masked_v3": entity_v3_text,
                    "topic_distorted": topic_distorted_text(clean_chunk, topic_matcher),
                    "structure_only": structure_only_text(clean_chunk),
                }
                view_cjk_counts = {
                    "clean": clean_cjk_count,
                    "train_global_masked": cjk_len(train_global_masked_text),
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


def write_mask_plan(path: Path, mask_plan: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(mask_plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
        f"- Punctuation normalization: `{PUNCTUATION_NORMALIZATION_VERSION}`",
        f"- Cross-book decontamination: `{summary['cross_book_decontamination']}`",
        f"- Masking policy: `{summary['masking_policy']}`",
        f"- Mask vocabulary fit books: {summary['mask_fit_book_count']}",
        f"- Mask fit-book hash: `{summary['mask_fit_book_ids_sha256']}`",
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
        f"- Punctuation normalization: `{summary['punctuation_normalization']}`",
        f"- Books changed by punctuation normalization: {summary['books_with_punctuation_normalization']}",
        f"- Authors with >=3 usable books: {summary['authors_with_3plus_usable_books']}",
        f"- Authors with >=5 usable books: {summary['authors_with_5plus_usable_books']}",
        f"- Exact duplicate cleaned texts: {len(duplicates)}",
        f"- Cross-book duplicate fingerprints removed: "
        f"{summary['cross_book_decontamination']['detected_fingerprint_count']}",
        f"- Cross-book duplicate lines removed: "
        f"{summary['cross_book_decontamination']['removed_line_count']}",
        f"- Cross-book duplicate CJK removed: "
        f"{summary['cross_book_decontamination']['removed_cjk_count']}",
        f"- Remaining checked cross-book fingerprints: "
        f"{summary['cross_book_decontamination']['remaining_fingerprint_count']}",
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
        choices=["paths", "clean", "mask", "chunks", "all"],
        default="clean",
        help=(
            "Normalize manifest paths, run corpus cleanup, fit the train-only mask "
            "plan, rebuild chunk views from current artifacts, or run all stages."
        ),
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

    if args.stage == "paths":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changed = normalize_manifest_paths(manifest, dataset_root)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / "raw_manifest_snapshot.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"stage": "paths", "manifest": str(manifest_path), "paths_changed": changed},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.stage in {"mask", "chunks"}:
        records = json.loads(
            (output_dir / "cleaned_manifest.json").read_text(encoding="utf-8")
        )
        splits = json.loads((output_dir / "splits.json").read_text(encoding="utf-8"))
        if {
            record.get("cross_book_decontamination")
            for record in records
            if record.get("exists")
        } != {CROSS_BOOK_DECONTAMINATION_VERSION}:
            raise ValueError("Cannot rebuild masking artifacts from stale cleaned texts")
        if args.stage == "mask":
            split_names = split_lookup(splits)
            if args.include_excluded_chunks:
                mask_records = records
            else:
                mask_records = [
                    record
                    for record in records
                    if split_names.get(
                        (str(record["author"]), str(record["title"])),
                        "excluded",
                    )
                    != "excluded"
                ]
            mask_plan = select_mask_terms(mask_records, splits)
            mask_plan_path = dataset_root / "masked/mask_terms.json"
            write_mask_plan(mask_plan_path, mask_plan)
            print(
                json.dumps(
                    {
                        "stage": "mask",
                        "mask_plan": str(mask_plan_path),
                        "masking_policy": mask_plan["masking_policy"],
                        "fit_book_count": mask_plan["provenance"]["fit_book_count"],
                        "global_term_count": len(
                            mask_plan["global_terms"]["entity_terms_v2"]
                        ),
                        "rescue_term_count": len(
                            mask_plan["global_terms"][
                                "concentration_rescue_terms"
                            ]
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        mask_plan_path = dataset_root / "masked/mask_terms.json"
        mask_plan = json.loads(mask_plan_path.read_text(encoding="utf-8"))
        if mask_plan.get("masking_policy") != MASKING_POLICY_VERSION:
            raise ValueError("Cannot rebuild chunks from a stale mask plan")
        chunk_summary = generate_chunk_views(
            records,
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
        write_chunk_report(dataset_root / "masked/masking_report.md", chunk_summary)
        print(
            json.dumps(
                {
                    "stage": "chunks",
                    "output_dir": str(output_dir),
                    "dataset_root": str(dataset_root),
                    "chunks_by_view": chunk_summary["chunks_by_view"],
                    "chunk_outputs": chunk_summary["output_paths"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized_manifest_paths = normalize_manifest_paths(manifest, dataset_root)
    if normalized_manifest_paths:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    records = [book_record(row, dataset_root, text_root) for row in manifest]
    stale_cleaned_files_removed = prune_stale_cleaned_texts(text_root, records)
    passage_decontamination = decontaminate_cross_book_passages(records)
    splits = split_books(records, clean_author(args.target_author))
    summary = summarize(records, splits)
    summary["cross_book_decontamination"] = passage_decontamination
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
    (output_dir / "cross_book_passage_report.json").write_text(
        json.dumps(passage_decontamination, ensure_ascii=False, indent=2) + "\n",
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
        "cross_book_duplicate_fingerprints_removed": passage_decontamination[
            "detected_fingerprint_count"
        ],
        "cross_book_duplicate_lines_removed": passage_decontamination[
            "removed_line_count"
        ],
        "cross_book_duplicate_cjk_removed": passage_decontamination[
            "removed_cjk_count"
        ],
        "stale_cleaned_files_removed": stale_cleaned_files_removed,
        "normalized_manifest_paths": normalized_manifest_paths,
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
        mask_plan = select_mask_terms(chunk_records, splits)
        write_mask_plan(dataset_root / "masked" / "mask_terms.json", mask_plan)
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
        write_chunk_report(dataset_root / "masked" / "masking_report.md", chunk_summary)
        output["chunks_by_view"] = chunk_summary["chunks_by_view"]
        output["chunk_outputs"] = chunk_summary["output_paths"]

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
