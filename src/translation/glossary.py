from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


class GlossaryTermMap(dict[str, str]):
    def __init__(self) -> None:
        super().__init__()
        self.lookup_to_source: dict[str, str] = {}
        self.conflicts: list[dict[str, str]] = []


def normalize_lookup_key(source: str) -> str:
    text = unicodedata.normalize("NFKC", source)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    text = _ARTICLE_RE.sub("", text, count=1)
    return text.casefold()


def _match_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_RE.sub(" ", text).casefold()


def _contains_lookup_key(haystack: str, lookup_key: str) -> bool:
    if not lookup_key:
        return False

    start = 0
    while (position := haystack.find(lookup_key, start)) >= 0:
        end = position + len(lookup_key)
        left_boundary = (
            not lookup_key[0].isalnum()
            or position == 0
            or not haystack[position - 1].isalnum()
        )
        right_boundary = (
            not lookup_key[-1].isalnum()
            or end == len(haystack)
            or not haystack[end].isalnum()
        )
        if left_boundary and right_boundary:
            return True
        start = position + 1
    return False


def _term_zh(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("zh") or "").strip()


def _entry_aliases(entry: dict[str, Any]) -> list[str]:
    aliases = entry.get("aliases")
    if not isinstance(aliases, list):
        return []
    return [str(alias).strip() for alias in aliases if str(alias).strip()]


def _register_lookup(
    terms: GlossaryTermMap,
    *,
    source: str,
    target: str,
    canonical_source: str | None = None,
) -> str | None:
    source = source.strip()
    lookup_key = normalize_lookup_key(source)
    if not source or not lookup_key:
        return canonical_source

    existing_source = terms.lookup_to_source.get(lookup_key)
    if existing_source is None:
        if canonical_source is None:
            terms[source] = target
            canonical_source = source
        terms.lookup_to_source[lookup_key] = canonical_source
        return canonical_source

    existing_target = terms.get(existing_source, "")
    if existing_target != target:
        terms.conflicts.append(
            {
                "normalized_key": lookup_key,
                "source": source,
                "zh": target,
                "existing_source": existing_source,
                "existing_zh": existing_target,
            }
        )
    return existing_source


def load_glossary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"terms": {}, "style_notes": []}
    return json.loads(path.read_text(encoding="utf-8"))


def glossary_terms(glossary: dict[str, Any]) -> dict[str, str]:
    source_terms = glossary.get("terms", {})
    terms = GlossaryTermMap()
    if not isinstance(source_terms, dict):
        return terms

    for key, value in source_terms.items():
        source = str(key).strip()
        target = _term_zh(value)
        if not source or not target:
            continue
        canonical_source = _register_lookup(terms, source=source, target=target)
        if canonical_source is None or not isinstance(value, dict):
            continue
        for alias in _entry_aliases(value):
            _register_lookup(
                terms,
                source=alias,
                target=target,
                canonical_source=canonical_source,
            )
    return terms


def matched_terms(text: str, terms: dict[str, str]) -> dict[str, str]:
    haystack = _match_text(text)
    if isinstance(terms, GlossaryTermMap):
        matches: dict[str, str] = {}
        for lookup_key, source in terms.lookup_to_source.items():
            if _contains_lookup_key(haystack, lookup_key) and source in terms:
                matches[source] = terms[source]
        return matches

    return {
        source: target
        for source, target in terms.items()
        if (lookup_key := normalize_lookup_key(source))
        and _contains_lookup_key(haystack, lookup_key)
    }
