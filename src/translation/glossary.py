from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.crawl.snapshot import write_json


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


def _alias_identity(source: str) -> str:
    text = unicodedata.normalize("NFKC", source)
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _term_zh(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("zh") or "").strip()


def _entry_aliases(entry: dict[str, Any]) -> list[str]:
    aliases = entry.get("aliases")
    if not isinstance(aliases, list):
        return []
    return [str(alias).strip() for alias in aliases if str(alias).strip()]


def _add_alias(entry: dict[str, Any], source: str, alias: str) -> bool:
    alias = alias.strip()
    if not alias:
        return False
    identities = {_alias_identity(source)}
    existing_aliases = _entry_aliases(entry)
    identities.update(_alias_identity(item) for item in existing_aliases)
    if _alias_identity(alias) in identities:
        return False
    aliases = existing_aliases
    aliases.append(alias)
    entry["aliases"] = aliases
    return True


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


def _candidate_conflict(
    *,
    source: str,
    zh: str,
    existing_source: str,
    existing_zh: str,
    lookup_key: str,
    candidate: dict[str, Any],
) -> dict[str, str]:
    return {
        "normalized_key": lookup_key,
        "source": source,
        "zh": zh,
        "existing_source": existing_source,
        "existing_zh": existing_zh,
        "reason": str(candidate.get("reason") or "").strip(),
        "confidence": str(candidate.get("confidence") or "").strip(),
        "source_chunk": str(candidate.get("chunk_id") or "").strip(),
    }


def _existing_lookup_index(terms: dict[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for source, entry in terms.items():
        source_text = str(source).strip()
        zh = _term_zh(entry)
        if not source_text or not zh:
            continue
        lookup_key = normalize_lookup_key(source_text)
        if not lookup_key:
            continue
        index.setdefault(lookup_key, source_text)
        if isinstance(entry, dict):
            for alias in _entry_aliases(entry):
                alias_key = normalize_lookup_key(alias)
                if alias_key:
                    index.setdefault(alias_key, source_text)
    return index


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
            if lookup_key and lookup_key in haystack and source in terms:
                matches[source] = terms[source]
        return matches

    return {
        source: target
        for source, target in terms.items()
        if (lookup_key := normalize_lookup_key(source)) and lookup_key in haystack
    }


def merge_glossary_candidates(
    *,
    glossary_path: Path,
    candidates_path: Path,
) -> dict[str, int]:
    glossary = load_glossary(glossary_path)
    glossary.setdefault("terms", {})
    terms = glossary["terms"]
    if not isinstance(terms, dict):
        raise ValueError(f"glossary terms must be an object: {glossary_path}")

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    lookup_index = _existing_lookup_index(terms)
    added = 0
    kept = 0
    skipped = 0
    aliases_added = 0
    conflicts: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            skipped += 1
            continue
        source = str(candidate.get("source") or "").strip()
        zh = str(candidate.get("zh") or "").strip()
        if not source or not zh:
            skipped += 1
            continue

        lookup_key = normalize_lookup_key(source)
        if not lookup_key:
            skipped += 1
            continue
        existing_source = lookup_index.get(lookup_key)
        existing = terms.get(existing_source) if existing_source else None
        existing_zh = _term_zh(existing)
        if existing_source and existing_zh:
            if existing_zh != zh:
                conflicts.append(
                    _candidate_conflict(
                        source=source,
                        zh=zh,
                        existing_source=existing_source,
                        existing_zh=existing_zh,
                        lookup_key=lookup_key,
                        candidate=candidate,
                    )
                )
                continue
            if isinstance(existing, dict) and _add_alias(existing, existing_source, source):
                aliases_added += 1
            kept += 1
            continue

        terms[source] = {
            "zh": zh,
            "note": str(candidate.get("reason") or "translation run candidate").strip(),
            "confidence": str(candidate.get("confidence") or "").strip(),
            "source_chunk": str(candidate.get("chunk_id") or "").strip(),
        }
        lookup_index[lookup_key] = source
        added += 1

    if conflicts:
        write_json(candidates_path.with_name("glossary_conflicts.json"), {"conflicts": conflicts})
    glossary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(glossary_path, glossary)
    return {
        "added": added,
        "kept": kept,
        "skipped": skipped,
        "aliases_added": aliases_added,
        "conflicts": len(conflicts),
    }
