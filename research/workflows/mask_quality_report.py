#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CHINESE_PUNCT_RE = re.compile(r"[，。！？；：、“”‘’（）《》【】…—]")
PLACEHOLDER_NAME_RE = r"(?:CONTENT|TERM|NAME|PLACE|ORG|NUM|LATIN)"
MALFORMED_PLACEHOLDER_RE = re.compile(
    rf"<<{PLACEHOLDER_NAME_RE}>>|<{PLACEHOLDER_NAME_RE}>>|<<{PLACEHOLDER_NAME_RE}>"
)
RESIDUE_RE = re.compile(
    r"作者有话要说|作者的话|晋江文学城|jjwxc|请收藏|霸王票|营养液加更|"
    r"最新网址|返回目录|手机阅读|https?://|www\.",
    re.I,
)
PLACEHOLDERS = ("<CONTENT>", "<TERM>", "<NAME>", "<PLACE>", "<ORG>", "<NUM>", "<LATIN>")
VIEWS = (
    "clean",
    "train_global_masked",
    "entity_masked",
    "entity_masked_v2",
    "entity_masked_v3",
    "topic_distorted",
    "structure_only",
)


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: str
    author: str
    title: str
    split: str
    chunk_index: int
    clean_cjk_count: int
    view_cjk_count: int


def cjk_len(text: str) -> int:
    return len(CJK_RE.findall(text))


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def per_1k(count: int, denominator: int) -> float:
    return (count / denominator * 1000) if denominator else 0.0


def truncate_text(text: str, limit: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def jsonl_records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc


def chunk_paths(dataset_root: Path) -> dict[str, Path]:
    return {
        "clean": dataset_root / "unmasked" / "chunks.clean.jsonl",
        "train_global_masked": dataset_root / "masked" / "chunks.train_global_masked.jsonl",
        "entity_masked": dataset_root / "masked" / "chunks.entity_masked.jsonl",
        "entity_masked_v2": dataset_root / "masked" / "chunks.entity_masked_v2.jsonl",
        "entity_masked_v3": dataset_root / "masked" / "chunks.entity_masked_v3.jsonl",
        "topic_distorted": dataset_root / "masked" / "chunks.topic_distorted.jsonl",
        "structure_only": dataset_root / "masked" / "chunks.structure_only.jsonl",
    }


def load_clean_meta(path: Path) -> list[ChunkMeta]:
    records: list[ChunkMeta] = []
    for item in jsonl_records(path):
        records.append(
            ChunkMeta(
                chunk_id=str(item["chunk_id"]),
                author=str(item["author"]),
                title=str(item["title"]),
                split=str(item["split"]),
                chunk_index=int(item["chunk_index"]),
                clean_cjk_count=int(item["chunk_clean_cjk_count"]),
                view_cjk_count=int(item["chunk_view_cjk_count"]),
            )
        )
    return records


def view_stats(view: str, path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "path": str(path),
        "rows": 0,
        "authors": Counter(),
        "books": set(),
        "splits": Counter(),
        "clean_cjk": 0,
        "view_cjk": 0,
        "paragraphs": 0,
        "punctuation": 0,
        "residue_hit_chunks": 0,
        "malformed_placeholder_chunks": 0,
        "placeholder_counts": Counter(),
        "generic_wen": 0,
        "generic_mou": 0,
    }
    for item in jsonl_records(path):
        text = str(item["text"])
        stats["rows"] += 1
        stats["authors"][str(item["author"])] += 1
        stats["books"].add((str(item["author"]), str(item["title"])))
        stats["splits"][str(item["split"])] += 1
        stats["clean_cjk"] += int(item["chunk_clean_cjk_count"])
        stats["view_cjk"] += int(item["chunk_view_cjk_count"])
        stats["paragraphs"] += max(1, text.count("\n") + 1)
        stats["punctuation"] += len(CHINESE_PUNCT_RE.findall(text))
        if RESIDUE_RE.search(text):
            stats["residue_hit_chunks"] += 1
        if MALFORMED_PLACEHOLDER_RE.search(text):
            stats["malformed_placeholder_chunks"] += 1
        for placeholder in PLACEHOLDERS:
            stats["placeholder_counts"][placeholder] += text.count(placeholder)
        if view in {"topic_distorted", "structure_only"}:
            stats["generic_wen"] += text.count("文")
        if view in {"train_global_masked", "entity_masked_v3"}:
            stats["generic_mou"] += text.count("某")
    stats["author_count"] = len(stats["authors"])
    stats["book_count"] = len(stats["books"])
    stats["authors"] = dict(stats["authors"])
    stats["books"] = len(stats["books"])
    stats["splits"] = dict(stats["splits"])
    stats["placeholder_counts"] = dict(stats["placeholder_counts"])
    return stats


def stable_pick(records: list[ChunkMeta], seed: str) -> ChunkMeta | None:
    if not records:
        return None
    return min(
        records,
        key=lambda item: hashlib.sha256(f"{seed}:{item.chunk_id}".encode("utf-8")).hexdigest(),
    )


def select_samples(records: list[ChunkMeta], *, target_author: str, sample_count: int) -> list[ChunkMeta]:
    by_author_split: dict[tuple[str, str], list[ChunkMeta]] = defaultdict(list)
    by_author: dict[str, list[ChunkMeta]] = defaultdict(list)
    for item in records:
        by_author_split[(item.author, item.split)].append(item)
        by_author[item.author].append(item)

    samples: list[ChunkMeta] = []
    seen: set[str] = set()

    def add(candidate: ChunkMeta | None) -> None:
        if candidate and candidate.chunk_id not in seen:
            samples.append(candidate)
            seen.add(candidate.chunk_id)

    for split_name in ("proxy_transfer", "train", "dev", "test"):
        add(stable_pick(by_author_split.get((target_author, split_name), []), f"target:{split_name}"))

    comparison_authors = [
        author
        for author, _count in sorted(
            ((author, len(items)) for author, items in by_author.items() if author != target_author),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    split_cycle = ("test", "dev", "train")
    for index, author in enumerate(comparison_authors):
        if len(samples) >= sample_count:
            break
        split_name = split_cycle[index % len(split_cycle)]
        candidate = stable_pick(by_author_split.get((author, split_name), []), f"comparison:{author}:{split_name}")
        add(candidate or stable_pick(by_author.get(author, []), f"comparison:{author}:any"))

    return samples[:sample_count]


def load_sample_views(paths: dict[str, Path], sample_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    samples: dict[str, dict[str, dict[str, Any]]] = {chunk_id: {} for chunk_id in sample_ids}
    for view, path in paths.items():
        for item in jsonl_records(path):
            chunk_id = str(item["chunk_id"])
            if chunk_id in sample_ids:
                samples[chunk_id][view] = item
    return samples


def load_mask_plan(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("books", []):
        plan[(str(item.get("author") or ""), str(item.get("title") or ""))] = item
    return plan


def sample_flags(sample: dict[str, dict[str, Any]], mask_info: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    clean = sample.get("clean", {})
    entity = sample.get("entity_masked_v3") or sample.get("entity_masked_v2") or sample.get("entity_masked", {})
    topic = sample.get("topic_distorted", {})
    structure = sample.get("structure_only", {})
    clean_text = str(clean.get("text") or "")
    entity_text = str(entity.get("text") or "")
    topic_text = str(topic.get("text") or "")
    structure_text = str(structure.get("text") or "")
    clean_cjk = max(cjk_len(clean_text), 1)
    entity_terms = [str(term) for term in mask_info.get("entity_terms_v2", []) or mask_info.get("entity_terms", [])]
    remaining_terms = [term for term in entity_terms[:80] if term and term in entity_text]
    if remaining_terms:
        flags.append("review: sampled entity mask terms still visible: " + ", ".join(remaining_terms[:8]))
    entity_ratio = cjk_len(entity_text) / clean_cjk
    if entity_ratio > 0.98:
        flags.append("review: entity_masked keeps almost all CJK content")
    if "<TERM>" in entity_text:
        flags.append("review: current entity mask still contains literal <TERM> markers")
    topic_visible_ratio = (cjk_len(topic_text) - topic_text.count("文")) / clean_cjk
    if topic_visible_ratio > 0.35:
        flags.append("review: topic_distorted may preserve too much lexical content")
    structure_non_generic = sum(1 for char in CJK_RE.findall(structure_text) if char != "文")
    if structure_non_generic:
        flags.append(f"fail: structure_only has {structure_non_generic} non-generic CJK chars")
    if CHINESE_PUNCT_RE.findall(clean_text) and not CHINESE_PUNCT_RE.findall(structure_text):
        flags.append("fail: structure_only lost Chinese punctuation")
    return flags or ["ok"]


def write_report(
    *,
    output_path: Path,
    dataset_root: Path,
    paths: dict[str, Path],
    stats: dict[str, dict[str, Any]],
    samples: list[ChunkMeta],
    sample_views: dict[str, dict[str, dict[str, Any]]],
    mask_plan: dict[tuple[str, str], dict[str, Any]],
    excerpt_chars: int,
) -> None:
    clean_rows = stats["clean"]["rows"]
    lines = [
        "# Mask Quality QA Report",
        "",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')}",
        "",
        "## Executive Summary",
        "",
    ]
    row_counts = {view: stats[view]["rows"] for view in VIEWS}
    residue_hits = sum(int(stats[view]["residue_hit_chunks"]) for view in VIEWS)
    malformed = sum(int(stats[view]["malformed_placeholder_chunks"]) for view in VIEWS)
    same_rows = len(set(row_counts.values())) == 1
    lines.extend(
        [
            f"- {'PASS' if same_rows else 'FAIL'}: all views have matching row counts ({clean_rows:,} expected).",
            f"- {'PASS' if residue_hits == 0 else 'REVIEW'}: known scrape and author-note residue hit chunks: {residue_hits}.",
            f"- {'PASS' if malformed == 0 else 'REVIEW'}: malformed placeholder chunks: {malformed}.",
            "- REVIEW: `entity_masked` should be checked by human samples because heuristic CJK n-gram masking can over-mask ordinary phrases.",
            "- REVIEW: `topic_distorted` and `structure_only` are diagnostic views, not readable training text.",
        ]
    )
    lines.extend(["", "## Input Files", ""])
    for view in VIEWS:
        lines.append(f"- {view}: `{paths[view]}`")

    lines.extend(["", "## View-Level Metrics", ""])
    lines.append(
        "| View | Rows | Authors | Books | Avg Clean CJK | Avg View CJK | View/Clean CJK | "
        "<CONTENT>/1k | <TERM>/1k | <NUM>/1k | <LATIN>/1k | 某/1k | 文/1k | Residue Chunks | Malformed |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for view in VIEWS:
        item = stats[view]
        rows = max(int(item["rows"]), 1)
        clean_cjk = int(item["clean_cjk"])
        view_cjk = int(item["view_cjk"])
        placeholders = item["placeholder_counts"]
        lines.append(
            f"| {view} | {item['rows']:,} | {item['author_count']} | {item['book_count']} | "
            f"{clean_cjk / rows:.0f} | {view_cjk / rows:.0f} | {pct(view_cjk / clean_cjk if clean_cjk else 0)} | "
            f"{per_1k(int(placeholders.get('<CONTENT>', 0)), clean_cjk):.1f} | "
            f"{per_1k(int(placeholders.get('<TERM>', 0)), clean_cjk):.1f} | "
            f"{per_1k(int(placeholders.get('<NUM>', 0)), clean_cjk):.1f} | "
            f"{per_1k(int(placeholders.get('<LATIN>', 0)), clean_cjk):.1f} | "
            f"{per_1k(int(item['generic_mou']), clean_cjk):.1f} | "
            f"{per_1k(int(item['generic_wen']), clean_cjk):.1f} | "
            f"{item['residue_hit_chunks']} | {item['malformed_placeholder_chunks']} |"
        )

    lines.extend(["", "## Split Coverage", ""])
    lines.append("| View | Train | Dev | Test | Proxy Transfer |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for view in VIEWS:
        splits = stats[view]["splits"]
        lines.append(
            f"| {view} | {splits.get('train', 0):,} | {splits.get('dev', 0):,} | "
            f"{splits.get('test', 0):,} | {splits.get('proxy_transfer', 0):,} |"
        )

    lines.extend(["", "## Human QA Samples", ""])
    for sample_index, meta in enumerate(samples, start=1):
        sample = sample_views.get(meta.chunk_id, {})
        clean = sample.get("clean", {})
        entity = sample.get("entity_masked", {})
        entity_v2 = sample.get("entity_masked_v2", {})
        entity_v3 = sample.get("entity_masked_v3", {})
        topic = sample.get("topic_distorted", {})
        structure = sample.get("structure_only", {})
        mask_info = mask_plan.get((meta.author, meta.title), {})
        flags = sample_flags(sample, mask_info)
        clean_text = str(clean.get("text") or "")
        entity_text = str(entity.get("text") or "")
        entity_v2_text = str(entity_v2.get("text") or "")
        entity_v3_text = str(entity_v3.get("text") or "")
        topic_text = str(topic.get("text") or "")
        structure_text = str(structure.get("text") or "")
        clean_cjk = max(cjk_len(clean_text), 1)
        lines.extend(
            [
                f"### Sample {sample_index}: {meta.author} / {meta.title} / {meta.split} / chunk {meta.chunk_index}",
                "",
                f"- chunk_id: `{meta.chunk_id}`",
                f"- clean CJK: {cjk_len(clean_text):,}",
                f"- entity v1 CJK retention: {pct(cjk_len(entity_text) / clean_cjk)}",
                f"- entity v2 CJK retention: {pct(cjk_len(entity_v2_text) / clean_cjk)}",
                f"- entity v3 CJK retention: {pct(cjk_len(entity_v3_text) / clean_cjk)}",
                f"- entity v1 markers: `<CONTENT>` {entity_text.count('<CONTENT>')}, `<NUM>` {entity_text.count('<NUM>')}, `<LATIN>` {entity_text.count('<LATIN>')}",
                f"- entity v2 markers: `<TERM>` {entity_v2_text.count('<TERM>')}, `<NUM>` {entity_v2_text.count('<NUM>')}, `<LATIN>` {entity_v2_text.count('<LATIN>')}",
                f"- entity v3 markers: `某` {entity_v3_text.count('某')}, `<NUM>` {entity_v3_text.count('<NUM>')}, `<LATIN>` {entity_v3_text.count('<LATIN>')}",
                f"- topic visible non-generic CJK ratio: {pct((cjk_len(topic_text) - topic_text.count('文')) / clean_cjk)}",
                f"- structure non-generic CJK chars: {sum(1 for char in CJK_RE.findall(structure_text) if char != '文')}",
                f"- flags: {'; '.join(flags)}",
                "",
                "| View | Excerpt |",
                "| --- | --- |",
                f"| clean | {markdown_inline_excerpt(clean_text, excerpt_chars)} |",
                f"| entity_masked | {markdown_inline_excerpt(entity_text, excerpt_chars)} |",
                f"| entity_masked_v2 | {markdown_inline_excerpt(entity_v2_text, excerpt_chars)} |",
                f"| entity_masked_v3 | {markdown_inline_excerpt(entity_v3_text, excerpt_chars)} |",
                f"| topic_distorted | {markdown_inline_excerpt(topic_text, excerpt_chars)} |",
                f"| structure_only | {markdown_inline_excerpt(structure_text, excerpt_chars)} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Notes",
            "",
            "- `clean` is the upper-bound diagnostic view and should not be treated as content-controlled style evidence.",
            "- `entity_masked` is the first candidate style-meter view because it removes book-specific terms while preserving readable syntax.",
            "- `topic_distorted` tests whether signal survives mostly through punctuation, function words, dialogue shape, and sentence rhythm.",
            "- `structure_only` is a stress test. It should preserve punctuation and layout, but it is not intended for production scoring by itself.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def markdown_inline_excerpt(text: str, limit: int) -> str:
    text = truncate_text(text, limit)
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return f"`{text}`"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a readable QA report for masked and unmasked style chunks.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("generated/style_research/benchmarks/mask_quality_samples.md"))
    parser.add_argument("--stats-output", type=Path, default=Path("generated/style_research/benchmarks/mask_quality_stats.json"))
    parser.add_argument("--target-author", default="非天夜翔")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--excerpt-chars", type=int, default=220)
    args = parser.parse_args()

    paths = chunk_paths(args.dataset_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit("missing chunk files: " + ", ".join(missing))

    clean_meta = load_clean_meta(paths["clean"])
    samples = select_samples(clean_meta, target_author=args.target_author, sample_count=args.sample_count)
    sample_views = load_sample_views(paths, {item.chunk_id for item in samples})
    mask_plan = load_mask_plan(args.dataset_root / "masked" / "mask_terms.json")
    stats = {view: view_stats(view, path) for view, path in paths.items()}

    write_report(
        output_path=args.output,
        dataset_root=args.dataset_root,
        paths=paths,
        stats=stats,
        samples=samples,
        sample_views=sample_views,
        mask_plan=mask_plan,
        excerpt_chars=args.excerpt_chars,
    )
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "report": str(args.output),
                "stats": stats,
                "sample_chunk_ids": [item.chunk_id for item in samples],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.output), "stats": str(args.stats_output), "samples": len(samples)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
