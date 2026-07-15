#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
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
    r"最新网址|返回目录|手机阅读|这个段落是图片段落|请访问正确的网站|"
    r"原版未篡改内容请移至|关闭广告拦截功能|退出浏览器阅读模式|"
    r"https?://|www\.",
    re.I,
)
PLACEHOLDERS = ("<CONTENT>", "<TERM>", "<NAME>", "<PLACE>", "<ORG>", "<NUM>", "<LATIN>")
VIEWS = (
    "clean",
    "train_global_masked",
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


def view_stats(
    view: str,
    path: Path,
    *,
    clean_natural_mou: dict[str, int] | None = None,
) -> dict[str, Any]:
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
        "_mask_density_by_split": defaultdict(Counter),
        "_mask_density_by_author": defaultdict(Counter),
        "_mask_density_by_book": defaultdict(Counter),
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
            chunk_id = str(item["chunk_id"])
            mask_count = max(
                text.count("某") - (clean_natural_mou or {}).get(chunk_id, 0),
                0,
            )
            clean_cjk = int(item["chunk_clean_cjk_count"])
            stats["generic_mou"] += mask_count
            density_groups = (
                ("_mask_density_by_split", str(item["split"])),
                ("_mask_density_by_author", str(item["author"])),
                (
                    "_mask_density_by_book",
                    f"{item['author']} / {item['title']}",
                ),
            )
            for group_name, group_key in density_groups:
                bucket = stats[group_name][group_key]
                bucket["rows"] += 1
                bucket["clean_cjk"] += clean_cjk
                bucket["mask_chars"] += mask_count
    stats["author_count"] = len(stats["authors"])
    stats["book_count"] = len(stats["books"])
    stats["authors"] = dict(stats["authors"])
    stats["books"] = len(stats["books"])
    stats["splits"] = dict(stats["splits"])
    stats["placeholder_counts"] = dict(stats["placeholder_counts"])
    for private_name, public_name in (
        ("_mask_density_by_split", "mask_density_by_split"),
        ("_mask_density_by_author", "mask_density_by_author"),
        ("_mask_density_by_book", "mask_density_by_book"),
    ):
        groups = stats.pop(private_name)
        stats[public_name] = {
            key: {
                "rows": int(values["rows"]),
                "clean_cjk": int(values["clean_cjk"]),
                "mask_chars": int(values["mask_chars"]),
                "mask_chars_per_1k_cjk": per_1k(
                    int(values["mask_chars"]), int(values["clean_cjk"])
                ),
            }
            for key, values in sorted(groups.items())
        }
    return stats


def load_clean_natural_mou(path: Path) -> dict[str, int]:
    return {
        str(item["chunk_id"]): str(item["text"]).count("某")
        for item in jsonl_records(path)
    }


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


def load_mask_plan(
    path: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str], dict[str, Any]]:
    if not path.exists():
        return {}, [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    plan: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("books", []):
        plan[(str(item.get("author") or ""), str(item.get("title") or ""))] = item
    global_terms = [
        str(term)
        for term in payload.get("global_terms", {}).get("entity_terms_v2", [])
    ]
    return plan, global_terms, payload


def sample_flags(
    sample: dict[str, dict[str, Any]],
    mask_info: dict[str, Any],
    global_terms: list[str],
) -> list[str]:
    flags: list[str] = []
    clean = sample.get("clean", {})
    selected = sample.get("train_global_masked", {})
    entity = sample.get("entity_masked_v3", {})
    topic = sample.get("topic_distorted", {})
    structure = sample.get("structure_only", {})
    clean_text = str(clean.get("text") or "")
    selected_text = str(selected.get("text") or "")
    entity_text = str(entity.get("text") or "")
    topic_text = str(topic.get("text") or "")
    structure_text = str(structure.get("text") or "")
    clean_cjk = max(cjk_len(clean_text), 1)
    clean_natural_mou = clean_text.count("某")
    selected_mask_count = max(
        selected_text.count("某") - clean_natural_mou,
        0,
    )
    entity_mask_count = max(entity_text.count("某") - clean_natural_mou, 0)
    remaining_global_terms = [
        term for term in global_terms if term and term in selected_text
    ]
    if remaining_global_terms:
        flags.append(
            "fail: sampled train-global mask terms still visible: "
            + ", ".join(remaining_global_terms[:8])
        )
    selected_visible_ratio = (
        cjk_len(selected_text) - selected_mask_count
    ) / clean_cjk
    if selected_visible_ratio > 0.995:
        flags.append("review: train_global_masked keeps almost all CJK content")
    entity_terms = [str(term) for term in mask_info.get("entity_terms_v2", []) or mask_info.get("entity_terms", [])]
    remaining_terms = [term for term in entity_terms[:80] if term and term in entity_text]
    if remaining_terms:
        flags.append("review: sampled entity mask terms still visible: " + ", ".join(remaining_terms[:8]))
    entity_visible_ratio = (
        cjk_len(entity_text) - entity_mask_count
    ) / clean_cjk
    if entity_visible_ratio > 0.98:
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
    global_terms: list[str],
    masking_policy: str,
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
        f"- Masking policy: `{masking_policy}`; fixed global vocabulary: {len(global_terms):,} terms.",
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
            "- REVIEW: `train_global_masked` should be checked by human samples and density breakdowns because heuristic CJK n-gram masking can over-mask ordinary phrases.",
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

    selected_density = stats["train_global_masked"]
    lines.extend(["", "## Selected Mask Density", ""])
    lines.append(
        "The selected train-global representation preserves mask length, so density is "
        "reported as masked `某` characters per 1,000 clean CJK characters."
    )
    lines.extend(["", "### By Split", ""])
    lines.append("| Split | Rows | Clean CJK | Masked CJK | 某/1k CJK |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label, values in selected_density["mask_density_by_split"].items():
        lines.append(
            f"| {label} | {values['rows']:,} | {values['clean_cjk']:,} | "
            f"{values['mask_chars']:,} | {values['mask_chars_per_1k_cjk']:.1f} |"
        )

    for group_key, heading, display_count in (
        ("mask_density_by_author", "Authors", 10),
        ("mask_density_by_book", "Books", 12),
    ):
        values_by_label = selected_density[group_key]
        ordered = sorted(
            values_by_label.items(),
            key=lambda item: (-item[1]["mask_chars_per_1k_cjk"], item[0]),
        )
        densities = [item[1]["mask_chars_per_1k_cjk"] for item in ordered]
        lines.extend(["", f"### By {heading}", ""])
        if densities:
            lines.append(
                f"Range {min(densities):.1f}-{max(densities):.1f}; "
                f"median {statistics.median(densities):.1f} masked characters per 1,000 CJK."
            )
            lines.append("")
        lines.append(f"| {heading[:-1]} | Rows | Clean CJK | Masked CJK | 某/1k CJK |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for label, values in ordered[:display_count]:
            lines.append(
                f"| {label} | {values['rows']:,} | {values['clean_cjk']:,} | "
                f"{values['mask_chars']:,} | {values['mask_chars_per_1k_cjk']:.1f} |"
            )
        lines.append("")
        lines.append(
            f"The JSON statistics retain all {len(ordered)} {heading.lower()}; "
            f"the table shows the {min(display_count, len(ordered))} highest-density cases."
        )

    lines.extend(["", "## Human QA Samples", ""])
    for sample_index, meta in enumerate(samples, start=1):
        sample = sample_views.get(meta.chunk_id, {})
        clean = sample.get("clean", {})
        selected = sample.get("train_global_masked", {})
        entity_v3 = sample.get("entity_masked_v3", {})
        topic = sample.get("topic_distorted", {})
        structure = sample.get("structure_only", {})
        mask_info = mask_plan.get((meta.author, meta.title), {})
        flags = sample_flags(sample, mask_info, global_terms)
        clean_text = str(clean.get("text") or "")
        selected_text = str(selected.get("text") or "")
        entity_v3_text = str(entity_v3.get("text") or "")
        topic_text = str(topic.get("text") or "")
        structure_text = str(structure.get("text") or "")
        clean_cjk = max(cjk_len(clean_text), 1)
        selected_mask_count = max(
            selected_text.count("某") - clean_text.count("某"),
            0,
        )
        lines.extend(
            [
                f"### Sample {sample_index}: {meta.author} / {meta.title} / {meta.split} / chunk {meta.chunk_index}",
                "",
                f"- chunk_id: `{meta.chunk_id}`",
                f"- clean CJK: {cjk_len(clean_text):,}",
                f"- train-global masked share: {pct(selected_mask_count / clean_cjk)}",
                f"- book-local diagnostic masked share: {pct(max(entity_v3_text.count('某') - clean_text.count('某'), 0) / clean_cjk)}",
                f"- entity v3 inserted masks: `某` {max(entity_v3_text.count('某') - clean_text.count('某'), 0)}, `<NUM>` {entity_v3_text.count('<NUM>')}, `<LATIN>` {entity_v3_text.count('<LATIN>')}",
                f"- topic visible non-generic CJK ratio: {pct((cjk_len(topic_text) - topic_text.count('文')) / clean_cjk)}",
                f"- structure non-generic CJK chars: {sum(1 for char in CJK_RE.findall(structure_text) if char != '文')}",
                f"- flags: {'; '.join(flags)}",
                "",
                "| View | Excerpt |",
                "| --- | --- |",
                f"| clean | {markdown_inline_excerpt(clean_text, excerpt_chars)} |",
                f"| train_global_masked | {markdown_inline_excerpt(selected_text, excerpt_chars)} |",
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
            "- `train_global_masked` is the selected style-meter view: its vocabulary is fit on training books only and the same fixed transform is applied to every split.",
            "- `entity_masked_v3` is a more aggressive book-local diagnostic view, not the selected attribution representation.",
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
    mask_plan_path = args.dataset_root / "masked" / "mask_terms.json"
    mask_plan, global_terms, mask_plan_payload = load_mask_plan(mask_plan_path)
    clean_natural_mou = load_clean_natural_mou(paths["clean"])
    stats = {
        view: view_stats(
            view,
            path,
            clean_natural_mou=clean_natural_mou,
        )
        for view, path in paths.items()
    }

    write_report(
        output_path=args.output,
        dataset_root=args.dataset_root,
        paths=paths,
        stats=stats,
        samples=samples,
        sample_views=sample_views,
        mask_plan=mask_plan,
        global_terms=global_terms,
        masking_policy=str(mask_plan_payload.get("masking_policy") or "unknown"),
        excerpt_chars=args.excerpt_chars,
    )
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "report": str(args.output),
                "mask_plan": str(mask_plan_path),
                "mask_plan_sha256": hashlib.sha256(mask_plan_path.read_bytes()).hexdigest(),
                "masking_policy": mask_plan_payload.get("masking_policy"),
                "global_mask_term_count": len(global_terms),
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
