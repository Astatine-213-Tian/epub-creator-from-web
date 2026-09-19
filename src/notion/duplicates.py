"""Read-only shared-extra preflight and explicit decisions for possible duplicates."""

from __future__ import annotations

import asyncio
import difflib
import json
import shlex
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from opencc import OpenCC

from src.content.blocks import content_signature
from src.notion.markdown import from_markdown, to_markdown
from src.notion.reader import NotionReader, notion_id
from src.runtime.files import digest, write_json

SIMILARITY_THRESHOLD = 0.70
CONTAINMENT_THRESHOLD = 0.60
MIN_FUZZY_LENGTH = 80
OVERLAP_WINDOW = 20
MIN_LOCAL_OVERLAP = 60
SIMPLIFIED = OpenCC("t2s")


def fingerprint(title: str, blocks: list[dict]) -> str:
    """Exact editable title/content snapshot; never normalize an approval."""
    return digest(
        json.dumps([title, content_signature(blocks)], ensure_ascii=False).encode()
    )


def comparison_text(blocks: list[dict]) -> str:
    text = "".join(run["text"] for block in blocks for run in block["runs"])
    text = SIMPLIFIED.convert(unicodedata.normalize("NFKC", text)).casefold()
    return "".join(char for char in text if char.isalnum())


@dataclass
class TextProfile:
    text: str
    grams: Counter

    @classmethod
    def from_blocks(cls, blocks: list[dict]) -> TextProfile:
        text = comparison_text(blocks)
        return cls(text, Counter(text[i : i + 4] for i in range(len(text) - 3)))

    @cached_property
    def windows(self) -> set[str]:
        return {
            self.text[i : i + OVERLAP_WINDOW]
            for i in range(len(self.text) - OVERLAP_WINDOW + 1)
        }


def shared_regions(text: str, windows: set[str]) -> list[tuple[int, int]]:
    """Union of matching long windows; a character contributes at most once."""
    regions: list[tuple[int, int]] = []
    for start in range(len(text) - OVERLAP_WINDOW + 1):
        if text[start : start + OVERLAP_WINDOW] not in windows:
            continue
        end = start + OVERLAP_WINDOW
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))
    return regions


def local_overlap(left: TextProfile, right: TextProfile) -> tuple[int, list[str]]:
    """Find shared passages anywhere, independent of document size or order."""
    windows = left.windows & right.windows
    if not windows:
        return 0, []
    source_regions = shared_regions(left.text, windows)
    existing_regions = shared_regions(right.text, windows)
    # Taking the smaller union prevents repeats on one side from inflating the
    # evidence. Overlapping windows are not counted as separate passages.
    chars = min(
        sum(end - start for start, end in source_regions),
        sum(end - start for start, end in existing_regions),
    )
    snippets = []
    for start, end in source_regions:
        snippet = left.text[start : min(start + 100, end)]
        # A union of matching windows need not be contiguous in the other text.
        # Every excerpt displayed must itself occur literally on both sides.
        while snippet not in right.text:
            snippet = snippet[:-1]
        if snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) == 3:
            break
    return chars, snippets


def similarity(left: TextProfile, right: TextProfile) -> dict | None:
    """Flag global similarity, partial containment, or substantial local overlap."""
    shorter = min(len(left.text), len(right.text))
    if not shorter:
        return None
    if left.text == right.text:
        score = coverage = 1.0
        reasons = ["归一化正文相同"]
        shared_chars, snippets = shorter, [left.text[:100]]
    else:
        if shorter < MIN_LOCAL_OVERLAP:
            return None
        common = (left.grams & right.grams).total()
        a, b = left.grams.total(), right.grams.total()
        score = 2 * common / (a + b)
        coverage = common / min(a, b)
        reasons = []
        if shorter >= MIN_FUZZY_LENGTH:
            if score >= SIMILARITY_THRESHOLD:
                reasons.append("整体相似")
            if coverage >= CONTAINMENT_THRESHOLD:
                reasons.append("部分内容包含")
        # Each common 20-character window implies at least 17 shared 4-grams.
        # Skip the larger index when even a single such window is impossible.
        shared_chars, snippets = (
            local_overlap(left, right) if common >= OVERLAP_WINDOW - 3 else (0, [])
        )
        if shared_chars >= MIN_LOCAL_OVERLAP:
            reasons.append("局部文字重合")
        if not reasons:
            return None
    return {
        "similarity": round(score, 6),
        "containment": round(coverage, 6),
        "source_length": len(left.text),
        "existing_length": len(right.text),
        "reasons": reasons,
        "overlap_chars": shared_chars,
        "overlap_snippets": snippets,
    }


def review_key(source: str, candidates: list[dict]) -> str:
    snapshots = sorted((c["page_id"], c["fingerprint"]) for c in candidates)
    return digest(json.dumps([source, snapshots]).encode())


def review_report(book: dict, state: Path, documents: dict) -> Path:
    path = state.parent / "extra-review.md"
    lines = [
        "# 共享番外重复核对",
        "",
        "疑似重复条目确认前暂停上传。复用保留 Notion 现有标题和正文，只增加作品关联。",
        "相似度按正文四字片段重合计算，不是正确率；包含率指较短版本被覆盖的比例。",
        "比较忽略繁简、标点、空白、大小写和排版；差异摘要保留原文。",
        "",
    ]
    command = "uv run book-notion resolve-extra --state " + shlex.quote(str(state))
    for index, item in enumerate(book["extras"], 1):
        review = item.get("duplicate_review")
        if not review or item.get("page_id"):
            continue
        status = "已选择" if review.get("decision") else "待确认"
        lines.extend([f"## {index}. {item['title']}（{status}）", ""])
        if not review["candidates"]:
            lines.extend(["之前的候选已移除或内容有变；请重新确认是否新建。", ""])
        for candidate in review["candidates"]:
            page_id = candidate["page_id"]
            lines.extend(
                [
                    f"### [{candidate['title']}](https://www.notion.so/{page_id})",
                    "",
                    f"正文相似度：{candidate['similarity']:.1%}；包含率：{candidate['containment']:.1%}。",
                    f"归一化字数：新爬取 {candidate['source_length']} / 已有 {candidate['existing_length']}。",
                    "提醒原因：" + "、".join(candidate["reasons"]) + "。",
                    f"重合内容覆盖至少 {candidate['overlap_chars']} 字（两侧分别去除重叠后取较小值）。",
                    "",
                    "```bash",
                    f"{command} --extra {index} --use-existing {page_id}",
                    "```",
                    "",
                ]
            )
            if candidate["overlap_snippets"]:
                lines.extend(["归一化重合片段（最多 3 处，每处最多 100 字）：", ""])
                for snippet in candidate["overlap_snippets"]:
                    lines.extend(["> " + snippet, ""])
            changes = list(
                difflib.unified_diff(
                    documents[page_id]["markdown"].splitlines(),
                    to_markdown(item["blocks"]).splitlines(),
                    fromfile="Notion 已有正文",
                    tofile="新爬取正文",
                    n=2,
                    lineterm="",
                )
            )
            if changes:
                # Bound the human preview, not the comparison or candidate list.
                lines.extend(
                    [
                        "<details><summary>原文差异摘要（最多 80 行）</summary>",
                        "",
                        "````diff",
                    ]
                )
                lines.extend(line[:500] for line in changes[:80])
                lines.extend(
                    [
                        "````",
                        "",
                        "长行截至 500 字；完整来源见同目录 import.json，已有正文见上方 Notion 链接。",
                        "</details>",
                        "",
                    ]
                )
            else:
                lines.extend(["正文 Markdown 相同；请核对标题及作品关联。", ""])
        lines.extend(
            [
                "确认是独立内容，需要新建：",
                "",
                "```bash",
                f"{command} --extra {index} --create-new",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "选择完成后续传：",
            "",
            "```bash",
            f"uv run book-notion resume --state {shlex.quote(str(state))}",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))
    return path


def resolve_extra(state: Path, index: int, *, use_existing: str | None = None) -> None:
    """Save a local approval bound to the report snapshots; caller holds state lock."""
    book = json.loads(state.read_text())
    if not 1 <= index <= len(book["extras"]):
        raise ValueError("--extra must be a 1-based index from extra-review.md")
    item = book["extras"][index - 1]
    review = item.get("duplicate_review")
    if item.get("page_id") or not review:
        raise ValueError("This extra has no pending duplicate review")
    if fingerprint(item["title"], item["blocks"]) != review["source_fingerprint"]:
        raise ValueError("Source changed after review; resume to refresh the report")
    decision = {"action": "create", "review_key": review["key"]}
    if use_existing:
        page_id = notion_id(use_existing)
        if page_id not in {c["page_id"] for c in review["candidates"]}:
            raise ValueError("Choose an existing page listed in this extra's review")
        decision.update(action="reuse", page_id=page_id)
    review["decision"] = decision
    write_json(state, book)
    print(
        f"Saved extra {index}: {decision['action']}; run book-notion resume to recheck and upload"
    )


async def preflight_extras(
    book: dict, state: Path, config: dict, reader: NotionReader
) -> None:
    """Inspect the whole library once; never infer absence from ranked search results."""
    pending = [
        item
        for item in book["extras"]
        if not item.get("page_id") or (item.get("reused") and not item.get("verified"))
    ]
    if not pending:
        return
    if any(item.get("pending") for item in pending):
        raise ValueError(
            "A page create response was lost; reconcile the checkpoint before retrying"
        )
    database = config["databases"]["extras"]
    view = await reader.view(database["view_id"])
    if notion_id(view["dataSourceUrl"]) != database["data_source_id"] or any(
        view.get(key) for key in ("advancedFilter", "filter", "quickFilters")
    ):
        raise ValueError(
            "Shared-extra similarity check requires an unfiltered inventory view"
        )
    rows = await reader.rows(database["view_id"])
    print(
        f"Checking {len(pending)} incoming extras against {len(rows)} shared Notion pages…",
        flush=True,
    )
    semaphore = asyncio.Semaphore(4)

    async def read(row: dict) -> tuple[str, dict]:
        async with semaphore:
            props, markdown = await reader.document(row["id"])
        blocks = from_markdown(markdown)
        title = props.get("番外")
        if not isinstance(title, str):
            raise ValueError(
                "Shared extra is missing its title; duplicate check incomplete"
            )
        return row["id"], {
            "title": title,
            "markdown": markdown,
            "fingerprint": fingerprint(title, blocks),
            "profile": TextProfile.from_blocks(blocks),
        }

    # Any unreadable page fails the check rather than silently treating it as novel content.
    documents = dict(await asyncio.gather(*(read(row) for row in rows)))
    links, unresolved = [], 0
    for item in pending:
        # An interrupted upload may have selected a page without verifying the
        # relation yet. Re-evaluate that choice on resume, just like a new one.
        previously_selected = bool(item.get("reused"))
        if previously_selected:
            for key in ("page_id", "reused", "reuse_fingerprint"):
                item.pop(key, None)
        profile = TextProfile.from_blocks(item["blocks"])
        source = fingerprint(item["title"], item["blocks"])
        candidates = []
        for page_id, document in documents.items():
            if score := similarity(profile, document["profile"]):
                candidates.append(
                    {
                        "page_id": page_id,
                        "title": document["title"],
                        "fingerprint": document["fingerprint"],
                        **score,
                    }
                )
        candidates.sort(
            key=lambda c: (-c["similarity"], -c["containment"], c["page_id"])
        )
        previous = item.get("duplicate_review", {})
        if not candidates and not previous and not previously_selected:
            continue
        # Keep exact-title + exact-presentation reuse; renamed/proofread text needs a decision.
        if (
            len(candidates) == 1
            and candidates[0]["fingerprint"] == source
            and not previous
        ):
            links.append((item, candidates[0]))
            continue
        key = review_key(source, candidates)
        review = {"source_fingerprint": source, "key": key, "candidates": candidates}
        decision = previous.get("decision", {})
        if decision.get("review_key") == key:
            review["decision"] = decision
            if decision["action"] == "reuse":
                links.append(
                    (
                        item,
                        next(
                            c for c in candidates if c["page_id"] == decision["page_id"]
                        ),
                    )
                )
        else:
            unresolved += 1
        item["duplicate_review"] = review
    write_json(state, book)
    if any(item.get("duplicate_review") for item in pending):
        report = review_report(book, state, documents)
        if unresolved:
            raise ValueError(
                f"WARNING: {unresolved} shared extras may already exist. Upload paused for manual review: {report}"
            )
    ids = [item["page_id"] for item in book["extras"] if item.get("page_id")]
    ids.extend(candidate["page_id"] for _, candidate in links)
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Multiple source extras would link to the same page; reconcile the source outline"
        )
    for item, candidate in links:
        item.update(
            page_id=candidate["page_id"],
            reused=True,
            reuse_fingerprint=candidate["fingerprint"],
        )
    write_json(state, book)
