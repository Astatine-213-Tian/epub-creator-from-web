#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


from experiments.shared.paths import RESEARCH_ROOT


REPO_ROOT = RESEARCH_ROOT

from src.crawl.snapshot import write_json  # noqa: E402


REFUSAL_RE = re.compile(
    r"抱歉|无法(?:完成|提供|翻译)|I can.?t|I cannot|As an AI|作为.{0,8}AI|```|"
    r"<REDACTED>|PLACEHOLDER|TODO|\"paragraphs\"\s*:|\"content_plan\"\s*:"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def element_text(element: ElementTree.Element) -> str:
    return "".join(element.itertext()).strip()


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Eternal Gate Method4 EPUB Validation",
        "",
        f"> Status: **{report['status']}**",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        f"| EPUB | `{report['epub_path']}` |",
        f"| SHA-256 | `{report['epub_sha256']}` |",
        f"| Archive size | {report['epub_size_bytes']:,} bytes |",
        f"| Source/chapter pages | {report['chapter_count']} |",
        f"| Bilingual Chinese paragraphs | {report['paragraph_count']:,} |",
        f"| Navigation chapter links | {report['nav_chapter_link_count']} |",
        f"| NCX chapter entries | {report['ncx_chapter_entry_count']} |",
        f"| Independent repairs present | {report['repair_count']} / {report['expected_repair_count']} |",
        f"| Refusal/JSON residue hits | {len(report['residue_hits'])} |",
        f"| Failures | {len(report['failures'])} |",
        "",
        "## Per Chapter",
        "",
        "| ID | Title | Chinese paragraphs | Exact text match |",
        "| --- | --- | ---: | --- |",
    ]
    for row in report["chapters"]:
        lines.append(
            f"| `{row['chapter_id']}` | {row['title']} | {row['paragraph_count']} | "
            f"{'PASS' if row['exact_text_match'] else 'FAIL'} |"
        )
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("None.")
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "uv run python experiments/iteration5/application/audit_method4_bilingual_epub.py \\",
            f"  --epub {report['epub_path']} \\",
            f"  --snapshot-dir {report['snapshot_dir']} \\",
            f"  --style-run-dir {report['style_run_dir']}",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(*, epub_path: Path, snapshot_dir: Path, style_run_dir: Path) -> dict[str, Any]:
    failures: list[str] = []
    manifest = read_json(snapshot_dir / "manifest.json")
    chapters = list(manifest.get("chapters") or [])
    repair_path = style_run_dir / "audits/independent_semantic_repairs.v1.json"
    repair_spec = read_json(repair_path) if repair_path.exists() else {"repairs": []}
    repair_texts = [str(row["replacement_zh"]) for row in repair_spec.get("repairs") or []]

    chapter_reports: list[dict[str, Any]] = []
    all_zh: list[str] = []
    with zipfile.ZipFile(epub_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            failures.append(f"CRC failure: {bad_member}")
        members = archive.infolist()
        if not members or members[0].filename != "mimetype":
            failures.append("mimetype is not the first archive member")
        elif members[0].compress_type != zipfile.ZIP_STORED:
            failures.append("mimetype is compressed")
        chapter_members = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"EPUB/chap_01_\d{3}\.xhtml", name)
        )
        if len(chapter_members) != len(chapters):
            failures.append(
                f"chapter XHTML count {len(chapter_members)} != source {len(chapters)}"
            )

        for number, chapter in enumerate(chapters, 1):
            chapter_id = str(chapter["id"])
            member = f"EPUB/chap_01_{number:03d}.xhtml"
            source_payload = read_json(snapshot_dir / str(chapter["path"]))
            source_count = sum(
                1
                for block in source_payload.get("blocks") or []
                if block.get("type") == "content"
            )
            expected_payload = read_json(
                style_run_dir / "translations" / f"{chapter_id}.json"
            )
            expected = [
                str(row.get("zh") or "").strip()
                for row in expected_payload.get("translations") or []
            ]
            if member not in archive.namelist():
                failures.append(f"missing chapter member: {member}")
                observed: list[str] = []
                title = ""
            else:
                root = ElementTree.fromstring(archive.read(member))
                title_nodes = [node for node in root.iter() if local_name(node.tag) == "h2"]
                title = element_text(title_nodes[0]) if title_nodes else ""
                observed = [
                    element_text(node)
                    for node in root.iter()
                    if local_name(node.tag) == "p"
                    and "zh-translation" in str(node.get("class") or "").split()
                ]
            exact = observed == expected
            if not exact:
                failures.append(f"Chinese paragraph mismatch in chapter {chapter_id}")
            if len(expected) != source_count:
                failures.append(
                    f"source/translation count mismatch in chapter {chapter_id}: "
                    f"{source_count} != {len(expected)}"
                )
            if title != str(chapter["title"]):
                failures.append(
                    f"chapter title mismatch {chapter_id}: {title!r} != {chapter['title']!r}"
                )
            all_zh.extend(observed)
            chapter_reports.append(
                {
                    "chapter_id": chapter_id,
                    "title": str(chapter["title"]),
                    "source_paragraph_count": source_count,
                    "paragraph_count": len(observed),
                    "exact_text_match": exact,
                }
            )

        nav_root = ElementTree.fromstring(archive.read("EPUB/nav.xhtml"))
        nav_hrefs = [
            str(node.get("href") or "")
            for node in nav_root.iter()
            if local_name(node.tag) == "a"
        ]
        expected_chapter_targets = {
            f"chap_01_{number:03d}.xhtml" for number in range(1, len(chapters) + 1)
        }
        nav_chapters = {
            href
            for href in nav_hrefs
            if re.fullmatch(r"chap_01_\d{3}\.xhtml", href)
        }
        if nav_chapters != expected_chapter_targets:
            failures.append(
                "nav chapter targets do not match source chapters"
            )

        ncx_root = ElementTree.fromstring(archive.read("EPUB/toc.ncx"))
        ncx_sources = [
            str(node.get("src") or "")
            for node in ncx_root.iter()
            if local_name(node.tag) == "content"
        ]
        ncx_chapters = {
            src
            for src in ncx_sources
            if re.fullmatch(r"chap_01_\d{3}\.xhtml", src)
        }
        if ncx_chapters != expected_chapter_targets:
            failures.append(
                "NCX chapter targets do not match source chapters"
            )

        opf_text = archive.read("EPUB/content.opf").decode("utf-8", errors="replace")
        if html.escape(str(manifest["title"])) not in opf_text:
            failures.append("book title missing from content.opf")
        if html.escape(str(manifest["author"])) not in opf_text:
            failures.append("book author missing from content.opf")

    joined_zh = "\n".join(all_zh)
    residue_hits = sorted(set(REFUSAL_RE.findall(joined_zh)))
    if residue_hits:
        failures.append(f"refusal/JSON residue found: {residue_hits}")
    missing_repairs = [text for text in repair_texts if text not in all_zh]
    if missing_repairs:
        failures.append(f"missing independently repaired paragraphs: {len(missing_repairs)}")

    return {
        "schema_version": 1,
        "schema": "method4_bilingual_epub_validation.v1",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "PASS" if not failures else "FAIL",
        "epub_path": str(epub_path),
        "epub_sha256": file_sha256(epub_path),
        "epub_size_bytes": epub_path.stat().st_size,
        "snapshot_dir": str(snapshot_dir),
        "style_run_dir": str(style_run_dir),
        "chapter_count": len(chapter_reports),
        "paragraph_count": len(all_zh),
        "nav_chapter_link_count": len(nav_chapters),
        "ncx_chapter_entry_count": len(ncx_chapters),
        "expected_repair_count": len(repair_texts),
        "repair_count": len(repair_texts) - len(missing_repairs),
        "residue_hits": residue_hits,
        "chapters": chapter_reports,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the Eternal Gate method4 EPUB.")
    parser.add_argument("--epub", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--style-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(
        epub_path=args.epub.expanduser().resolve(),
        snapshot_dir=args.snapshot_dir.expanduser().resolve(),
        style_run_dir=args.style_run_dir.expanduser().resolve(),
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else args.style_run_dir.expanduser().resolve() / "audits"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "method4_epub_validation.json", report)
    write_markdown(output_dir / "method4_epub_validation.md", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
