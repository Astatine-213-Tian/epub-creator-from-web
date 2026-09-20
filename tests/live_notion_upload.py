"""Verify synthetic crawl output through the real Notion upload workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

CRAWLER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRAWLER))

from notion_books import FIELDS, NotionBooks, from_markdown, relation_ids

from src.content.blocks import content_signature
from src.content.models import Chapter, Volume
from src.crawler.models import CrawledBook
from src.notion.mcp import AUTH_FILE, TokenStore, connect
from src.workflows.ingest import OutputOptions, write_outputs


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    path.chmod(0o600)


def load(path: Path):
    return json.loads(path.read_text())


def source(run: dict) -> CrawledBook:
    marker = run["marker"]
    return CrawledBook(
        title=f"[E2E {marker}] 合成书籍",
        author=f"[E2E {marker}] 测试作者",
        source_url=f"https://example.invalid/notion-books-e2e/{marker}",
        intro_paragraphs=[f"简介校验 {marker}：仅用于验证真实导入与回读。"],
        volumes=[
            Volume(
                "测试卷",
                [
                    Chapter(
                        "第一章 格式",
                        html_blocks=[
                            f"<p>FIRST-{marker}：普通正文与中文。</p>",
                            "<p><strong>粗体</strong>、<em>斜体</em>、<u>下划线</u>、<s>删除线</s>。</p>",
                            "<p>第一行<br/>第二行</p>",
                            "<p></p>",
                            "<h3>三级标题</h3>",
                            '<p style="text-align: center">居中文字</p>',
                            '<p style="text-align: right">右侧署名</p>',
                        ],
                    ),
                    Chapter(
                        "第二章 保持",
                        [f"SECOND-{marker}：此章在其他章节更新时必须保持原样。"],
                    ),
                ],
            ),
            Volume(
                "番外",
                [
                    Chapter(
                        f"独立番外 {marker}",
                        [f"EXTRA-{marker}：共享番外的唯一测试内容。"],
                    )
                ],
            ),
        ],
    )


async def with_notion(action):
    store = TokenStore(AUTH_FILE)
    with store.locked():
        async with connect(store) as tools:
            return await action(NotionBooks(tools), tools)


def import_book(root: Path) -> None:
    manifest = root / "run.json"
    run = (
        load(manifest)
        if manifest.exists()
        else {"marker": uuid.uuid4().hex[:12], "started_at": time.time()}
    )
    config = load(CRAWLER / "book_specs/notion/config.json")
    run["config"] = config
    save(manifest, run)
    save(root / "book_specs/notion/config.json", config)
    os.chdir(root)
    try:
        result = write_outputs(source(run), OutputOptions(output_formats=("notion",)))
    finally:
        checkpoints = list(root.glob("generated/notion_cms_sources/*/import.json"))
        if len(checkpoints) == 1:
            run["import_state"] = str(checkpoints[0])
            run["work_id"] = load(checkpoints[0]).get("work_id")
        save(manifest, run)
    book = load(result.notion_state)

    async def verify(reader, tools):
        work, _ = await reader.document(book["work_id"])
        assert work[FIELDS["title"]] == book["metadata"]["title"]
        run["author_ids"] = relation_ids(work[FIELDS["authors"]])
        save(manifest, run)
        for author_id in run["author_ids"]:
            author, _ = await reader.document(author_id)
            assert author[FIELDS["authors"]] == book["metadata"]["creator"]
        for item in list(book["chapters"].values()) + book["extras"]:
            props, body = await reader.document(item["page_id"])
            title_field = (
                "chapter_title" if item in book["chapters"].values() else "extra_title"
            )
            assert props[FIELDS[title_field]] == item["title"]
            assert content_signature(from_markdown(body)) == content_signature(
                item["blocks"]
            )
        print(
            "PASS live Notion readback: all chapter and extra blocks match", flush=True
        )

    asyncio.run(with_notion(verify))
    run["import_verified"] = True
    save(manifest, run)
    # Exercise the real completed-checkpoint path, with no recreated pages.
    repeated = write_outputs(source(run), OutputOptions(output_formats=("notion",)))
    assert repeated.notion_state.resolve() == result.notion_state.resolve()
    assert load(repeated.notion_state) == book
    run["resume_verified"] = True
    save(manifest, run)
    print("PASS import resume preserves page identities and checkpoint", flush=True)


def verify_cleanup(root: Path) -> None:
    run = load(root / "run.json")
    book = load(Path(run["import_state"])) if run.get("import_state") else {}
    fixture = source(run)

    async def verify(reader, tools):
        expected = {
            "works": (
                {book.get("work_id"), run.get("work_id")},
                FIELDS["title"],
                fixture.title,
            ),
            "authors": (set(run.get("author_ids", [])), FIELDS["authors"], fixture.author),
            "extras": (
                {item.get("page_id") for item in book.get("extras", [])},
                FIELDS["extra_title"],
                f"独立番外 {run['marker']}",
            ),
        }
        for kind, (removed, title_field, title) in expected.items():
            rows = await reader.rows(run["config"]["databases"][kind]["view_id"])
            # Unique fixture names also catch a create whose response was lost.
            remaining = {
                row["id"]
                for row in rows
                if row["id"] in removed or row.get(title_field) == title
            }
            assert not remaining, f"Test {kind} still active: {sorted(remaining)}"

    asyncio.run(with_notion(verify))
    run["notion_cleanup_verified"] = True
    save(root / "run.json", run)
    print("PASS test work, author and extras removed from live inventories", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["import", "verify-cleanup"])
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    root = args.state.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.mode == "import":
        import_book(root)
    else:
        verify_cleanup(root)


if __name__ == "__main__":
    main()
