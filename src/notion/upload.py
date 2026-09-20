"""Upload prepared crawls as CMS drafts; CMS owns publication after upload."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

from notion_books import (
    FIELDS,
    NotionBooks,
    chapter_entries,
    from_markdown,
    relation_ids,
    to_markdown,
)

from src.notion.cms import (
    CONFIG,
    STORAGE,
    ensure_views,
    ensure_work,
)
from src.notion.duplicates import fingerprint, preflight_extras
from src.notion.mcp import (
    AUTH_FILE,
    TokenStore,
    connect,
    error_message,
    exclusive_lock,
)
from src.runtime.files import digest, write_json


def source_digest(book: dict) -> str:
    source = {k: book[k] for k in ("metadata", "sections", "chapters", "extras")}
    return digest(json.dumps(source, ensure_ascii=False, sort_keys=True).encode())


async def upload_row(
    item: dict,
    book: dict,
    state: Path,
    *,
    data_source: str,
    properties: dict,
    title_property: str,
    tools,
) -> None:
    reader = NotionBooks(tools)
    if not item.get("page_id"):
        if item.get("pending"):
            raise ValueError(
                "A page create response was lost; reconcile the checkpoint before retrying"
            )
        item["pending"] = True
        write_json(state, book)
        item["page_id"] = await reader.create_page(
            data_source,
            properties,
            content=to_markdown(item["blocks"]),
        )
        item.pop("pending")
        write_json(state, book)
    if item.get("verified"):
        # A completed row now belongs to the editor; never overwrite later edits.
        return
    props, body = await reader.document(item["page_id"])
    expected = item.get("reuse_fingerprint") or fingerprint(
        item["title"], item["blocks"]
    )
    if fingerprint(props.get(title_property), from_markdown(body)) != expected:
        raise ValueError(
            "CMS draft readback differs from prepared content; reconcile without overwriting edits"
        )
    if FIELDS["parent_title"] in properties and (
        props.get(FIELDS["parent_title"]) or ""
    ) != (properties[FIELDS["parent_title"]] or ""):
        raise ValueError("CMS draft parent title differs from prepared content")
    if FIELDS["related_works"] in properties and book["work_id"] not in relation_ids(
        props.get(FIELDS["related_works"])
    ):
        raise ValueError("CMS extra readback has the wrong work relation")
    item["verified"] = True
    write_json(state, book)


async def upload_draft(book: dict, state: Path, config: dict, *, tools) -> None:
    if (
        book.get("storage") != STORAGE
        or book.get("catalog_id") != config["databases"]["works"]["data_source_id"]
    ):
        raise ValueError(
            "Not a checkpoint for this CMS catalog; old library checkpoints cannot be reused"
        )
    entries = chapter_entries(book)
    reader = NotionBooks(tools)
    if not book.get("uploaded"):
        # Resolve possible shared duplicates before creating a work or uploading rows.
        await preflight_extras(book, state, config, reader)
    if book.get("cover_asset") and not book.get("cover_uploaded"):
        from src.notion.cover import api_token, validate_cover

        api_token()
        validate_cover(Path(book["cover_asset"]).read_bytes())
    await ensure_work(book, state, config, tools=tools)
    await ensure_views(book, state, config, tools=tools)
    if book.get("uploaded"):
        print("Draft already uploaded; later Notion edits are preserved")
        return
    # Existing unknown rows indicate manual edits or an ambiguous earlier create.
    for key, items in [
        ("chapters_view_id", list(book["chapters"].values())),
        ("view_id", book["extras"]),
    ]:
        rows = await reader.rows(book[key])
        known = {i["page_id"] for i in items if i.get("page_id")}
        required = {
            i["page_id"]
            for i in items
            if i.get("page_id") and not (i.get("reused") and not i.get("verified"))
        }
        actual = {r["id"] for r in rows}
        if actual - known or required - actual:
            raise ValueError(
                "CMS rows differ from the import checkpoint; reconcile before appending"
            )
    await reader.ensure_options(
        book["chapters_data_source_id"],
        {FIELDS["parent_title"]: [p for _, p in entries]},
    )
    # New records enter manual views at the top. Create from the end, then verify
    # the actual editorial order instead of assuming query insertion order.
    for member, parent in reversed(entries):
        item = book["chapters"][member]
        await upload_row(
            item,
            book,
            state,
            data_source=book["chapters_data_source_id"],
            properties={
                FIELDS["chapter_title"]: item["title"],
                FIELDS["parent_title"]: parent or None,
            },
            title_property=FIELDS["chapter_title"],
            tools=tools,
        )
    for item in reversed(book["extras"]):
        if item.get("reused") and not item.get("verified"):
            props, body = await reader.document(item["page_id"])
            expected = item.get("reuse_fingerprint") or fingerprint(
                item["title"], item["blocks"]
            )
            if (
                fingerprint(props.get(FIELDS["extra_title"]), from_markdown(body))
                != expected
            ):
                raise ValueError(
                    "Shared extra changed after duplicate review; reconcile before linking"
                )
            works = relation_ids(props.get(FIELDS["related_works"]))
            if book["work_id"] not in works:
                await reader.write_properties(
                    item["page_id"],
                    {FIELDS["related_works"]: works + [book["work_id"]]},
                )
        await upload_row(
            item,
            book,
            state,
            data_source=config["databases"]["extras"]["data_source_id"],
            properties={
                FIELDS["extra_title"]: item["title"],
                FIELDS["related_works"]: [book["work_id"]],
            },
            title_property=FIELDS["extra_title"],
            tools=tools,
        )
    for view, expected in [
        (
            book["chapters_view_id"],
            [book["chapters"][m]["page_id"] for m, _ in entries],
        ),
        (book["view_id"], [i["page_id"] for i in book["extras"]]),
    ]:
        if [r["id"] for r in await reader.rows(view)] != expected:
            raise ValueError(
                "CMS manual order differs from source; reorder the draft view, then resume"
            )
    if book.get("cover_asset"):
        from src.notion.cover import upload_cover

        await upload_cover(book, state, tools=tools)
    book["uploaded"] = True
    write_json(state, book)


def upload_source(
    source: dict,
    *,
    cover_bytes: bytes | None = None,
    config_path: Path = CONFIG,
) -> Path:
    config = json.loads(config_path.read_text())
    catalog = config["databases"]["works"]["data_source_id"]
    key = digest((catalog + source["identifier"]).encode())[:16]
    directory = Path("generated/notion_cms_sources") / key
    state = directory / "import.json"
    with exclusive_lock(directory / "import.lock"):
        fingerprint = source_digest(source)
        if state.exists():
            book = json.loads(state.read_text())
            if book.get("source_sha256") != fingerprint or book.get("cover_sha256") != (
                digest(cover_bytes) if cover_bytes else None
            ):
                raise ValueError(
                    "Crawl changed since this draft import; reconcile it with the saved checkpoint and Notion edits"
                )
        else:
            chapter_entries(source)
            book = copy.deepcopy(source)
            book.update(
                storage=STORAGE,
                catalog_id=catalog,
                source_sha256=fingerprint,
                cover_sha256=digest(cover_bytes) if cover_bytes else None,
            )
            if cover_bytes:
                from src.notion.cover import validate_cover

                suffix = validate_cover(cover_bytes)
                asset = directory / ("cover." + suffix)
                asset.write_bytes(cover_bytes)
                book["cover_asset"] = str(asset)
            write_json(state, book)

        async def online():
            store = TokenStore(AUTH_FILE)
            with store.locked():
                async with connect(store) as tools:
                    await upload_draft(book, state, config, tools=tools)
            print("CMS draft: https://www.notion.so/" + book["work_id"])

        try:
            asyncio.run(online())
        except Exception as error:
            # The MCP transport may wrap domain errors in task groups. Keep the
            # review warning visible to both ingestion CLIs without SDK internals.
            raise ValueError(error_message(error)) from None
    return state
