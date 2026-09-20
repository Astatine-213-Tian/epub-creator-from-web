"""Crawler-owned import decisions and checkpoints using the shared Notion schema."""

from __future__ import annotations

import asyncio
from pathlib import Path

from notion_books import FIELDS, NotionBooks, parent_id, work_properties

from src.runtime.files import write_json

CONFIG = Path("book_specs/notion/config.json")
STORAGE = "cms-chapter-database-v1"


def author_names(metadata: dict) -> list[str]:
    # Never guess separators in a pen name. Multi-author sources can supply creators.
    names = metadata.get("creators") or [metadata["creator"]]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or not n.strip() for n in names)
    ):
        raise ValueError("Source requires at least one named author")
    return list(dict.fromkeys(n.strip() for n in names))


async def ensure_work(book: dict, path: Path, config: dict, *, tools) -> None:
    reader = NotionBooks(tools)
    states = await reader.catalog(config["databases"])
    works_ds = config["databases"]["works"]["data_source_id"]
    if book.get("work_id"):
        if parent_id(await reader.fetch(book["work_id"]), "data-source") != works_ds:
            raise ValueError(
                "Checkpoint work belongs to another catalog; original library is read-only"
            )
        return
    names = author_names(book["metadata"])
    work_properties(book["metadata"], [])  # Validate before any remote writes.
    template = states["works"].get("default_page_template")
    if not template:
        raise ValueError("CMS Works requires its default book template")
    authors = await reader.rows(config["databases"]["authors"]["view_id"])
    works = await reader.rows(config["databases"]["works"]["view_id"])
    # Refuse ambiguous identity rather than creating or overwriting another edition.
    if any(row.get(FIELDS["title"]) == book["metadata"]["title"] for row in works):
        raise ValueError(
            "A work with this title already exists; reconcile identity and its import checkpoint"
        )
    if book.get("pending_work"):
        raise ValueError("A work create response was lost; reconcile before retrying")
    ids = []
    for name in names:
        found = [row for row in authors if row.get(FIELDS["authors"]) == name]
        if len(found) > 1:
            raise ValueError("Multiple authors share this name; resolve identity first")
        if found:
            ids.append(found[0]["id"])
            if book.get("pending_author") == name:
                book.pop("pending_author")
                write_json(path, book)
            continue
        if book.get("pending_author"):
            raise ValueError(
                "An author create response was lost; reconcile before retrying"
            )
        book["pending_author"] = name
        write_json(path, book)
        id = await reader.create_page(
            config["databases"]["authors"]["data_source_id"],
            {FIELDS["authors"]: name},
        )
        ids.append(id)
        book.pop("pending_author")
        write_json(path, book)
    properties = work_properties(book["metadata"], ids)
    await reader.ensure_options(
        works_ds,
        {
            FIELDS["subjects"]: properties[FIELDS["subjects"]],
            FIELDS["series"]: [properties.get(FIELDS["series"], "")],
            FIELDS["language"]: [properties[FIELDS["language"]]],
        },
    )
    book["pending_work"] = True
    write_json(path, book)
    book["work_id"] = await reader.create_page(
        works_ds,
        properties,
        template=template,
    )
    book.pop("pending_work")
    write_json(path, book)


async def ensure_views(book: dict, path: Path, config: dict, *, tools) -> None:
    for _ in range(30):
        source = await NotionBooks(tools).discover(
            book["work_id"],
            [config["databases"]["works"]["data_source_id"]],
            config["databases"]["extras"]["data_source_id"],
        )
        found = {
            "chapters_view_id": source["chapters_view"],
            "chapters_data_source_id": source["chapters_data_source"],
            "view_id": source["extras_view"],
        }
        if found["chapters_view_id"] and found["view_id"]:
            for key, value in found.items():
                if book.get(key) and book[key] != value:
                    raise ValueError(
                        "Book databases changed; reconcile the import checkpoint"
                    )
            book.update(found)
            write_json(path, book)
            return
        await asyncio.sleep(2)
    raise ValueError("Book template is not ready; resume this checkpoint later")
