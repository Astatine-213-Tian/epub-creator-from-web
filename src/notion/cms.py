"""The CMS storage contract: catalog metadata and owned chapter databases.

This adapter creates drafts only. Publishing buttons and service-owned properties
are deliberately outside its write interface.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import date
from pathlib import Path

from src.notion.mcp import finish
from src.notion.reader import NotionReader, notion_id, section
from src.runtime.files import write_json

CONFIG = Path("book_specs/notion/config.json")
STORAGE = "cms-chapter-database-v1"
CATALOG_SCHEMA = {
    "works": {
        "作品": "title",
        "作者": "relation",
        "书籍分类": "multi_select",
        "系列": "select",
        "系列序号": "number",
        "语言": "select",
        "简介": "text",
        "来源": "url",
        "出版日期": "date",
    },
    "authors": {"作者": "title"},
    "extras": {"番外": "title", "涉及作品": "relation"},
}


def schema_state(response: dict) -> dict:
    return json.loads(section(response["text"], "data-source-state"))


def require_schema(state: dict, expected: dict) -> None:
    for name, kind in expected.items():
        field = state.get("schema", {}).get(name, {})
        if field.get("type") != kind or field.get("readOnly"):
            raise ValueError(f"CMS property {name} must be editable {kind}")


def parent_id(response: dict, kind: str) -> str:
    match = re.search(rf'<parent-{kind} url="(?:\{{\{{)?([^"}}]+)', response["text"])
    if not match:
        raise ValueError(f"CMS object has no parent {kind}")
    return notion_id(match[1])


def chapter_entries(book: dict) -> list[tuple[str, str]]:
    """A CMS row is a leaf with zero or one parent title; never flatten deeper TOCs."""
    entries = []
    for node in book["sections"]:
        if "children" in node:
            if not node.get("title") or not node["children"]:
                raise ValueError("CMS parent titles must have chapters")
            for child in node["children"]:
                if "children" in child:
                    raise ValueError(
                        "CMS supports one parent title per chapter; use local EPUB for deeper TOCs"
                    )
                entries.append((child["member"], node["title"]))
        else:
            entries.append((node["member"], ""))
    if len({member for member, _ in entries}) != len(entries) or set(
        book["chapters"]
    ) != {m for m, _ in entries}:
        raise ValueError("Chapter outline must contain every chapter exactly once")
    if any(c.get("shared") for c in book["chapters"].values()):
        raise ValueError(
            "Shared extras belong in the shared database, not chapter rows"
        )
    return entries


def work_properties(metadata: dict, authors: list[str]) -> dict:
    props = {
        "作品": metadata["title"],
        "作者": authors,
        "书籍分类": metadata.get("subjects", []),
        "语言": metadata.get("language") or "zh-CN",
    }
    for field, key in [("简介", "description"), ("来源", "source"), ("系列", "series")]:
        if metadata.get(key):
            props[field] = metadata[key]
    position = metadata.get("series_position")
    if position not in (None, ""):
        number = float(position)
        if not math.isfinite(number):
            raise ValueError("Series position must be a finite number")
        props["系列序号"] = number
    if metadata.get("date"):
        value = date.fromisoformat(metadata["date"][:10]).isoformat()
        props.update(
            {
                "date:出版日期:start": value,
                "date:出版日期:end": None,
                "date:出版日期:is_datetime": 0,
            }
        )
    return props


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


async def catalog(reader: NotionReader, config: dict) -> dict:
    states = {}
    for key, expected in CATALOG_SCHEMA.items():
        database = config["databases"][key]
        state = schema_state(
            await reader.fetch("collection://" + database["data_source_id"])
        )
        require_schema(state, expected)
        view = await reader.view(database["view_id"])
        if notion_id(view["dataSourceUrl"]) != database["data_source_id"] or any(
            view.get(k) for k in ("advancedFilter", "filter", "quickFilters")
        ):
            raise ValueError(
                f"CMS {key} inventory view must show its complete data source"
            )
        states[key] = state
    return states


async def ensure_options(
    data_source: str, values: dict[str, list[str]], *, tools
) -> None:
    """Extend editable select fields without dropping existing options or colors."""
    reader = NotionReader(tools)
    state = schema_state(await reader.fetch("collection://" + data_source))
    statements = []
    for name, wanted in values.items():
        field = state["schema"][name]
        if field["type"] not in {"select", "multi_select"}:
            raise ValueError(f"CMS {name} is not a select property")
        options = field.get("options", [])
        existing = {o["name"] for o in options}
        missing = [v for v in dict.fromkeys(wanted) if v and v not in existing]
        if not missing:
            continue
        options = options + [{"name": n, "color": "default"} for n in missing]
        quoted = ", ".join(
            "'" + o["name"].replace("'", "''") + "':" + o.get("color", "default")
            for o in options
        )
        column = '"' + name.replace('"', '""') + '"'
        statements.append(
            f"ALTER COLUMN {column} SET {field['type'].upper()}({quoted})"
        )
    if statements:
        await finish(
            tools,
            await tools.call(
                "notion-update-data-source",
                {
                    "data_source_id": data_source,
                    "statements": "; ".join(statements),
                },
            ),
        )
        actual = schema_state(await reader.fetch("collection://" + data_source))
        for name, wanted in values.items():
            if set(filter(None, wanted)) - {
                o["name"] for o in actual["schema"][name].get("options", [])
            }:
                raise ValueError("CMS select options were not saved")


async def ensure_work(book: dict, path: Path, config: dict, *, tools) -> None:
    reader = NotionReader(tools)
    states = await catalog(reader, config)
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
    if any(row.get("作品") == book["metadata"]["title"] for row in works):
        raise ValueError(
            "A work with this title already exists; reconcile identity and its import checkpoint"
        )
    if book.get("pending_work"):
        raise ValueError("A work create response was lost; reconcile before retrying")
    ids = []
    for name in names:
        found = [row for row in authors if row.get("作者") == name]
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
        response = await finish(
            tools,
            await tools.call(
                "notion-create-pages",
                {
                    "allow_async": False,
                    "parent": {
                        "data_source_id": config["databases"]["authors"][
                            "data_source_id"
                        ]
                    },
                    "pages": [{"properties": {"作者": name}}],
                },
            ),
        )
        ids.append(notion_id(response["pages"][0]["id"]))
        book.pop("pending_author")
        write_json(path, book)
    properties = work_properties(book["metadata"], ids)
    await ensure_options(
        works_ds,
        {
            "书籍分类": properties["书籍分类"],
            "系列": [properties.get("系列", "")],
            "语言": [properties["语言"]],
        },
        tools=tools,
    )
    book["pending_work"] = True
    write_json(path, book)
    response = await finish(
        tools,
        await tools.call(
            "notion-create-pages",
            {
                "allow_async": False,
                "parent": {"data_source_id": works_ds},
                "pages": [
                    {"properties": properties, "template_id": notion_id(template)}
                ],
            },
        ),
    )
    book["work_id"] = notion_id(response["pages"][0]["id"])
    book.pop("pending_work")
    write_json(path, book)


def exact_book_filter(value: dict | None, work: str) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("type") == "group":
        children = value.get("filters", [])
        return (
            value.get("operator") == "and"
            and len(children) == 1
            and exact_book_filter(children[0], work)
        )
    target = value.get("value", {})
    return (
        value.get("type") == "property"
        and value.get("property") == "涉及作品"
        and value.get("propertyType") == "relation"
        and value.get("operator") == "relation_contains"
        and target.get("type") == "exact"
        and notion_id(target["value"]) == work
    )


def manual_view(view: dict) -> None:
    if view.get("type") != "table" or any(
        view.get(k)
        for k in ("sorts", "groupBy", "subGroupBy", "quickFilters", "filter")
    ):
        raise ValueError("CMS editorial view must be an ungrouped manual table")


async def discover(book_id: str, config: dict, *, tools) -> dict:
    reader = NotionReader(tools)
    page = await reader.fetch(book_id)
    if parent_id(page, "data-source") != config["databases"]["works"]["data_source_id"]:
        raise ValueError("Book is outside the configured CMS catalog")
    body = "" if "<blank-page>" in page["text"] else section(page["text"], "content")
    found = {}
    seen = set()
    for url in re.findall(r'<database url="(?:\{\{)?([^"}]+)', body):
        database = await reader.fetch(notion_id(url))
        for view_id in re.findall(
            r'<view url="(?:\{\{)?view://([a-f0-9-]+)', database["text"]
        ):
            if view_id in seen:
                continue
            seen.add(view_id)
            view = await reader.view(view_id)
            ds = notion_id(view["dataSourceUrl"])
            if ds == config["databases"]["extras"]["data_source_id"]:
                if not exact_book_filter(view.get("advancedFilter"), book_id):
                    continue
                manual_view(view)
                if "view_id" in found:
                    raise ValueError("Multiple manual extra views on book")
                found["view_id"] = view_id
            elif view.get("name") == "正文":
                manual_view(view)
                if view.get("advancedFilter"):
                    raise ValueError("The 正文 view must be unfiltered")
                source = await reader.fetch("collection://" + ds)
                schema = schema_state(source)
                require_schema(schema, {"章节": "title", "所属标题": "select"})
                canonical = notion_id(source["url"])
                if (
                    schema.get("name") != "正文"
                    or canonical != notion_id(url)
                    or parent_id(database, "page") != book_id
                ):
                    raise ValueError(
                        "正文 must be the book-owned database, not a linked database"
                    )
                if "chapters_view_id" in found:
                    raise ValueError("Multiple manual chapter views on book")
                found.update(chapters_view_id=view_id, chapters_data_source_id=ds)
    return found


async def ensure_views(book: dict, path: Path, config: dict, *, tools) -> None:
    for _ in range(30):
        found = await discover(book["work_id"], config, tools=tools)
        if "chapters_view_id" in found and "view_id" in found:
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
