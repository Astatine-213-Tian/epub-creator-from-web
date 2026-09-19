from __future__ import annotations

import json
import re
from uuid import UUID

from src.notion.mcp import MCPTools


def notion_id(value: str) -> str:
    value = value.strip("{}")
    match = re.search(
        r"([0-9a-fA-F]{32}|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})(?:[/?#]|$)",
        value,
    )
    if not match:
        raise ValueError("Invalid Notion page or view reference")
    return str(UUID(match[1]))


def section(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}(?:\s[^>]*)?>\n(.*?)\n</{tag}>", text, re.S)
    if not match:
        raise ValueError(f"Notion response is missing {tag}")
    return match[1]


def relation_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError("Incomplete Notion relation") from None
    if not isinstance(value, list):
        raise ValueError("Missing or incomplete Notion relation")
    ids = [notion_id(item) for item in value]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate Notion relation")
    return ids


class NotionReader:
    def __init__(self, tools: MCPTools) -> None:
        self.tools = tools

    async def fetch(self, id: str) -> dict:
        return await self.tools.call("notion-fetch", {"id": id})

    async def view(self, id: str) -> dict:
        response = await self.fetch(f"view://{id}")
        return json.loads(section(response["text"], "view"))

    async def rows(self, view: str) -> list[dict]:
        args = {"mode": "view", "view_url": f"view://{view}", "page_size": 100}
        rows, seen, cursors = [], set(), set()
        while True:
            response = await self.tools.call(
                "notion-query-data-sources", {"data": args}
            )
            if (
                response.get("truncated")
                or response.get("request_status", {}).get("type") == "incomplete"
            ):
                raise ValueError("Incomplete Notion view query")
            for row in response["results"]:
                id = notion_id(row["url"])
                if id in seen:
                    raise ValueError("Duplicate Notion page in view query")
                seen.add(id)
                rows.append({**row, "id": id})
            if response.get("has_more") is False:
                return rows
            cursor = response.get("next_cursor")
            if not cursor or cursor in cursors:
                raise ValueError("Incomplete Notion view pagination")
            cursors.add(cursor)
            args = {**args, "start_cursor": cursor}

    async def document(
        self, id: str, ancestors: frozenset[str] = frozenset()
    ) -> tuple[dict, str]:
        if id in ancestors or len(ancestors) >= 20:
            raise ValueError("Cyclic or excessively nested Notion content")
        response = await self.fetch(id)
        text = response["text"]
        properties = (
            json.loads(section(text, "properties")) if "<properties>" in text else {}
        )
        body = "" if "<blank-page>" in text else section(text, "content")
        unknown = re.findall(
            r'<unknown\s[^>]*url="(?:\{\{)?([^"}]+)(?:\}\})?"[^>]*/>', body
        )
        missing = response.get("unknown_block_ids", [])
        count = response.get("unknown_block_count", len(missing))
        if not {notion_id(item) for item in missing}.issubset(
            {notion_id(item) for item in unknown}
        ):
            raise ValueError(f"Notion page {id} has missing content without a position")
        if (response.get("truncated") or missing or count) and (
            not unknown or count > len(unknown)
        ):
            raise ValueError(
                f"Notion page {id} is incomplete and cannot be reconstructed"
            )
        for url in unknown:
            child = notion_id(url)
            _, replacement = await self.document(child, ancestors | {id})
            pattern = (
                r'<unknown\s[^>]*url="(?:\{\{)?' + re.escape(url) + r'(?:\}\})?"[^>]*/>'
            )
            body = re.sub(pattern, lambda _, value=replacement: value, body)
        if response.get("in_trash") or response.get("archived"):
            raise ValueError(f"Notion page {id} is archived")
        return properties, body
