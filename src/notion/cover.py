"""Upload native CMS covers through the public API; content still uses MCP."""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path

import httpx
from notion_books import API_VERSION, NotionBooks, cover_request, notion_id, page_cover
from PIL import Image

from src.runtime.files import digest, write_json

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000


def validate_cover(data: bytes) -> str:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("CMS cover must be nonempty and at most 10 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG"}:
                    raise ValueError("CMS cover must be PNG or JPEG")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("CMS cover must be at most 25 million pixels")
                suffix = "png" if image.format == "PNG" else "jpg"
                image.verify()
        return suffix
    except (
        OSError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as error:
        raise ValueError("Invalid CMS cover image") from error


def api_token() -> str:
    token = os.environ.get("NOTION_API_TOKEN", "").strip()
    if not token:
        raise ValueError(
            "Automatic cover upload requires NOTION_API_TOKEN in the process environment; .env is not read"
        )
    return token


async def api_json(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> dict:
    try:
        response = await client.request(
            method, "https://api.notion.com/v1/" + path, **kwargs
        )
    except httpx.HTTPError:
        raise ValueError(
            "Notion cover API connection failed; resume the saved checkpoint"
        ) from None
    if not response.is_success:
        raise ValueError(
            f"Notion cover API failed (HTTP {response.status_code}); check token permissions and resume"
        )
    return response.json()


async def upload_cover(book: dict, state: Path, *, tools) -> None:
    if book.get("cover_uploaded"):
        return
    data = Path(book["cover_asset"]).read_bytes()
    if digest(data) != book["cover_sha256"]:
        raise ValueError("Cover asset changed since preparation")
    suffix = validate_cover(data)
    token = api_token()
    reader = NotionBooks(tools)
    page = await reader.fetch(book["work_id"])
    existing_cover = page_cover(page)
    if book.get("cover_pending"):
        # Never overwrite a later manual cover after an uncertain PATCH response.
        raise ValueError(
            "Cover attachment result is uncertain; inspect the page and reconcile cover_pending before retrying"
        )
    if existing_cover is not None:
        raise ValueError(
            "Book already has a cover; reconcile it before uploading the crawler cover"
        )
    async with httpx.AsyncClient(
        headers={
            "Authorization": "Bearer " + token,
            "Notion-Version": API_VERSION,
        },
        timeout=60,
        follow_redirects=False,
    ) as client:
        # Verify API access before allocating or sending a file.
        await api_json(client, "GET", "pages/" + book["work_id"])
        request = cover_request("create", suffix=suffix)
        upload = await api_json(
            client, request["method"], request["path"], json=request["json"]
        )
        id = notion_id(upload["id"])
        request = cover_request("send", id=id, suffix=suffix)
        sent = await api_json(
            client,
            request["method"],
            request["path"],
            files={"file": (request["filename"], data, request["content_type"])},
        )
        if sent.get("status") != "uploaded":
            raise ValueError("Notion did not finish uploading the cover")
        book["cover_pending"] = id
        write_json(state, book)
        request = cover_request("attach", id=id, page=book["work_id"])
        await api_json(client, request["method"], request["path"], json=request["json"])
    cover = page_cover(await reader.fetch(book["work_id"]))
    if cover is None or cover["kind"] != "file":
        raise ValueError("CMS cover readback is not a native uploaded file")
    # Only download the newly attached native file; never persist its signed URL.
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        response = await client.get(cover["url"])
        if not response.is_success or digest(response.content) != book["cover_sha256"]:
            raise ValueError("CMS uploaded cover bytes differ from the prepared cover")
    book.pop("cover_pending")
    book["cover_uploaded"] = True
    write_json(state, book)
