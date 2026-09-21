"""Upload native CMS covers through the public API; content still uses MCP."""

from __future__ import annotations

import base64
import io
import os
import warnings
from pathlib import Path

import httpx
from notion_books import API_VERSION, NotionBooks
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


async def api_request(client: httpx.AsyncClient, request: dict) -> dict:
    kwargs = {}
    if "json" in request:
        kwargs["json"] = request["json"]
    if "file" in request:
        file = request["file"]
        kwargs["files"] = {
            "file": (
                file["filename"],
                base64.b64decode(file["data"]),
                file["content_type"],
            )
        }
    response = await client.request(
        request["method"], "https://api.notion.com/v1/" + request["path"], **kwargs
    )
    try:
        body = response.json()
    except ValueError:
        if response.is_success:
            raise
        body = {}
    return {
        "status": response.status_code,
        "body": body,
        "request_id": response.headers.get("x-request-id", ""),
        "retry_after": response.headers.get("retry-after", ""),
    }


async def upload_cover(book: dict, state: Path, *, tools) -> None:
    if book.get("cover_uploaded"):
        return
    data = Path(book["cover_asset"]).read_bytes()
    if digest(data) != book["cover_sha256"]:
        raise ValueError("Cover asset changed since preparation")
    suffix = validate_cover(data)
    token = api_token()
    reader = NotionBooks(tools)
    page = await reader.page(book["work_id"])
    if not page.cover_known:
        raise ValueError("Notion did not provide page cover metadata")
    existing_cover = page.cover
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
        api = NotionBooks(api=lambda request: api_request(client, request))
        await api.check_page_access(book["work_id"])
        id = await api.upload_cover(suffix, data)
        book["cover_pending"] = id
        write_json(state, book)
        await api.attach_cover(book["work_id"], id)
    page = await reader.page(book["work_id"])
    if not page.cover_known:
        raise ValueError("Notion did not provide page cover metadata")
    cover = page.cover
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
