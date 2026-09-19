"""Upload native CMS covers through the public API; content still uses MCP."""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path

import httpx
from PIL import Image

from src.notion.reader import NotionReader, notion_id
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
    reader = NotionReader(tools)
    page = await reader.fetch(book["work_id"])
    if "cover" not in page:
        raise ValueError("CMS cover is unavailable; cannot safely attach a cover")
    if book.get("cover_pending"):
        # Never overwrite a later manual cover after an uncertain PATCH response.
        raise ValueError(
            "Cover attachment result is uncertain; inspect the page and reconcile cover_pending before retrying"
        )
    if page["cover"] is not None:
        raise ValueError(
            "Book already has a cover; reconcile it before uploading the crawler cover"
        )
    async with httpx.AsyncClient(
        headers={
            "Authorization": "Bearer " + token,
            "Notion-Version": "2025-09-03",
        },
        timeout=60,
        follow_redirects=False,
    ) as client:
        # Verify API access before allocating or sending a file.
        await api_json(client, "GET", "pages/" + book["work_id"])
        filename = "cover." + suffix
        mime = "image/png" if suffix == "png" else "image/jpeg"
        upload = await api_json(
            client,
            "POST",
            "file_uploads",
            json={
                "mode": "single_part",
                "filename": filename,
                "content_type": mime,
            },
        )
        id = notion_id(upload["id"])
        sent = await api_json(
            client,
            "POST",
            f"file_uploads/{id}/send",
            files={"file": (filename, data, mime)},
        )
        if sent.get("status") != "uploaded":
            raise ValueError("Notion did not finish uploading the cover")
        book["cover_pending"] = id
        write_json(state, book)
        await api_json(
            client,
            "PATCH",
            "pages/" + book["work_id"],
            json={
                "cover": {"type": "file_upload", "file_upload": {"id": id}},
            },
        )
    cover = (await reader.fetch(book["work_id"])).get("cover")
    if not isinstance(cover, dict) or cover.get("type") != "file":
        raise ValueError("CMS cover readback is not a native uploaded file")
    # Only download the newly attached native file; never persist its signed URL.
    async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
        response = await client.get(cover["file"]["url"])
        if not response.is_success or digest(response.content) != book["cover_sha256"]:
            raise ValueError("CMS uploaded cover bytes differ from the prepared cover")
    book.pop("cover_pending")
    book["cover_uploaded"] = True
    write_json(state, book)
