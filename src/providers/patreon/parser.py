from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape as escape_html
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

import zendriver as zd
import requests
from bs4 import BeautifulSoup

from src.crawl.snapshot import CRAWL_SCHEMA_VERSION, write_json
from src.core.epub_writer import is_scene_break_text, render_scene_break, write_epub
from src.core.models import Chapter, Volume
from src.fetch.browser import resolve_browser_executable
from src.fetch.parallel import crawl_items

HOST = "https://www.patreon.com"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
DEFAULT_AUTHOR = "顾雪柔"
DEFAULT_TITLE = "永恒之门"
PROFILE_DIR_ENV = "BOOKLIB_PATREON_PROFILE_DIR"
DEFAULT_PROFILE_DIR = (
    Path.home() / ".local" / "share" / "epub-creator-from-web" / "patreon-profile"
)
COLLECTION_RE = re.compile(r"/collection/(\d+)")
SPACE_RE = re.compile(r"\s+")
CHAPTER_PREFIX_RE = re.compile(r"^\s*(chapter\s+\d+)\s*[—–-]\s*", re.IGNORECASE)


class PatreonAuthError(RuntimeError):
    """Raised when Patreon returns teaser-only data for member-only posts."""


@dataclass
class BookMeta:
    collection_id: str
    title: str
    author: str
    intro_paragraphs: list[str]
    volume_title: str = ""
    cover_url: str | None = None
    cover_bytes: bytes | None = None
    cover_mime: str = "image/jpeg"


@dataclass(frozen=True)
class PostRef:
    post_id: str
    title: str
    url: str
    published_at: str = ""
    order: int = 0
    current_user_can_view: bool | None = None


class PatreonFetcher:
    def __init__(
        self,
        *,
        headless: bool = False,
        delay: float = 0.4,
        profile_dir: Path | None = None,
    ) -> None:
        self.headless = headless
        self.delay = delay
        self.profile_dir = profile_dir or _profile_dir()
        self.browser: zd.Browser | None = None
        self._page: zd.Tab | None = None
        self._request_lock = asyncio.Lock()

    async def start(self) -> None:
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.profile_dir.chmod(0o700)
        except OSError as exc:
            logging.debug("could not chmod Patreon profile dir: %s", exc)

        config = zd.Config(
            user_data_dir=self.profile_dir,
            headless=self.headless,
            browser_executable_path=resolve_browser_executable(),
            sandbox=False,
            browser_connection_timeout=1.0,
            browser_connection_max_tries=30,
        )
        self.browser = await zd.start(config)

    async def stop(self) -> None:
        if self.browser:
            await self.browser.stop()

    async def open_collection(self, collection_url: str) -> None:
        assert self.browser is not None
        self._page = await self.browser.get(collection_url)
        await _wait_for_document(self._page)
        await asyncio.sleep(self.delay)

    async def get_json(self, url: str) -> dict[str, Any]:
        text = await self._fetch_text(
            url,
            headers={
                "accept": "application/json, text/plain, */*",
                "x-requested-with": "XMLHttpRequest",
            },
        )
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            sample = SPACE_RE.sub(" ", text[:240]).strip()
            raise RuntimeError(f"Patreon API returned non-JSON for {url}: {sample}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Patreon API returned unexpected JSON for {url}")
        return data

    async def get_bytes(self, url: str) -> tuple[bytes, str]:
        assert self.browser is not None
        page = await self._request_page()
        async with self._request_lock:
            payload = await page.evaluate(
                f"""
                (async () => {{
                  const r = await fetch({url!r}, {{credentials: 'include'}});
                  const buf = new Uint8Array(await r.arrayBuffer());
                  let s = '';
                  for (const b of buf) s += String.fromCharCode(b);
                  return {{
                    ok: r.ok,
                    status: r.status,
                    statusText: r.statusText,
                    contentType: r.headers.get('content-type') || '',
                    body: btoa(s)
                  }};
                }})()
                """,
                await_promise=True,
            )
        await asyncio.sleep(self.delay)

        if not payload.get("ok"):
            status = payload.get("status")
            status_text = payload.get("statusText") or ""
            raise RuntimeError(f"Patreon fetch failed with HTTP {status} {status_text}: {url}")
        return base64.b64decode(payload.get("body") or ""), str(
            payload.get("contentType") or ""
        )

    async def _fetch_text(self, url: str, *, headers: dict[str, str]) -> str:
        page = await self._request_page()
        header_lines = ", ".join(f"{key!r}: {value!r}" for key, value in headers.items())
        async with self._request_lock:
            payload = await page.evaluate(
                f"""
                (async () => {{
                  const r = await fetch({url!r}, {{
                    credentials: 'include',
                    headers: {{{header_lines}}}
                  }});
                  const text = await r.text();
                  return {{
                    ok: r.ok,
                    status: r.status,
                    statusText: r.statusText,
                    url: r.url,
                    body: text
                  }};
                }})()
                """,
                await_promise=True,
            )
        await asyncio.sleep(self.delay)

        if not payload.get("ok"):
            status = payload.get("status")
            status_text = payload.get("statusText") or ""
            body = SPACE_RE.sub(" ", str(payload.get("body") or "")[:240]).strip()
            raise RuntimeError(
                f"Patreon API failed with HTTP {status} {status_text}: {url}. {body}"
            )
        return str(payload.get("body") or "")

    async def _request_page(self) -> zd.Tab:
        assert self.browser is not None
        if self._page is None:
            self._page = await self.browser.get(HOST + "/")
            await _wait_for_document(self._page)
        return self._page


def _profile_dir() -> Path:
    raw = os.environ.get(PROFILE_DIR_ENV)
    return Path(raw).expanduser() if raw else DEFAULT_PROFILE_DIR


async def _wait_for_document(page: zd.Tab) -> None:
    for _ in range(30):
        try:
            ready_state = await page.evaluate("document.readyState")
        except Exception as exc:  # noqa: BLE001
            logging.debug("document readiness check failed: %s", exc)
            ready_state = None
        if ready_state in {"interactive", "complete"}:
            return
        await asyncio.sleep(0.5)


def _collection_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    match = COLLECTION_RE.search(parsed.path)
    if match:
        return match.group(1)
    if url.isdigit():
        return url
    raise ValueError(f"Patreon collection URL must contain /collection/<id>: {url}")


def _api_collection_url(collection_id: str, *, include_posts: bool = True) -> str:
    suffix = "?include=posts" if include_posts else ""
    return f"{HOST}/api/collection/{collection_id}{suffix}"


def _api_post_url(post_id: str) -> str:
    return (
        f"{HOST}/api/posts/{post_id}"
        "?include=campaign,user,attachments,images,post_file,post_files"
    )


def _api_post_comments_url(post_id: str) -> str:
    return f"{HOST}/api/posts/{post_id}/comments?{urlencode({'include': 'commenter,replies'})}"


def _api_campaign_posts_url(campaign_id: str, *, count: int = 20) -> str:
    query = urlencode(
        {
            "include": "user,campaign,attachments,post_file,post_files,images",
            "sort": "-published_at",
            "page[count]": str(count),
        }
    )
    return f"{HOST}/api/campaigns/{campaign_id}/posts?{query}"


def _normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def _clean_title(title: str, *, fallback: str) -> str:
    value = _normalize_text(BeautifulSoup(title or "", "html.parser").get_text(" "))
    return value or fallback


def _parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _is_published(attrs: dict[str, Any]) -> bool:
    published_at = str(attrs.get("published_at") or "")
    published = _parse_iso_datetime(published_at)
    if published is None:
        return False
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return published <= datetime.now(timezone.utc)


def _included_by_type(data: dict[str, Any], type_name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in data.get("included") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != type_name:
            continue
        item_id = str(item.get("id") or "")
        if item_id:
            output[item_id] = item
    return output


def _relationship_ids(item: dict[str, Any], name: str) -> list[str]:
    relationship = ((item.get("relationships") or {}).get(name) or {}).get("data")
    if isinstance(relationship, list):
        return [str(ref.get("id")) for ref in relationship if isinstance(ref, dict) and ref.get("id")]
    if isinstance(relationship, dict) and relationship.get("id"):
        return [str(relationship["id"])]
    return []


def _parse_collection(data: dict[str, Any], collection_id: str) -> tuple[BookMeta, list[PostRef]]:
    collection_item = data.get("data") or {}
    if not isinstance(collection_item, dict):
        raise RuntimeError("Patreon collection API returned no collection data")
    attrs = collection_item.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}

    description = _html_to_text(attrs.get("description") or "")
    collection_title = _normalize_text(str(attrs.get("title") or ""))
    intro = [f"Patreon collection: {collection_title}"] if collection_title else []
    intro.extend(description)

    cover_url = _best_image_url(attrs.get("thumbnail") or attrs.get("image"))
    meta = BookMeta(
        collection_id=collection_id,
        title=DEFAULT_TITLE,
        author=DEFAULT_AUTHOR,
        intro_paragraphs=intro,
        volume_title=description[0] if description else collection_title,
        cover_url=cover_url,
        cover_mime=_mime_from_url(cover_url),
    )

    included_posts = _included_by_type(data, "post")
    ids = _relationship_ids(collection_item, "posts")
    if not ids:
        ids = [str(post_id) for post_id in attrs.get("post_ids") or []]
    if not ids:
        ids = list(included_posts)

    refs: list[PostRef] = []
    for index, post_id in enumerate(ids):
        post = included_posts.get(str(post_id), {})
        post_attrs = post.get("attributes") or {}
        if not isinstance(post_attrs, dict):
            post_attrs = {}
        if post_attrs and not _is_published(post_attrs):
            continue
        title = _clean_title(str(post_attrs.get("title") or ""), fallback=f"Chapter {index + 1:03d}")
        url = _post_url(post_attrs, f"{HOST}/posts/{post_id}")
        refs.append(
            PostRef(
                post_id=str(post_id),
                title=title,
                url=url,
                published_at=str(post_attrs.get("published_at") or ""),
                order=index,
                current_user_can_view=post_attrs.get("current_user_can_view"),
            )
        )
    refs.sort(key=_post_ref_sort_key)
    return meta, refs


def _campaign_id_from_collection_data(data: dict[str, Any]) -> str | None:
    for item in data.get("included") or []:
        if item.get("type") != "post":
            continue
        campaign = ((item.get("relationships") or {}).get("campaign") or {}).get("data") or {}
        campaign_id = campaign.get("id")
        if campaign_id:
            return str(campaign_id)
    for item in data.get("included") or []:
        if item.get("type") == "campaign" and item.get("id"):
            return str(item["id"])
    return None


def _chapter_number_from_title(title: str) -> int | None:
    match = re.match(r"^\s*chapter\s+(\d+)\b", title, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _source_chapter_label(number: int | None) -> str | None:
    if number is None:
        return None
    return f"Chapter {number:03d}"


def _source_chapter_file_stem(number: int | None) -> str | None:
    if number is None:
        return None
    return f"chapter-{number:03d}"


def _clear_json_files(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.glob("*.json"):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _post_ref_from_api_post(post: dict[str, Any], *, order: int) -> PostRef | None:
    post_id = str(post.get("id") or "")
    attrs = post.get("attributes") or {}
    if not post_id or not isinstance(attrs, dict) or not _is_published(attrs):
        return None
    if attrs.get("current_user_can_view") is False:
        return None
    title = _clean_title(str(attrs.get("title") or ""), fallback=f"Chapter {order + 1:03d}")
    return PostRef(
        post_id=post_id,
        title=title,
        url=_post_url(attrs, f"{HOST}/posts/{post_id}"),
        published_at=str(attrs.get("published_at") or ""),
        order=order,
        current_user_can_view=attrs.get("current_user_can_view"),
    )


async def _supplement_refs_from_campaign_feed(
    fetcher: PatreonFetcher,
    collection_data: dict[str, Any],
    refs: list[PostRef],
) -> tuple[list[PostRef], dict[str, Any] | None]:
    campaign_id = _campaign_id_from_collection_data(collection_data)
    if not campaign_id:
        return refs, None

    existing_ids = {ref.post_id for ref in refs}
    existing_numbers = [
        number
        for ref in refs
        if (number := _chapter_number_from_title(ref.title)) is not None
    ]
    if not existing_numbers:
        return refs, None

    latest_published = max(
        (
            published
            for ref in refs
            if (published := _parse_iso_datetime(ref.published_at)) is not None
        ),
        default=None,
    )
    if latest_published is not None and latest_published.tzinfo is None:
        latest_published = latest_published.replace(tzinfo=timezone.utc)
    max_existing_number = max(existing_numbers)

    feed_data = await fetcher.get_json(_api_campaign_posts_url(campaign_id))
    feed_posts = feed_data.get("data") or []
    if isinstance(feed_posts, dict):
        feed_posts = [feed_posts]

    supplements: list[PostRef] = []
    for index, post in enumerate(feed_posts, len(refs)):
        if not isinstance(post, dict) or str(post.get("id") or "") in existing_ids:
            continue
        ref = _post_ref_from_api_post(post, order=index)
        if ref is None:
            continue
        number = _chapter_number_from_title(ref.title)
        if number is None or number <= max_existing_number:
            continue
        published = _parse_iso_datetime(ref.published_at)
        if published is not None:
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if latest_published is not None and published <= latest_published:
                continue
        supplements.append(ref)

    if not supplements:
        return refs, feed_data

    merged = [*refs, *supplements]
    merged.sort(key=_post_ref_sort_key)
    added = ", ".join(f"{ref.post_id} {ref.title}" for ref in supplements)
    print(
        f"[+] supplemented {len(supplements)} newer campaign feed post(s): {added}",
        file=sys.stderr,
    )
    return merged, feed_data


def _post_ref_sort_key(ref: PostRef) -> tuple[float, int]:
    published = _parse_iso_datetime(ref.published_at)
    if published is None:
        return (float("inf"), ref.order)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return (published.timestamp(), ref.order)


def _best_image_url(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    for key in ("original", "large", "url", "thumbnail_url"):
        nested = value.get(key)
        if isinstance(nested, str) and nested:
            return nested
        if isinstance(nested, dict):
            nested_url = _best_image_url(nested)
            if nested_url:
                return nested_url
    return None


def _mime_from_url(url: str | None) -> str:
    if not url:
        return "image/jpeg"
    clean_url = urlparse(url).path
    return mimetypes.guess_type(clean_url)[0] or "image/jpeg"


def _html_to_text(html: str) -> list[str]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs: list[str] = []
    block_tags = soup.find_all(["p", "li", "blockquote", "h1", "h2", "h3", "h4"])
    if block_tags:
        for tag in block_tags:
            text = _normalize_text(tag.get_text(" ", strip=True))
            if text:
                paragraphs.append(text)
    else:
        for text in soup.get_text("\n").splitlines():
            cleaned = _normalize_text(text)
            if cleaned:
                paragraphs.append(cleaned)
    return paragraphs


def _extract_json_paragraphs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return _extract_json_paragraphs(json.loads(value))
        except json.JSONDecodeError:
            return [_normalize_text(value)]
    if isinstance(value, list):
        paragraphs: list[str] = []
        for item in value:
            paragraphs.extend(_extract_json_paragraphs(item))
        return paragraphs
    if not isinstance(value, dict):
        return []

    node_type = str(value.get("type") or "")
    content = value.get("content")
    if node_type in {"paragraph", "heading", "blockquote", "list_item"}:
        text = _text_from_json_node(value)
        return [text] if text else []

    paragraphs: list[str] = []
    if isinstance(content, list):
        for child in content:
            paragraphs.extend(_extract_json_paragraphs(child))
    return paragraphs


def _extract_json_html_blocks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            return _extract_json_html_blocks(json.loads(value))
        except json.JSONDecodeError:
            text = _normalize_text(value)
            if is_scene_break_text(text):
                return [render_scene_break()]
            return [f"<p>{escape_html(text)}</p>"]
    if isinstance(value, list):
        blocks: list[str] = []
        for item in value:
            blocks.extend(_extract_json_html_blocks(item))
        return blocks
    if not isinstance(value, dict):
        return []

    node_type = str(value.get("type") or "")
    content = value.get("content")
    if node_type == "doc":
        return _extract_json_html_blocks(content)
    if node_type == "paragraph":
        text = _text_from_json_node(value)
        if is_scene_break_text(text):
            return [render_scene_break()]
        inline = _inline_html(content)
        return [f"<p>{inline}</p>"] if inline else []
    if node_type == "heading":
        level = int((value.get("attrs") or {}).get("level") or 3)
        level = min(max(level, 1), 4)
        inline = _inline_html(content)
        return [f"<h{level}>{inline}</h{level}>"] if inline else []
    if node_type == "blockquote":
        inner = "\n".join(_extract_json_html_blocks(content))
        return [f"<blockquote>{inner}</blockquote>"] if inner else []
    if node_type in {"bulletList", "orderedList"}:
        tag = "ul" if node_type == "bulletList" else "ol"
        items: list[str] = []
        for child in content or []:
            item_html = "".join(_extract_json_html_blocks((child or {}).get("content")))
            if item_html:
                items.append(f"<li>{item_html}</li>")
        return [f"<{tag}>{''.join(items)}</{tag}>"] if items else []
    if node_type == "horizontalRule":
        return ["<hr/>"]
    if node_type == "hard_break":
        return ["<br/>"]
    return _extract_json_html_blocks(content)


def _inline_html(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragment = _inline_html(item)
            if not fragment:
                continue
            if fragments and _needs_inline_space(fragments[-1], fragment):
                fragments.append(" ")
            fragments.append(fragment)
        return "".join(fragments)
    if not isinstance(value, dict):
        return ""

    node_type = str(value.get("type") or "")
    if node_type == "text":
        text = escape_html(str(value.get("text") or ""))
        return _apply_text_marks(text, value.get("marks") or [])
    if node_type == "hard_break":
        return "<br/>"
    return _inline_html(value.get("content"))


def _visible_text_fragment(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text()


def _needs_inline_space(left: str, right: str) -> bool:
    left_text = _visible_text_fragment(left)
    right_text = _visible_text_fragment(right)
    if not left_text or not right_text:
        return False
    if left_text[-1].isspace() or right_text[0].isspace():
        return False
    return bool(re.match(r"[\w“‘\"']", right_text[0], flags=re.UNICODE))


def _apply_text_marks(text: str, marks: list[Any]) -> str:
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        mark_type = str(mark.get("type") or "")
        attrs = mark.get("attrs") or {}
        if mark_type == "italic":
            text = f"<em>{text}</em>"
        elif mark_type == "bold":
            text = f"<strong>{text}</strong>"
        elif mark_type == "code":
            text = f"<code>{text}</code>"
        elif mark_type == "link":
            href = escape_html(str(attrs.get("href") or ""), quote=True)
            if href:
                text = f'<a href="{href}">{text}</a>'
    return text


def _text_from_json_node(node: dict[str, Any]) -> str:
    parts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
            return
        if not isinstance(value, dict):
            return
        node_type = str(value.get("type") or "")
        if node_type == "text" and isinstance(value.get("text"), str):
            parts.append(value["text"])
        elif node_type == "hard_break":
            parts.append("\n")
        else:
            walk(value.get("content"))

    walk(node.get("content"))
    return _normalize_text("".join(parts))


def _content_candidates(attrs: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for key in (
        "content",
        "content_html",
        "body",
        "body_html",
        "post_content",
        "post_content_html",
    ):
        value = attrs.get(key)
        if value:
            candidates.append(value)
    for key in ("content_json", "content_json_string", "body_json", "body_json_string"):
        value = attrs.get(key)
        if value:
            candidates.append(value)
    metadata = attrs.get("post_metadata")
    if isinstance(metadata, dict):
        for key in ("content", "content_json", "content_json_string"):
            value = metadata.get(key)
            if value:
                candidates.append(value)
    return candidates


def _html_block_candidates(attrs: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for key in ("content_json", "content_json_string", "body_json", "body_json_string"):
        value = attrs.get(key)
        if value:
            candidates.append(value)
    metadata = attrs.get("post_metadata")
    if isinstance(metadata, dict):
        for key in ("content_json", "content_json_string"):
            value = metadata.get(key)
            if value:
                candidates.append(value)
    return candidates


def _extract_post_html_blocks(attrs: dict[str, Any]) -> list[str]:
    for candidate in _html_block_candidates(attrs):
        blocks = _extract_json_html_blocks(candidate)
        if blocks:
            return blocks
    return []


def _extract_post_paragraphs(attrs: dict[str, Any]) -> list[str]:
    for candidate in _content_candidates(attrs):
        if isinstance(candidate, str):
            if "<" in candidate and ">" in candidate:
                paragraphs = _html_to_text(candidate)
            else:
                paragraphs = _extract_json_paragraphs(candidate)
        elif isinstance(candidate, (dict, list)):
            paragraphs = _extract_json_paragraphs(candidate)
        else:
            paragraphs = []
        cleaned = [_normalize_text(item) for item in paragraphs if _normalize_text(item)]
        if cleaned:
            return cleaned
    return []


def _post_attrs(data: dict[str, Any]) -> dict[str, Any]:
    item = data.get("data") or {}
    if not isinstance(item, dict):
        return {}
    attrs = item.get("attributes") or {}
    return attrs if isinstance(attrs, dict) else {}


def _post_url(attrs: dict[str, Any], fallback: str) -> str:
    url = str(attrs.get("url") or fallback)
    if url.startswith("/"):
        return urljoin(HOST, url)
    return url


def _user_display_name(user: dict[str, Any] | None, fallback: str = "") -> str:
    attrs = (user or {}).get("attributes") or {}
    return (
        _normalize_text(str(attrs.get("full_name") or ""))
        or _normalize_text(str(attrs.get("vanity") or ""))
        or _normalize_text(str(attrs.get("first_name") or ""))
        or fallback
    )


def _flatten_comment(
    comment: dict[str, Any],
    users: dict[str, dict[str, Any]],
    *,
    parent_id: str | None = None,
) -> dict[str, Any]:
    attrs = comment.get("attributes") or {}
    commenter = (((comment.get("relationships") or {}).get("commenter") or {}).get("data") or {})
    user_id = str(commenter.get("id") or "")
    return {
        "id": str(comment.get("id") or ""),
        "parent_id": parent_id,
        "author": _user_display_name(users.get(user_id), fallback=user_id) or "unknown",
        "body": attrs.get("body") or "",
        "is_by_creator": bool(attrs.get("is_by_creator")),
        "is_by_patron": bool(attrs.get("is_by_patron")),
        "is_liked_by_creator": bool(attrs.get("is_liked_by_creator")),
        "created": attrs.get("created") or "",
    }


async def fetch_post_comments_authenticated(
    fetcher: PatreonFetcher,
    post_id: str,
) -> list[dict[str, Any]]:
    url: str | None = _api_post_comments_url(post_id)
    comments: list[dict[str, Any]] = []
    while url:
        data = await fetcher.get_json(url)
        users = {
            str(item.get("id")): item
            for item in data.get("included", [])
            if item.get("type") == "user"
        }
        included_comments = {
            str(item.get("id")): item
            for item in data.get("included", [])
            if item.get("type") == "comment"
        }
        for item in data.get("data") or []:
            if item.get("type") != "comment":
                continue
            parent = _flatten_comment(item, users)
            comments.append(parent)
            reply_refs = (((item.get("relationships") or {}).get("replies") or {}).get("data") or [])
            for reply_ref in reply_refs:
                reply = included_comments.get(str(reply_ref.get("id")))
                if reply is not None:
                    comments.append(_flatten_comment(reply, users, parent_id=parent["id"]))
        url = (data.get("links") or {}).get("next")
    return comments


def _chapter_from_post(post_ref: PostRef, data: dict[str, Any]) -> Chapter:
    attrs = _post_attrs(data)
    title = _clean_title(str(attrs.get("title") or post_ref.title), fallback=post_ref.title)
    paragraphs = _extract_post_paragraphs(attrs)
    html_blocks = _extract_post_html_blocks(attrs)
    if not paragraphs:
        viewable = attrs.get("current_user_can_view")
        teaser = attrs.get("teaser_text") or attrs.get("teaser_text_json_string")
        if viewable is False or teaser:
            raise PatreonAuthError(_auth_message(post_ref))
        keys = ", ".join(sorted(attrs.keys()))
        raise RuntimeError(f"could not find body content for Patreon post {post_ref.post_id}: {keys}")

    if paragraphs and _normalize_title_like(paragraphs[0]) == _normalize_title_like(title):
        paragraphs = paragraphs[1:]
        if html_blocks:
            html_blocks = html_blocks[1:]
    return Chapter(title=title, paragraphs=paragraphs, html_blocks=html_blocks or None)


def _normalize_title_like(value: str) -> str:
    value = CHAPTER_PREFIX_RE.sub("", value)
    return SPACE_RE.sub("", value).lower()


def _auth_message(post_ref: PostRef) -> str:
    return (
        "Patreon returned teaser-only content for "
        f"{post_ref.title} ({post_ref.url}). "
        f"Log in once with the browser profile at {_profile_dir()} and rerun the command. "
        f"Override the profile path with {PROFILE_DIR_ENV} if needed."
    )


async def _fetch_post(fetcher: PatreonFetcher, post_ref: PostRef) -> Chapter:
    data = await fetcher.get_json(_api_post_url(post_ref.post_id))
    return _chapter_from_post(post_ref, data)


async def _fetch_post_snapshot(
    fetcher: PatreonFetcher,
    post_ref: PostRef,
) -> tuple[Chapter, dict[str, Any]]:
    data = await fetcher.get_json(_api_post_url(post_ref.post_id))
    return _chapter_from_post(post_ref, data), data


async def _download_cover(fetcher: PatreonFetcher, meta: BookMeta) -> None:
    if not meta.cover_url:
        return
    try:
        data, content_type = await asyncio.to_thread(_fetch_cover_direct, meta.cover_url)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] cover download failed: {exc}", file=sys.stderr)
        return
    if data:
        meta.cover_bytes = data
        if content_type:
            meta.cover_mime = content_type.split(";", 1)[0].strip() or meta.cover_mime


def _fetch_cover_direct(url: str) -> tuple[bytes, str]:
    response = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    response.raise_for_status()
    return response.content, response.headers.get("content-type", "")


async def _crawl_posts_with_login_retry(
    fetcher: PatreonFetcher,
    refs: list[PostRef],
    *,
    concurrency: int,
    headless: bool,
) -> list[Chapter]:
    try:
        return await crawl_items(
            refs,
            lambda ref: _fetch_post(fetcher, ref),
            concurrency=concurrency,
            item_name="post",
        )
    except Exception as exc:
        auth_error = _find_auth_error(exc)
        if auth_error is None or headless or not sys.stdin.isatty():
            raise

        print(f"[!] {auth_error}", file=sys.stderr)
        print(
            "[!] Finish logging in to Patreon in the opened browser window; "
            "the session will be saved in this provider profile.",
            file=sys.stderr,
        )
        await asyncio.to_thread(input, "Press Enter after Patreon login is complete...")
        return await crawl_items(
            refs,
            lambda ref: _fetch_post(fetcher, ref),
            concurrency=concurrency,
            item_name="post",
        )


async def _crawl_post_snapshots_with_login_retry(
    fetcher: PatreonFetcher,
    refs: list[PostRef],
    *,
    concurrency: int,
    headless: bool,
) -> list[tuple[Chapter, dict[str, Any]]]:
    try:
        return await crawl_items(
            refs,
            lambda ref: _fetch_post_snapshot(fetcher, ref),
            concurrency=concurrency,
            item_name="post",
        )
    except Exception as exc:
        auth_error = _find_auth_error(exc)
        if auth_error is None or headless or not sys.stdin.isatty():
            raise

        print(f"[!] {auth_error}", file=sys.stderr)
        print(
            "[!] Finish logging in to Patreon in the opened browser window; "
            "the session will be saved in this provider profile.",
            file=sys.stderr,
        )
        await asyncio.to_thread(input, "Press Enter after Patreon login is complete...")
        return await crawl_items(
            refs,
            lambda ref: _fetch_post_snapshot(fetcher, ref),
            concurrency=concurrency,
            item_name="post",
        )


async def _safe_fetch_comments(fetcher: PatreonFetcher, ref: PostRef) -> list[dict[str, Any]]:
    try:
        return await fetch_post_comments_authenticated(fetcher, ref.post_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] comments fetch failed for {ref.title}: {exc}", file=sys.stderr)
        return []


def _find_auth_error(exc: BaseException) -> PatreonAuthError | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, PatreonAuthError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _snapshot_blocks(chapter: Chapter) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    html_blocks = chapter.html_blocks or [f"<p>{escape_html(text)}</p>" for text in chapter.paragraphs]
    blocks: list[dict[str, Any]] = []
    paragraphs: list[dict[str, Any]] = []
    paragraph_index = 0
    for html in html_blocks:
        text = _normalize_text(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
        if is_scene_break_text(text) or html.lstrip().startswith("<hr"):
            blocks.append({"type": "separator", "html": html})
            continue
        if not text:
            blocks.append({"type": "html", "html": html})
            continue
        paragraph = {"index": paragraph_index, "english": text}
        paragraphs.append(paragraph)
        blocks.append(
            {
                "type": "content",
                "index": paragraph_index,
                "english": text,
                "html": html,
            }
        )
        paragraph_index += 1
    return blocks, paragraphs


async def crawl_snapshot(
    book_url: str,
    output_dir: Path,
    *,
    title: str | None = None,
    author: str | None = None,
    headless: bool = False,
    delay: float = 0.4,
    concurrency: int = 2,
    include_comments: bool = True,
) -> Path:
    collection_id = _collection_id_from_url(book_url)
    collection_url = _resolve_book_url(book_url)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chapters").mkdir(exist_ok=True)
    (output_dir / "chapters" / "by-source").mkdir(exist_ok=True)
    (output_dir / "comments").mkdir(exist_ok=True)
    (output_dir / "comments" / "by-source").mkdir(exist_ok=True)
    (output_dir / "raw_posts").mkdir(exist_ok=True)
    (output_dir / "assets").mkdir(exist_ok=True)
    _clear_json_files(output_dir / "chapters" / "by-source")
    _clear_json_files(output_dir / "comments" / "by-source")

    fetcher = PatreonFetcher(headless=headless, delay=delay)
    await fetcher.start()
    try:
        await fetcher.open_collection(collection_url)
        collection_data = await fetcher.get_json(_api_collection_url(collection_id))
        meta, refs = _parse_collection(collection_data, collection_id)
        refs, campaign_feed_data = await _supplement_refs_from_campaign_feed(
            fetcher,
            collection_data,
            refs,
        )
        if title:
            meta.title = title
        if author:
            meta.author = author
        if not refs:
            raise RuntimeError(f"Patreon collection {collection_id} has no published posts")

        write_json(output_dir / "raw_collection.json", collection_data)
        if campaign_feed_data is not None:
            write_json(output_dir / "raw_campaign_posts.json", campaign_feed_data)
        print(f"[+] found {len(refs)} published Patreon post(s)", file=sys.stderr)
        await _download_cover(fetcher, meta)
        post_results = await _crawl_post_snapshots_with_login_retry(
            fetcher,
            refs,
            concurrency=concurrency,
            headless=headless,
        )
        if include_comments:
            comment_lists = await crawl_items(
                refs,
                lambda ref: _safe_fetch_comments(fetcher, ref),
                concurrency=max(1, min(concurrency, 4)),
                item_name="comments",
            )
        else:
            comment_lists = [[] for _ in refs]
    finally:
        await fetcher.stop()

    cover: dict[str, Any] = {"url": meta.cover_url, "mime": meta.cover_mime}
    if meta.cover_bytes:
        cover_name = f"cover{mimetypes.guess_extension(meta.cover_mime) or '.jpg'}"
        cover_path = output_dir / "assets" / cover_name
        cover_path.write_bytes(meta.cover_bytes)
        cover["path"] = f"assets/{cover_name}"

    manifest_chapters: list[dict[str, Any]] = []
    chapter_index: list[dict[str, Any]] = []
    for index, (ref, result, comments) in enumerate(zip(refs, post_results, comment_lists, strict=True), 1):
        chapter, raw_post = result
        chapter_id = f"{index:02d}"
        source_chapter_number = _chapter_number_from_title(chapter.title)
        source_chapter_label = _source_chapter_label(source_chapter_number)
        source_file_stem = _source_chapter_file_stem(source_chapter_number)
        source_chapter_path = (
            f"chapters/by-source/{source_file_stem}.json" if source_file_stem else None
        )
        source_comments_path = (
            f"comments/by-source/{source_file_stem}.json" if source_file_stem else None
        )
        blocks, paragraphs = _snapshot_blocks(chapter)
        chapter_payload = {
            "id": chapter_id,
            "title": chapter.title,
            "source_id": ref.post_id,
            "source_url": ref.url,
            "source_chapter_number": source_chapter_number,
            "source_chapter_label": source_chapter_label,
            "published_at": ref.published_at,
            "order": index - 1,
            "paragraphs": paragraphs,
            "blocks": blocks,
        }
        write_json(output_dir / "chapters" / f"{chapter_id}.json", chapter_payload)
        write_json(output_dir / "comments" / f"{chapter_id}.json", comments)
        if source_file_stem:
            write_json(
                output_dir / "chapters" / "by-source" / f"{source_file_stem}.json",
                chapter_payload,
            )
            write_json(
                output_dir / "comments" / "by-source" / f"{source_file_stem}.json",
                comments,
            )
        write_json(output_dir / "raw_posts" / f"{ref.post_id}.json", raw_post)
        manifest_chapters.append(
            {
                "id": chapter_id,
                "title": chapter.title,
                "source_id": ref.post_id,
                "source_url": ref.url,
                "source_chapter_number": source_chapter_number,
                "source_chapter_label": source_chapter_label,
                "published_at": ref.published_at,
                "order": index - 1,
                "path": f"chapters/{chapter_id}.json",
                "comments_path": f"comments/{chapter_id}.json",
                "source_chapter_path": source_chapter_path,
                "source_comments_path": source_comments_path,
                "paragraph_count": len(paragraphs),
                "block_count": len(blocks),
            }
        )
        chapter_index.append(
            {
                "id": chapter_id,
                "title": chapter.title,
                "source_chapter_number": source_chapter_number,
                "source_chapter_label": source_chapter_label,
                "path": f"chapters/{chapter_id}.json",
                "comments_path": f"comments/{chapter_id}.json",
                "source_chapter_path": source_chapter_path,
                "source_comments_path": source_comments_path,
            }
        )

    manifest = {
        "schema_version": CRAWL_SCHEMA_VERSION,
        "source": {
            "provider": "patreon",
            "url": collection_url,
            "collection_id": collection_id,
        },
        "title": meta.title,
        "author": meta.author,
        "language": "en",
        "target_language": "zh-CN",
        "intro_paragraphs": meta.intro_paragraphs,
        "volume_title": meta.volume_title,
        "cover": cover,
        "chapters": manifest_chapters,
    }
    write_json(output_dir / "chapter_index.json", chapter_index)
    write_json(output_dir / "manifest.json", manifest)
    return output_dir / "manifest.json"


async def crawl_book(
    book_url: str,
    *,
    headless: bool = False,
    delay: float = 0.4,
    concurrency: int = 2,
) -> tuple[BookMeta, list[Volume]]:
    collection_id = _collection_id_from_url(book_url)
    collection_url = _resolve_book_url(book_url)
    fetcher = PatreonFetcher(headless=headless, delay=delay)
    await fetcher.start()
    try:
        await fetcher.open_collection(collection_url)
        collection_data = await fetcher.get_json(_api_collection_url(collection_id))
        meta, refs = _parse_collection(collection_data, collection_id)
        refs, _campaign_feed_data = await _supplement_refs_from_campaign_feed(
            fetcher,
            collection_data,
            refs,
        )
        if not refs:
            raise RuntimeError(f"Patreon collection {collection_id} has no published posts")

        print(f"[+] found {len(refs)} published Patreon post(s)", file=sys.stderr)
        await _download_cover(fetcher, meta)
        chapters = await _crawl_posts_with_login_retry(
            fetcher,
            refs,
            concurrency=concurrency,
            headless=headless,
        )
    finally:
        await fetcher.stop()

    volume_title = meta.volume_title or ""
    return meta, [Volume(title=volume_title, chapters=chapters)]


def build_epub(meta: BookMeta, volumes: list[Volume], out_path: Path) -> None:
    write_epub(
        identifier=f"patreon-{meta.collection_id}-{int(time.time())}",
        title=meta.title,
        author=meta.author,
        volumes=volumes,
        out_path=out_path,
        intro_paragraphs=meta.intro_paragraphs,
        cover_bytes=meta.cover_bytes,
        cover_mime=meta.cover_mime,
    )


def _resolve_book_url(target: str) -> str:
    collection_id = _collection_id_from_url(target)
    return f"{HOST}/collection/{collection_id}?view=condensed"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download a Patreon collection into an EPUB"
    )
    parser.add_argument("url", help="Patreon collection URL")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.4)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args(argv)

    meta, volumes = asyncio.run(
        crawl_book(
            args.url,
            headless=args.headless,
            delay=args.delay,
            concurrency=args.concurrency,
        )
    )
    build_epub(meta, volumes, args.output)
    print(f"[+] wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
