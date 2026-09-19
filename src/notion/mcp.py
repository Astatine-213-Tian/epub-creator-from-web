from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import tempfile
import threading
import time
import webbrowser
from contextlib import asynccontextmanager, contextmanager
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import AsyncIterator, Iterator
from urllib.parse import parse_qs, urlsplit

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
)

SERVER_URL = "https://mcp.notion.com/mcp"
AUTH_FILE = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state"))
    / "epub-creator-from-web/notion-mcp.json"
)


class LoginRequired(ValueError):
    def __init__(self) -> None:
        super().__init__("Notion authorization required. Run: uv run book-notion login")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(lock, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another Notion sync command is running; retry after it finishes"
            ) from None
        yield
    finally:
        os.close(lock)


class TokenStore:
    """One private, atomically replaced file; callers hold its process lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict = {}
        self.context = None

    @contextmanager
    def locked(self) -> Iterator[None]:
        with exclusive_lock(Path(str(self.path) + ".lock")):
            if self.path.exists():
                self.path.chmod(0o600)
                try:
                    self.data = json.loads(self.path.read_text())
                except (ValueError, OSError):
                    raise ValueError(
                        "Cannot load Notion credentials; restore the credential file"
                    ) from None
            yield

    def save(self) -> None:
        fd, name = tempfile.mkstemp(prefix=".notion-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as output:
                json.dump(self.data, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.path)
        finally:
            Path(name).unlink(missing_ok=True)

    async def get_tokens(self) -> OAuthToken | None:
        value = self.data.get("tokens")
        return OAuthToken.model_validate(value) if value else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.data["tokens"] = tokens.model_dump(mode="json")
        self.data["expires_at"] = (
            time.time() + tokens.expires_in if tokens.expires_in is not None else None
        )
        if self.context and self.context.oauth_metadata:
            self.data["oauth_metadata"] = self.context.oauth_metadata.model_dump(
                mode="json"
            )
        self.save()

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        value = self.data.get("client")
        return OAuthClientInformationFull.model_validate(value) if value else None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self.data["client"] = client_info.model_dump(mode="json")
        self.save()

    def clear_tokens(self) -> None:
        self.data.pop("tokens", None)
        self.data.pop("expires_at", None)
        self.save()


class PersistentOAuth(OAuthClientProvider):
    """Persist expiry across SDK sessions and stop on a rejected refresh grant."""

    async def _handle_refresh_response(self, response: httpx.Response) -> bool:
        await response.aread()
        if response.status_code != 200:
            try:
                code = response.json().get("error")
            except ValueError:
                code = None
            if code in {"invalid_grant", "invalid_client"}:
                self.context.storage.clear_tokens()
                raise LoginRequired()
            raise ValueError(
                f"Notion token refresh failed (HTTP {response.status_code}); retry later"
            )
        try:
            OAuthToken.model_validate_json(response.content)
        except ValueError:
            raise ValueError("Notion returned invalid refresh credentials") from None
        return await super()._handle_refresh_response(response)


class CallbackServer:
    def __init__(self, redirect_uri: str | None = None) -> None:
        self.result: tuple[str, str | None] | None = None
        self.error = False
        self.expected_state: str | None = None
        self.done = threading.Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlsplit(self.path)
                args = parse_qs(parsed.query)
                if (
                    not owner.expected_state
                    or parsed.path != "/callback"
                    or args.get("state", [None])[0] != owner.expected_state
                ):
                    self.send_error(400)
                    return
                if args.get("code"):
                    owner.result = (args["code"][0], args.get("state", [None])[0])
                else:
                    owner.error = True
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    "Authorization received. You can close this tab.".encode()
                )
                owner.done.set()

            def log_message(self, *_args) -> None:
                pass

        port = urlsplit(redirect_uri).port if redirect_uri else 0
        self.server = HTTPServer(("127.0.0.1", port or 0), Handler)
        self.redirect_uri = f"http://127.0.0.1:{self.server.server_port}/callback"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    async def redirect(self, url: str) -> None:
        self.expected_state = parse_qs(urlsplit(url).query)["state"][0]
        print(f"Authorize Notion in your browser:\n{url}", flush=True)
        webbrowser.open(url)

    async def callback(self) -> tuple[str, str | None]:
        if not await asyncio.to_thread(self.done.wait, 600):
            raise ValueError("Notion login timed out; run book-notion login again")
        if self.error or not self.result:
            raise ValueError("Notion authorization was not granted")
        return self.result

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


async def require_login(*_args) -> None:
    raise LoginRequired()


class MCPTools:
    def __init__(self, session: ClientSession) -> None:
        self.session = session
        self.last_call = 0.0
        self.pacing = asyncio.Lock()

    async def call(self, name: str, arguments: dict) -> dict:
        attempts = (
            3
            if name
            in {"notion-fetch", "notion-query-data-sources", "notion-get-async-task"}
            else 1
        )
        for attempt in range(attempts):
            async with self.pacing:
                await asyncio.sleep(max(0, 0.4 - (time.monotonic() - self.last_call)))
                self.last_call = time.monotonic()
            try:
                result = await self.session.call_tool(name, arguments)
                if result.isError:
                    messages = " ".join(
                        part.text for part in result.content if part.type == "text"
                    )
                    error = ValueError(
                        f"Notion MCP {name} failed; inspect the tool error before retrying"
                    )
                    error.tool_message = messages
                    raise error
                break
            except LoginRequired:
                raise
            except Exception:
                # Reads are safe to repeat; writes require checkpoint reconciliation.
                if attempt + 1 == attempts:
                    raise
                await asyncio.sleep(2 ** (attempt + 1))
        if result.structuredContent is not None:
            return result.structuredContent
        parts = [part.text for part in result.content if part.type == "text"]
        if len(parts) != 1:
            raise ValueError(f"Unexpected Notion MCP {name} result")
        try:
            value = json.loads(parts[0])
        except ValueError:
            raise ValueError(f"Notion MCP {name} returned invalid JSON") from None
        if not isinstance(value, dict):
            raise ValueError(f"Notion MCP {name} returned an invalid object")
        return value


@asynccontextmanager
async def connect(store: TokenStore, *, login: bool = False) -> AsyncIterator[MCPTools]:
    if not login and (not store.data.get("tokens") or not store.data.get("client")):
        raise LoginRequired()
    client = await store.get_client_info()
    redirect_uri = (
        str(client.redirect_uris[0]) if client and client.redirect_uris else None
    )
    callback = CallbackServer(redirect_uri) if login else None
    metadata = OAuthClientMetadata(
        client_name="EPUB fanwai sync",
        redirect_uris=[callback.redirect_uri if callback else redirect_uri],
        token_endpoint_auth_method="none",
    )
    oauth = PersistentOAuth(
        SERVER_URL,
        metadata,
        store,
        callback.redirect if callback else require_login,
        callback.callback if callback else require_login,
    )
    store.context = oauth.context
    # SDK 1.x does not restore expiry or discovery metadata from TokenStorage.
    expiry = store.data.get("expires_at")
    oauth.context.token_expiry_time = expiry - 60 if expiry is not None else None
    if store.data.get("oauth_metadata"):
        oauth.context.oauth_metadata = OAuthMetadata.model_validate(
            store.data["oauth_metadata"]
        )
    logging.getLogger(OAuthClientProvider.__module__).disabled = True
    try:
        async with httpx.AsyncClient(
            auth=oauth, timeout=60, follow_redirects=True
        ) as http:
            async with streamable_http_client(SERVER_URL, http_client=http) as (
                read,
                write,
                _,
            ):
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=timedelta(seconds=660 if login else 120),
                ) as session:
                    await session.initialize()
                    yield MCPTools(session)
    finally:
        if callback:
            callback.close()


async def finish(tools, result: dict) -> dict:
    while result.get("object") == "async_task":
        if result["status"] == "succeeded":
            return result.get("result", result)
        if result["status"] in {"failed", "cancelled"}:
            raise ValueError(
                "Notion background write failed: "
                + json.dumps(result.get("error", {}), ensure_ascii=False)
            )
        await asyncio.sleep(min(10, result.get("poll_after_seconds", 2)))
        result = await tools.call("notion-get-async-task", {"task_id": result["id"]})
    return result


async def login(store: TokenStore) -> None:
    store.clear_tokens()
    async with connect(store, login=True) as tools:
        await tools.call("notion-fetch", {"id": "self"})
    print("Notion MCP login saved")


def error_message(error: BaseException) -> str:
    """Unwrap SDK task groups without printing request or credential details."""

    def known(exc: BaseException) -> str | None:
        if type(exc) is ValueError or isinstance(exc, LoginRequired):
            return str(exc)
        for child in getattr(exc, "exceptions", []):
            if message := known(child):
                return message
        return None

    return (
        known(error)
        or "Notion sync failed; check connectivity and retry. Run book-notion login if authorization expired."
    )
