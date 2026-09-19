from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp.types import CallToolResult, TextContent

from src.notion.mcp import (
    SERVER_URL,
    CallbackServer,
    LoginRequired,
    MCPTools,
    PersistentOAuth,
    TokenStore,
    error_message,
    require_login,
)


class AuthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "private" / "oauth.json"

    async def test_rotated_credentials_and_expiry_survive_restart_privately(self):
        store = TokenStore(self.path)
        client = OAuthClientInformationFull(
            client_id="test-client", redirect_uris=["http://127.0.0.1:8000/callback"]
        )
        with store.locked():
            await store.set_client_info(client)
            await store.set_tokens(
                OAuthToken(
                    access_token="first", refresh_token="refresh-1", expires_in=3600
                )
            )
            await store.set_tokens(
                OAuthToken(
                    access_token="second", refresh_token="refresh-2", expires_in=3600
                )
            )
            with self.assertRaisesRegex(ValueError, "Another"):
                with TokenStore(self.path).locked():
                    pass
        restored = TokenStore(self.path)
        with restored.locked():
            self.assertEqual((await restored.get_tokens()).refresh_token, "refresh-2")
            self.assertEqual(
                (await restored.get_client_info()).client_id, "test-client"
            )
            self.assertGreater(restored.data["expires_at"], time.time() + 3500)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("refresh-1", self.path.read_text())

    async def test_invalid_grant_stops_but_transient_failure_preserves_tokens(self):
        store = TokenStore(self.path)
        with store.locked():
            await store.set_tokens(
                OAuthToken(
                    access_token="private-value", refresh_token="private-refresh"
                )
            )
            auth = PersistentOAuth(
                SERVER_URL,
                OAuthClientMetadata(redirect_uris=["http://127.0.0.1:8000/callback"]),
                store,
                require_login,
                require_login,
            )
            with self.assertRaisesRegex(ValueError, "retry later"):
                await auth._handle_refresh_response(
                    httpx.Response(503, json={"error": "temporary"})
                )
            self.assertIsNotNone(await store.get_tokens())
            with self.assertRaises(LoginRequired) as error:
                await auth._handle_refresh_response(
                    httpx.Response(
                        400, json={"error": "invalid_grant", "detail": "private-value"}
                    )
                )
            self.assertNotIn("private-value", str(error.exception))
            self.assertIsNone(await store.get_tokens())

    async def test_callback_validates_state_and_accepts_only_loopback_callback(self):
        callback = CallbackServer()
        try:
            callback.expected_state = "expected"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    callback.redirect_uri,
                    params={"code": "private-code", "state": "wrong"},
                )
                self.assertEqual(response.status_code, 400)
                self.assertFalse(callback.done.is_set())
                response = await client.get(
                    callback.redirect_uri,
                    params={"code": "private-code", "state": "expected"},
                )
                self.assertEqual(response.status_code, 200)
            self.assertEqual(await callback.callback(), ("private-code", "expected"))
        finally:
            await asyncio.to_thread(callback.close)

    async def test_streaming_refresh_response_is_consumed_and_persisted(self):
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield json.dumps(
                    {
                        "access_token": "new-access",
                        "refresh_token": "new-refresh",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    }
                ).encode()

        store = TokenStore(self.path)
        with store.locked():
            auth = PersistentOAuth(
                SERVER_URL,
                OAuthClientMetadata(redirect_uris=["http://127.0.0.1:8000/callback"]),
                store,
                require_login,
                require_login,
            )
            result = await auth._handle_refresh_response(
                httpx.Response(200, stream=Stream())
            )
            self.assertTrue(result)
            self.assertEqual((await store.get_tokens()).refresh_token, "new-refresh")


class ToolTests(unittest.IsolatedAsyncioTestCase):
    def test_sdk_validation_details_do_not_leak_through_task_groups(self):
        class ExternalValidationError(ValueError):
            pass

        error = ExceptionGroup(
            "request failed", [ExternalValidationError("private-token")]
        )
        self.assertNotIn("private-token", error_message(error))
        self.assertEqual(
            error_message(
                ExceptionGroup("failed", [ValueError("Book outline changed")])
            ),
            "Book outline changed",
        )

    def setUp(self):
        sleeping = patch("src.notion.mcp.asyncio.sleep", new_callable=AsyncMock)
        sleeping.start()
        self.addCleanup(sleeping.stop)

    async def test_failed_reads_retry_but_uncertain_writes_do_not_repeat(self):
        session = AsyncMock()
        result = CallToolResult(content=[], structuredContent={"text": "body"})
        session.call_tool.side_effect = [TimeoutError(), result]
        self.assertEqual(
            await MCPTools(session).call("notion-fetch", {"id": "page"}),
            {"text": "body"},
        )
        self.assertEqual(session.call_tool.await_count, 2)
        session.reset_mock()
        session.call_tool.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            await MCPTools(session).call("notion-create-pages", {"pages": []})
        self.assertEqual(session.call_tool.await_count, 1)

    async def test_reads_structured_or_json_result_without_leaking_server_error(self):
        session = AsyncMock()
        tools = MCPTools(session)
        for result in [
            CallToolResult(content=[], structuredContent={"results": []}),
            CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"results": []}))]
            ),
        ]:
            session.call_tool.return_value = result
            tools.last_call = 0
            self.assertEqual(
                await tools.call("notion-fetch", {"id": "test"}), {"results": []}
            )
        session.call_tool.return_value = CallToolResult(
            isError=True, content=[TextContent(type="text", text="private-value")]
        )
        tools.last_call = 0
        with self.assertRaises(ValueError) as error:
            await tools.call("notion-fetch", {"id": "test"})
        self.assertNotIn("private-value", str(error.exception))


if __name__ == "__main__":
    unittest.main()
