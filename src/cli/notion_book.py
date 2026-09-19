"""Authenticate or resume a crawl upload to the Notion CMS."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.notion.cms import CONFIG
from src.notion.mcp import (
    AUTH_FILE,
    TokenStore,
    connect,
    error_message,
    exclusive_lock,
    login,
)
from src.notion.upload import upload_draft


async def resume(state: Path, config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    with exclusive_lock(state.parent / "import.lock"):
        book = json.loads(state.read_text())
        store = TokenStore(AUTH_FILE)
        with store.locked():
            async with connect(store) as tools:
                await upload_draft(book, state, config, tools=tools)
        print("CMS draft verified: https://www.notion.so/" + book["work_id"])


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["login", "logout", "resume"])
    parser.add_argument(
        "--state", type=Path, help="generated/notion_cms_sources/.../import.json"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.command == "resume":
        if args.state is None:
            parser.error("resume requires --state")
        asyncio.run(resume(args.state, args.config))
        return
    store = TokenStore(AUTH_FILE)
    with store.locked():
        if args.command == "login":
            asyncio.run(login(store))
        else:
            store.clear_tokens()
            print("Local Notion MCP tokens removed")


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; resume the saved draft checkpoint") from None
    except Exception as error:  # noqa: BLE001 - sanitize SDK errors at the CLI boundary
        raise SystemExit(error_message(error)) from None


if __name__ == "__main__":
    main()
