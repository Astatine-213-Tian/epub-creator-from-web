from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path


BROWSER_PATH_ENV = "BOOKLIB_BROWSER_PATH"


def resolve_browser_executable() -> str:
    """Return a Chromium/Chrome executable path for browser-backed providers.

    Resolution order:
    1. BOOKLIB_BROWSER_PATH environment override.
    2. Playwright-managed Chromium, if playwright is installed and browsers exist.
    3. Common Chromium/Chrome executable names on PATH.
    4. Common macOS app bundle paths.
    """
    env_path = os.environ.get(BROWSER_PATH_ENV)
    if env_path:
        return _existing_executable(env_path)

    playwright_path = _playwright_chromium_path()
    if playwright_path:
        return playwright_path

    for name in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
        "msedge",
    ):
        path = shutil.which(name)
        if path:
            return path

    if platform.system() == "Darwin":
        for path in (
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ):
            if Path(path).is_file():
                return path

    raise RuntimeError(
        "could not find a Chromium-compatible browser. "
        f"Set {BROWSER_PATH_ENV} to the browser executable path, or install "
        "Playwright Chromium / system Chromium."
    )


async def start_browser():
    import zendriver as zd

    return await zd.start(
        zd.Config(
            headless=False,
            browser_executable_path=resolve_browser_executable(),
            sandbox=False,
            browser_connection_timeout=1.0,
            browser_connection_max_tries=30,
        )
    )


async def wait_for_page_ready(
    tab,
    *,
    ready_selector: str | None = None,
    title_timeout: int = 60,
    selector_timeout: int = 60,
    selector_interval: float = 0.5,
    settle_delay: float = 0.5,
) -> None:
    import asyncio

    for _ in range(title_timeout):
        title = await tab.evaluate("document.title")
        if title and "moment" not in title.lower() and "稍候" not in title:
            break
        await asyncio.sleep(1)

    if ready_selector:
        selector_js = f"Boolean(document.querySelector({ready_selector!r}))"
        for _ in range(selector_timeout):
            if await tab.evaluate(selector_js):
                break
            await asyncio.sleep(selector_interval)

    await asyncio.sleep(settle_delay)


def _existing_executable(path: str) -> str:
    expanded = Path(path).expanduser()
    if expanded.is_file():
        return str(expanded)
    raise RuntimeError(f"{BROWSER_PATH_ENV} points to a missing file: {expanded}")


def _playwright_chromium_path() -> str | None:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None

    try:
        with sync_playwright() as p:
            path = p.chromium.executable_path
    except Exception:
        return None

    if path and Path(path).is_file():
        return path
    return None
