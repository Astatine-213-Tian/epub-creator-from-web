from __future__ import annotations

import importlib
import re
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Protocol

from src.providers.registry import PARSERS
from src.runtime.progress import ProgressLogger


@dataclass(frozen=True)
class SearchResult:
    parser: str
    title: str
    author: str
    url: str
    source: str
    snippet: str = ""
    raw_score: float = 0.0


@dataclass(frozen=True)
class BookPreview:
    parser: str
    title: str
    author: str
    url: str
    chapter_count: int | None
    first_chapters: tuple[str, ...]
    last_chapters: tuple[str, ...]
    intro: str = ""
    status: str = ""
    source: str = ""
    match_level: int = 0


class SearchModule(Protocol):
    def search_books(self, query: str, *, limit: int = 10) -> list[SearchResult]:
        ...

    def preview_book(self, result: SearchResult) -> BookPreview:
        ...


PROVIDER_PRIORITY = {
    "pili45": 3,
    "towasakata": 2,
    "xfxs": 1,
}
SEARCH_JOB_TIMEOUT_SECONDS = 90
PREVIEW_JOB_TIMEOUT_SECONDS = 120


def normalize_query_text(value: str) -> str:
    try:
        from opencc import OpenCC

        value = OpenCC("t2s").convert(value)
    except Exception:
        pass
    value = value.casefold()
    return re.sub(r"[\s\xa0　\W_]+", "", value)


def match_level(query: str, *, title: str, author: str = "", snippet: str = "") -> int:
    q = normalize_query_text(query)
    t = normalize_query_text(title)
    a = normalize_query_text(author)
    s = normalize_query_text(snippet)
    if not q:
        return 0
    if q == t:
        return 4
    if q in t or t in q:
        return 3
    tokens = [normalize_query_text(token) for token in re.split(r"[\s,，;；]+", query)]
    tokens = [token for token in tokens if token]
    if tokens and all(token in t for token in tokens):
        return 2
    if q in a or q in s:
        return 1
    return 0


def load_search_module(parser_name: str) -> SearchModule | None:
    try:
        return importlib.import_module(f"src.providers.{parser_name}.search")
    except ModuleNotFoundError as exc:
        if exc.name == f"src.providers.{parser_name}.search":
            return None
        raise


def _uses_browser_for_search(module: SearchModule) -> bool:
    return callable(getattr(module, "search_books_with_browser", None))


def _uses_browser_for_preview(module: SearchModule) -> bool:
    return callable(getattr(module, "preview_book_with_browser", None))


@dataclass(frozen=True)
class SearchJobOutcome:
    order: int
    name: str
    results: list[SearchResult] | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class PreviewJob:
    index: int
    result: SearchResult
    module: SearchModule


@dataclass(frozen=True)
class PreviewJobOutcome:
    index: int
    result: SearchResult
    preview: BookPreview | None = None
    error: Exception | None = None


def search_all(
    query: str,
    *,
    parser_name: str | None = None,
    limit_per_provider: int = 10,
    verbose: bool = False,
    debug: bool = False,
) -> list[SearchResult]:
    progress = ProgressLogger(verbose=verbose, debug_enabled=debug)
    results: list[SearchResult] = []
    jobs = []
    for order, spec in enumerate(PARSERS):
        if parser_name and spec.name != parser_name:
            continue
        module = load_search_module(spec.name)
        if module is None:
            continue
        jobs.append((order, spec.name, module))

    if not jobs:
        progress.info("Search complete: 0 unique result(s)")
        return []

    browser_jobs = [
        (order, name, module) for order, name, module in jobs if _uses_browser_for_search(module)
    ]
    thread_jobs = [
        (order, name, module) for order, name, module in jobs if not _uses_browser_for_search(module)
    ]
    worker_note = f"{len(thread_jobs)} threaded, {len(browser_jobs)} browser tab(s)"
    if debug:
        progress.info(f"Searching {len(jobs)} provider(s) in parallel ({worker_note})")
    else:
        search_counter = progress.counter("Search", len(jobs), "provider(s)")
        search_counter.start()
    completed: list[tuple[int, list[SearchResult]]] = []

    def record_search_outcome(outcome: SearchJobOutcome) -> None:
        if outcome.error is not None:
            progress.warning(f"search failed for {outcome.name}: {outcome.error}")
            if not debug:
                search_counter.advance(detail=f"{outcome.name}: failed")
            return
        provider_results = outcome.results or []
        if debug:
            progress.provider(outcome.name, f"{len(provider_results)} result(s)")
        else:
            search_counter.advance(detail=f"{outcome.name}: {len(provider_results)} result(s)")
        completed.append((outcome.order, provider_results))

    asyncio.run(
        _run_search_jobs(
            query,
            thread_jobs=thread_jobs,
            browser_jobs=browser_jobs,
            limit_per_provider=limit_per_provider,
            progress=progress,
            on_outcome=record_search_outcome,
        )
    )

    for _order, provider_results in sorted(completed, key=lambda item: item[0]):
        results.extend(provider_results)
    deduped = dedupe_results(results)
    if debug:
        progress.info(f"Search complete: {len(deduped)} unique result(s)")
    else:
        search_counter.finish(f"Search complete: {len(deduped)} unique result(s)")
    return deduped


async def _run_search_jobs(
    query: str,
    *,
    thread_jobs: list[tuple[int, str, SearchModule]],
    browser_jobs: list[tuple[int, str, SearchModule]],
    limit_per_provider: int,
    progress: ProgressLogger,
    on_outcome: Callable[[SearchJobOutcome], None],
) -> None:
    tasks: list[asyncio.Task[SearchJobOutcome]] = []
    loop = asyncio.get_running_loop()
    thread_executor = ThreadPoolExecutor(max_workers=len(thread_jobs)) if thread_jobs else None
    browser = None
    try:
        if thread_executor is not None:
            for order, name, module in thread_jobs:
                progress.detail(f"searching {name}")
                tasks.append(
                    asyncio.create_task(
                        _with_search_timeout(
                            _run_thread_search(
                                loop,
                                thread_executor,
                                order,
                                name,
                                module,
                                query,
                                limit_per_provider,
                            ),
                            order=order,
                            name=name,
                        )
                    )
                )

        if browser_jobs:
            browser = await _start_search_browser()
            for order, name, module in browser_jobs:
                progress.detail(f"searching {name} in browser tab")
                tasks.append(
                    asyncio.create_task(
                        _with_search_timeout(
                            _run_browser_tab_search(
                                order,
                                name,
                                module,
                                query,
                                limit_per_provider,
                                browser,
                            ),
                            order=order,
                            name=name,
                        )
                    )
                )

        for task in asyncio.as_completed(tasks):
            on_outcome(await task)
    finally:
        if browser is not None:
            await browser.stop()
        if thread_executor is not None:
            thread_executor.shutdown(wait=False, cancel_futures=True)


async def _run_thread_search(
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    order: int,
    name: str,
    module: SearchModule,
    query: str,
    limit_per_provider: int,
) -> SearchJobOutcome:
    try:
        results = await loop.run_in_executor(
            executor,
            partial(module.search_books, query, limit=limit_per_provider),
        )
    except Exception as exc:  # noqa: BLE001
        return SearchJobOutcome(order=order, name=name, error=exc)
    return SearchJobOutcome(order=order, name=name, results=results)


async def _with_search_timeout(
    task: asyncio.Future[SearchJobOutcome],
    *,
    order: int,
    name: str,
) -> SearchJobOutcome:
    try:
        return await asyncio.wait_for(task, timeout=SEARCH_JOB_TIMEOUT_SECONDS)
    except TimeoutError:
        return SearchJobOutcome(
            order=order,
            name=name,
            error=TimeoutError(f"timed out after {SEARCH_JOB_TIMEOUT_SECONDS}s"),
        )


async def _run_browser_tab_search(
    order: int,
    name: str,
    module: SearchModule,
    query: str,
    limit_per_provider: int,
    browser: Any,
) -> SearchJobOutcome:
    try:
        search_with_browser = getattr(module, "search_books_with_browser")
        results = await search_with_browser(query, limit=limit_per_provider, browser=browser)
    except Exception as exc:  # noqa: BLE001
        return SearchJobOutcome(order=order, name=name, error=exc)
    return SearchJobOutcome(order=order, name=name, results=results)


async def _start_search_browser() -> Any:
    from src.fetch.browser import start_browser

    return await start_browser()


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[tuple[str, str]] = set()
    deduped: list[SearchResult] = []
    for result in results:
        key = (result.parser, result.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


def rank_results(query: str, results: list[SearchResult]) -> list[SearchResult]:
    return sorted(
        results,
        key=lambda result: (
            match_level(
                query,
                title=result.title,
                author=result.author,
                snippet=result.snippet,
            ),
            result.raw_score,
            PROVIDER_PRIORITY.get(result.parser, 0),
        ),
        reverse=True,
    )


def build_previews(
    query: str,
    results: list[SearchResult],
    *,
    max_previews: int = 20,
    verbose: bool = False,
    debug: bool = False,
) -> list[BookPreview]:
    progress = ProgressLogger(verbose=verbose, debug_enabled=debug)
    previews: list[BookPreview] = []
    ranked = rank_results(query, results)[:max_previews]
    preview_jobs: list[PreviewJob] = []
    for index, result in enumerate(ranked, 1):
        module = load_search_module(result.parser)
        if module is None:
            continue
        preview_jobs.append(PreviewJob(index=index, result=result, module=module))

    if debug:
        browser_count = sum(1 for job in preview_jobs if _uses_browser_for_preview(job.module))
        progress.info(
            f"Building previews for {len(preview_jobs)} candidate(s) "
            f"({len(preview_jobs) - browser_count} threaded, {browser_count} browser tab(s))"
        )
    else:
        preview_counter = progress.counter("Preview", len(preview_jobs), "candidate(s)")
        preview_counter.start()

    def record_preview_outcome(outcome: PreviewJobOutcome) -> None:
        result = outcome.result
        if outcome.error is not None:
            progress.warning(f"preview failed for {result.parser} {result.url}: {outcome.error}")
            if not debug:
                preview_counter.advance(detail=f"{result.parser}: failed")
            return
        preview = outcome.preview
        if preview is None:
            if not debug:
                preview_counter.advance(detail=f"{result.parser}: failed")
            return
        previews.append(
            BookPreview(
                parser=preview.parser,
                title=preview.title,
                author=preview.author,
                url=preview.url,
                chapter_count=preview.chapter_count,
                first_chapters=preview.first_chapters,
                last_chapters=preview.last_chapters,
                intro=preview.intro,
                status=preview.status,
                source=preview.source,
                match_level=match_level(
                    query,
                    title=preview.title,
                    author=preview.author,
                    snippet=preview.intro,
                ),
            )
        )
        count = preview.chapter_count if preview.chapter_count is not None else "unknown"
        if debug:
            progress.info(f"Preview ready: {preview.parser} | {preview.title} ({count} chapter(s))")
        else:
            preview_counter.advance(detail=f"{preview.parser}: {count} chapter(s)")

    asyncio.run(
        _run_preview_jobs(
            preview_jobs,
            progress=progress,
            debug=debug,
            on_outcome=record_preview_outcome,
        )
    )
    ranked_previews = rank_previews(dedupe_previews(previews))
    if debug:
        progress.info(f"Previews complete: {len(ranked_previews)} usable preview(s)")
    else:
        preview_counter.finish(f"Previews complete: {len(ranked_previews)} usable preview(s)")
    return ranked_previews


async def _run_preview_jobs(
    jobs: list[PreviewJob],
    *,
    progress: ProgressLogger,
    debug: bool,
    on_outcome: Callable[[PreviewJobOutcome], None],
) -> None:
    browser_jobs = [job for job in jobs if _uses_browser_for_preview(job.module)]
    thread_jobs = [job for job in jobs if not _uses_browser_for_preview(job.module)]
    tasks: list[asyncio.Task[PreviewJobOutcome]] = []
    loop = asyncio.get_running_loop()
    thread_executor = ThreadPoolExecutor(max_workers=len(thread_jobs)) if thread_jobs else None
    browser = None
    try:
        if thread_executor is not None:
            for job in thread_jobs:
                if debug:
                    progress.info(f"Preview {job.index}/{len(jobs)}: {job.result.parser} | {job.result.title}")
                    progress.detail(job.result.url)
                tasks.append(
                    asyncio.create_task(
                        _with_preview_timeout(
                            _run_thread_preview(loop, thread_executor, job),
                            job=job,
                        )
                    )
                )

        if browser_jobs:
            browser = await _start_search_browser()
            for job in browser_jobs:
                if debug:
                    progress.info(f"Preview {job.index}/{len(jobs)}: {job.result.parser} | {job.result.title}")
                    progress.detail(job.result.url)
                    progress.detail(f"previewing {job.result.parser} in browser tab")
                tasks.append(
                    asyncio.create_task(
                        _with_preview_timeout(_run_browser_tab_preview(job, browser), job=job)
                    )
                )

        for task in asyncio.as_completed(tasks):
            on_outcome(await task)
    finally:
        if browser is not None:
            await browser.stop()
        if thread_executor is not None:
            thread_executor.shutdown(wait=False, cancel_futures=True)


async def _run_thread_preview(
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    job: PreviewJob,
) -> PreviewJobOutcome:
    try:
        preview = await loop.run_in_executor(
            executor,
            partial(job.module.preview_book, job.result),
        )
    except Exception as exc:  # noqa: BLE001
        return PreviewJobOutcome(index=job.index, result=job.result, error=exc)
    return PreviewJobOutcome(index=job.index, result=job.result, preview=preview)


async def _with_preview_timeout(
    task: asyncio.Future[PreviewJobOutcome],
    *,
    job: PreviewJob,
) -> PreviewJobOutcome:
    try:
        return await asyncio.wait_for(task, timeout=PREVIEW_JOB_TIMEOUT_SECONDS)
    except TimeoutError:
        return PreviewJobOutcome(
            index=job.index,
            result=job.result,
            error=TimeoutError(f"timed out after {PREVIEW_JOB_TIMEOUT_SECONDS}s"),
        )


async def _run_browser_tab_preview(job: PreviewJob, browser: Any) -> PreviewJobOutcome:
    try:
        preview_with_browser = getattr(job.module, "preview_book_with_browser")
        preview = await preview_with_browser(job.result, browser=browser)
    except Exception as exc:  # noqa: BLE001
        return PreviewJobOutcome(index=job.index, result=job.result, error=exc)
    return PreviewJobOutcome(index=job.index, result=job.result, preview=preview)


def rank_previews(previews: list[BookPreview]) -> list[BookPreview]:
    return sorted(
        previews,
        key=lambda preview: (
            preview.match_level,
            preview.chapter_count if preview.chapter_count is not None else -1,
            PROVIDER_PRIORITY.get(preview.parser, 0),
        ),
        reverse=True,
    )


def dedupe_previews(previews: list[BookPreview]) -> list[BookPreview]:
    best_by_key: dict[tuple[object, ...], BookPreview] = {}
    for preview in previews:
        key = (
            preview.parser,
            normalize_query_text(preview.title),
            normalize_query_text(preview.author),
            preview.chapter_count,
            preview.first_chapters,
            preview.last_chapters,
        )
        current = best_by_key.get(key)
        if current is None or _preview_url_rank(preview.url) < _preview_url_rank(current.url):
            best_by_key[key] = preview
    return list(best_by_key.values())


def _preview_url_rank(url: str) -> tuple[int, str]:
    match = re.search(r"blog-entry-(\d+)\.html", url)
    if match:
        return int(match.group(1)), url
    return 0, url


def print_previews(previews: list[BookPreview]) -> None:
    print("\n== Results ==", flush=True)
    for index, preview in enumerate(previews, 1):
        print(_preview_heading(index, preview), flush=True)
        _print_preview_detail(preview)


def choose_preview(previews: list[BookPreview], *, first: bool = False) -> BookPreview | None:
    if not previews:
        return None
    if first:
        print_previews(previews[:1])
        return previews[0]
    if _can_use_interactive_menu():
        return _choose_preview_menu(previews)

    print_previews(previews)
    while True:
        try:
            choice = input("Choose result number, or q to quit: ").strip()
        except EOFError:
            return None
        if choice.lower() in {"q", "quit", "exit"}:
            return None
        if not choice.isdigit():
            print("Enter a number from the result list, or q.")
            continue
        index = int(choice)
        if 1 <= index <= len(previews):
            return previews[index - 1]
        print("Choice is outside the result list.")


def fake_menu_previews() -> list[BookPreview]:
    return [
        BookPreview(
            parser="pili45",
            title="全球高考",
            author="木苏里",
            url="https://www.pili45.com/5/2951/info.html",
            chapter_count=176,
            first_chapters=("第1章 送命题┃开场即结局", "第2章 监考官┃“你不会……认识吧？”“忘了”"),
            last_chapters=("第175章 2026.02.28番外（上）", "第176章 2026.03.01番外（下）"),
        ),
        BookPreview(
            parser="mgsf",
            title="全球高考",
            author="木苏里",
            url="https://www.mangguoshufang.com/1/185/info.html",
            chapter_count=173,
            first_chapters=("第1章送命题┃开场即结局", "第2章监考官┃“你不会……认识吧？”“忘了”"),
            last_chapters=("第172章生日小剧场", "第173章生日小剧场2"),
        ),
        BookPreview(
            parser="towasakata",
            title="全球高考",
            author="木苏里",
            url="http://towasakata.blog.fc2.com/blog-entry-461.html",
            chapter_count=166,
            first_chapters=("第1章 送命题│开场即结局", "第2章 监考官│「你不会……认识吧？」「忘了」"),
            last_chapters=("第165章 夏│秦究问：「算惊喜么？」", "第166章 秋│世界灿烂盛大，欢迎回家"),
        ),
        BookPreview(
            parser="quanben",
            title="全球高考",
            author="木苏里",
            url="https://quanben.io/n/quanqiugaokao/",
            chapter_count=166,
            first_chapters=("1.送命题", "2.监考官"),
            last_chapters=("165.夏", "166.秋"),
            status="完结",
        ),
        BookPreview(
            parser="jrkywsy",
            title="全球高考",
            author="木苏里",
            url="http://jrkywsy.blog.fc2.com/blog-entry-1031.html",
            chapter_count=1,
            first_chapters=("全球高考",),
            last_chapters=("全球高考",),
        ),
        BookPreview(
            parser="quanben",
            title="全球高考风暴",
            author="悠闲小神",
            url="https://quanben.io/n/quanqiugaokaofengbao/",
            chapter_count=347,
            first_chapters=("0001 地球考生请准备", "0002 超稀有卡牌"),
            last_chapters=("0346 成为任务员", "0347 大结局"),
            status="完结",
        ),
        BookPreview(
            parser="quanben",
            title="全球高考：谁说历史无用？",
            author="快活医生",
            url="https://quanben.io/n/quanqiugaokao-shuishuolishiwuyong/",
            chapter_count=207,
            first_chapters=("第1章 历史无用？", "第2章 第一场考试"),
            last_chapters=("第206章 最终答案", "第207章 完结"),
            status="完结",
        ),
        BookPreview(
            parser="quanben",
            title="高考后：假期送快递成全球首富！",
            author="文文图图",
            url="https://quanben.io/n/gaokaohou-jiaqisongkuaidichengquanqiushoufu/",
            chapter_count=173,
            first_chapters=("第1章 假期快递", "第2章 新订单"),
            last_chapters=("第172章 全球首富", "第173章 大结局"),
            status="完结",
        ),
    ]


def _preview_heading(index: int, preview: BookPreview) -> str:
    chapter_count = (
        f"{preview.chapter_count} chapter(s)"
        if preview.chapter_count is not None
        else "unknown chapter count"
    )
    author = f" / {preview.author}" if preview.author else ""
    status = f" / {preview.status}" if preview.status else ""
    return f"[{index}] {preview.parser} | {preview.title}{author} | {chapter_count}{status}"


def _print_preview_detail(preview: BookPreview) -> None:
    if preview.first_chapters:
        print(f"    First: {' / '.join(preview.first_chapters)}", flush=True)
    if preview.last_chapters:
        print(f"    Last: {' / '.join(preview.last_chapters)}", flush=True)
    print(f"    URL: {preview.url}", flush=True)


def _can_use_interactive_menu() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _choose_preview_menu(previews: list[BookPreview]) -> BookPreview | None:
    import termios
    import tty
    import webbrowser

    selected = 0
    message = ""
    fd = sys.stdin.fileno()
    original_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("\x1b[?25l", end="", flush=True)
        needs_render = True
        while True:
            if needs_render:
                _render_preview_menu(previews, selected, message=message)
                message = ""
                needs_render = False
            key = _read_menu_key()
            if key in {"UP", "k"}:
                selected = (selected - 1) % len(previews)
                needs_render = True
            elif key in {"DOWN", "j"}:
                selected = (selected + 1) % len(previews)
                needs_render = True
            elif key in {"ENTER", "\n", "\r"}:
                print("\nSelected:")
                print(_preview_heading(selected + 1, previews[selected]), flush=True)
                _print_preview_detail(previews[selected])
                return previews[selected]
            elif key in {"o", "O"}:
                opened = webbrowser.open(previews[selected].url)
                message = "Opened source URL in browser." if opened else "Could not open browser."
                needs_render = True
            elif key in {"q", "Q"}:
                print()
                return None
    finally:
        print("\x1b[?25h", end="", flush=True)
        termios.tcsetattr(fd, termios.TCSADRAIN, original_settings)


def _render_preview_menu(previews: list[BookPreview], selected: int, *, message: str = "") -> None:
    print("\x1b[2J\x1b[H", end="")
    width = _terminal_width()
    print("== Choose Source ==")
    print("Use Up/Down or j/k to move. Enter selects. o opens source. q cancels.\n")
    print(_trim_for_terminal("    #  Provider     Chapters       Title                              Author", width))
    print(_trim_for_terminal("    " + "-" * max(36, width - 4), width))
    for index, preview in enumerate(previews, 1):
        line = _preview_menu_row(index, preview, width - 3)
        if index - 1 == selected:
            print(f"\x1b[7m > {line}\x1b[0m")
        else:
            print(f"   {line}")
    print()
    _print_selected_preview(previews[selected], width)
    if message:
        print(_trim_for_terminal(f"\n{message}", width))


def _preview_menu_row(index: int, preview: BookPreview, width: int) -> str:
    chapter_count = (
        f"{preview.chapter_count} ch"
        if preview.chapter_count is not None
        else "unknown"
    )
    prefix = f"{index:>2}. {preview.parser:<11} {chapter_count:<13} "
    author_width = 16
    title_width = min(34, max(8, width - _display_width(prefix) - author_width - 1))
    title = _pad_display(_trim_display(preview.title, title_width), title_width)
    author = _pad_display(_trim_display(preview.author, author_width), author_width)
    return f"{prefix}{title} {author}"


def _print_selected_preview(preview: BookPreview, width: int) -> None:
    print(_trim_for_terminal("Preview " + "-" * max(24, width - 8), width))
    print(_trim_for_terminal(f"Source:   {preview.parser}", width))
    author = f" / {preview.author}" if preview.author else ""
    status = f" / {preview.status}" if preview.status else ""
    print(_trim_for_terminal(f"Book:     {preview.title}{author}{status}", width))
    if preview.first_chapters:
        print(_trim_for_terminal(f"First:    {' / '.join(preview.first_chapters)}", width))
    if preview.last_chapters:
        print(_trim_for_terminal(f"Last:     {' / '.join(preview.last_chapters)}", width))
    print(_trim_for_terminal(f"URL:      {preview.url}", width))


def _read_menu_key() -> str:
    import select

    char = sys.stdin.read(1)
    if char != "\x1b":
        if char in {"\n", "\r"}:
            return "ENTER"
        if char == "A":
            return "UP"
        if char == "B":
            return "DOWN"
        return char

    sequence = ""
    while select.select([sys.stdin], [], [], 0.03)[0]:
        sequence += sys.stdin.read(1)
        if sequence.endswith(("A", "B")):
            break
    if sequence in {"[A", "OA"}:
        return "UP"
    if sequence in {"[B", "OB"}:
        return "DOWN"
    if sequence.startswith("[") and sequence.endswith("A"):
        return "UP"
    if sequence.startswith("[") and sequence.endswith("B"):
        return "DOWN"
    return ""


def _terminal_width() -> int:
    try:
        return max(40, __import__("shutil").get_terminal_size((100, 24)).columns)
    except Exception:
        return 100


def _trim_for_terminal(value: str, width: int) -> str:
    return _trim_display(value, width)


def _trim_display(value: str, width: int) -> str:
    if _display_width(value) <= width:
        return value
    if width <= 3:
        return _slice_display(value, width)
    return f"{_slice_display(value, width - 3)}..."


def _pad_display(value: str, width: int) -> str:
    return value + " " * max(0, width - _display_width(value))


def _slice_display(value: str, width: int) -> str:
    out = []
    used = 0
    for char in value:
        char_width = _char_display_width(char)
        if used + char_width > width:
            break
        out.append(char)
        used += char_width
    return "".join(out)


def _display_width(value: str) -> int:
    return sum(_char_display_width(char) for char in value)


def _char_display_width(char: str) -> int:
    import unicodedata

    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
