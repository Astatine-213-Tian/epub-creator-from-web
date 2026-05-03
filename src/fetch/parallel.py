from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeVar

from src.runtime.progress import ProgressLogger


class FetchRef(Protocol):
    title: str


TRef = TypeVar("TRef", bound=FetchRef)
TResult = TypeVar("TResult")


RetryDelay = Callable[[BaseException, int], float]


def default_retry_delay(_exc: BaseException, attempt: int) -> float:
    return 1.0 * (attempt + 1)


async def retry_async(
    label: str,
    action: Callable[[], Awaitable[TResult]],
    *,
    attempts: int = 5,
    retry_delay: RetryDelay = default_retry_delay,
) -> TResult:
    """Run an async action with stderr retry reporting."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")

    for attempt in range(attempts):
        try:
            return await action()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            delay = retry_delay(exc, attempt)
            ProgressLogger().detail(
                f"fetch failed for {label}: {exc}; "
                f"retrying ({attempt + 2}/{attempts}) in {delay:.1f}s"
            )
            await asyncio.sleep(delay)

    raise RuntimeError("unreachable")


async def crawl_items(
    refs: Sequence[TRef],
    fetch_one: Callable[[TRef], Awaitable[TResult]],
    *,
    fallback_one: Callable[[TRef], Awaitable[TResult]] | None = None,
    concurrency: int = 4,
    item_name: str = "item",
    fallback_label: str = "serial fallback",
) -> list[TResult]:
    """Fetch explicit-TOC items concurrently while preserving TOC order.

    The fast path runs with bounded concurrency. Failed items are retried in
    original order through `fallback_one`, which is useful for browser-backed
    parsers where a normal navigation is more reliable than JS `fetch()`.
    """
    if not refs:
        return []

    total = len(refs)
    limit = max(1, concurrency)
    fallback = fallback_one or fetch_one
    progress = ProgressLogger()
    verbose = progress.debug_is_enabled()
    counter = None if verbose else progress.counter("Fetch", total, f"{item_name}(s)")
    if counter is not None:
        counter.start()
    completed_count = 0
    success = False

    try:
        if limit == 1:
            results: list[TResult] = []
            for index, ref in enumerate(refs):
                progress.detail(f"[{index + 1}/{total}] {ref.title}")
                result = await fallback(ref)
                results.append(result)
                completed_count = len(results)
                _advance_chapter_progress(
                    counter,
                    progress,
                    total=total,
                    done=len(results),
                    failed=0,
                    current=ref.title,
                )
            success = True
            return results

        semaphore = asyncio.Semaphore(limit)
        progress_lock = asyncio.Lock()
        completed = 0
        failed = 0

        async def log_progress(label: str, ref: TRef) -> None:
            _advance_chapter_progress(
                counter,
                progress,
                total=total,
                done=completed,
                failed=failed,
                current=ref.title,
                label=label,
            )

        async def run_one(index: int, ref: TRef) -> TResult:
            nonlocal completed, completed_count, failed
            async with semaphore:
                progress.detail(f"[{index + 1}/{total}] {ref.title}")
                try:
                    result = await fetch_one(ref)
                except Exception:
                    async with progress_lock:
                        failed += 1
                        await log_progress(f"failed_{item_name}", ref)
                    raise
                async with progress_lock:
                    completed += 1
                    completed_count = completed
                    await log_progress("current", ref)
                return result

        results: list[TResult | BaseException] = await asyncio.gather(
            *(run_one(index, ref) for index, ref in enumerate(refs)),
            return_exceptions=True,
        )
        failed_indexes = [
            index
            for index, result in enumerate(results)
            if isinstance(result, BaseException)
        ]
        if failed_indexes:
            progress.warning(
                f"fast path failed for {len(failed_indexes)} {item_name}(s); "
                f"retrying {fallback_label}"
            )

        for index in failed_indexes:
            ref = refs[index]
            progress.detail(f"fallback [{index + 1}/{total}] {ref.title}")
            try:
                results[index] = await fallback(ref)
            except Exception as exc:
                progress.warning(f"fallback failed for {ref.title}: {exc}")
                raise RuntimeError(
                    f"{item_name} fetch failed after fallback: "
                    f"[{index + 1}/{total}] {ref.title}"
                ) from exc
            async with progress_lock:
                completed += 1
                failed -= 1
                completed_count = completed
                await log_progress("fallback_recovered", ref)

        output = [result for result in results if not isinstance(result, BaseException)]
        completed_count = len(output)
        success = True
        return output
    finally:
        if counter is not None:
            if success:
                counter.finish(f"Fetch complete: {completed_count}/{total} {item_name}(s)")
            else:
                counter.close()


def _advance_chapter_progress(
    counter,
    progress: ProgressLogger,
    *,
    total: int,
    done: int,
    failed: int,
    current: str,
    label: str = "current",
) -> None:
    detail = f"failed={failed}, {label}={current}"
    if counter is not None:
        counter.update(done, detail=detail)
        return

    progress.detail(
        f"progress: done={done}/{total}, "
        f"failed={failed}, remaining={total - done - failed}, {label}={current}"
    )
