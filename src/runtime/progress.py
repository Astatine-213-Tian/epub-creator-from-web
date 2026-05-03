from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO


_DEBUG_ENABLED = False


def configure_progress(*, debug: bool = False) -> None:
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = debug


@dataclass
class ProgressLogger:
    verbose: bool = True
    debug_enabled: bool | None = None
    stream: TextIO = sys.stderr
    _status_active: bool = field(default=False, init=False, repr=False)

    def debug_is_enabled(self) -> bool:
        return _DEBUG_ENABLED if self.debug_enabled is None else self.debug_enabled

    def can_update_status(self) -> bool:
        return (
            self.verbose
            and not self.debug_is_enabled()
            and hasattr(self.stream, "isatty")
            and self.stream.isatty()
        )

    def _clear_status_line(self) -> None:
        if self._status_active and self.can_update_status():
            print(file=self.stream, flush=True)
        self._status_active = False

    def info(self, message: str) -> None:
        if self.verbose:
            self._clear_status_line()
            print(f"[+] {message}", file=self.stream, flush=True)

    def detail(self, message: str) -> None:
        if self.verbose and self.debug_is_enabled():
            self._clear_status_line()
            print(f"    {message}", file=self.stream, flush=True)

    def section(self, title: str) -> None:
        if self.verbose:
            self._clear_status_line()
            print(f"\n== {title} ==", file=self.stream, flush=True)

    def status(self, message: str) -> None:
        if not self.verbose:
            return
        if not self.can_update_status():
            self.info(message)
            return
        print(f"\r\x1b[K[+] {message}", end="", file=self.stream, flush=True)
        self._status_active = True

    def finish_status(self, message: str) -> None:
        if not self.verbose:
            return
        if not self.can_update_status():
            self.info(message)
            return
        print(f"\r\x1b[K[+] {message}", file=self.stream, flush=True)
        self._status_active = False

    def warning(self, message: str) -> None:
        self._clear_status_line()
        print(f"[!] {message}", file=self.stream, flush=True)

    def provider(self, provider: str, message: str) -> None:
        self.info(f"{provider}: {message}")

    def provider_detail(self, provider: str, message: str) -> None:
        self.detail(f"{provider}: {message}")

    def counter(self, label: str, total: int, unit: str) -> ProgressCounter:
        return ProgressCounter(logger=self, label=label, total=total, unit=unit)


@dataclass
class ProgressCounter:
    logger: ProgressLogger
    label: str
    total: int
    unit: str
    completed: int = 0
    _bar: object | None = field(default=None, init=False, repr=False)

    def start(self, detail: str = "") -> None:
        self._start_bar()
        if self._bar is not None and not detail:
            return
        self.update(0, detail=detail)

    def update(self, completed: int | None = None, *, detail: str = "") -> None:
        old_completed = self.completed
        if completed is not None:
            self.completed = completed
        if self._bar is not None:
            delta = self.completed - old_completed
            if detail:
                self._bar.set_postfix_str(detail, refresh=False)
            if delta:
                self._bar.update(delta)
            else:
                self._bar.refresh()
            return
        message = f"{self.label}: {self.completed}/{self.total} {self.unit} complete"
        if detail:
            message = f"{message} ({detail})"
        self.logger.status(message)

    def advance(self, *, detail: str = "") -> None:
        self.update(self.completed + 1, detail=detail)

    def finish(self, message: str) -> None:
        if self._bar is not None:
            if self.completed < self.total:
                self._bar.update(self.total - self.completed)
                self.completed = self.total
            self.close()
            self.logger.info(message)
            return
        self.logger.finish_status(message)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _start_bar(self) -> None:
        if not self.logger.verbose or self.logger.debug_is_enabled() or self._bar is not None:
            return
        try:
            from tqdm import tqdm
        except Exception:
            return

        self._bar = tqdm(
            total=self.total,
            desc=self.label,
            unit=self.unit.replace("(s)", ""),
            file=self.logger.stream,
            dynamic_ncols=True,
            leave=True,
        )
