from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chapter:
    title: str
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class Volume:
    title: str
    chapters: list[Chapter] = field(default_factory=list)
