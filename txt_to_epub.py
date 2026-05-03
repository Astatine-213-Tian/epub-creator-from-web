#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from booklib import Chapter, Volume, write_epub


TITLE_RE = re.compile(r"^《(?P<title>.+?)》作者：(?P<author>.+?)$")
CHAPTER_RE = re.compile(r"^第(?P<num>\d+)章(?P<rest>.*)$")
FANWAI_START = 109


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-16")
    except UnicodeError:
        return path.read_text(encoding="utf-8")


def _shift_chapter_title(title: str) -> str:
    match = CHAPTER_RE.match(title)
    if not match:
        return title

    num = int(match.group("num"))
    rest = match.group("rest")
    if num >= 131:
        num -= 5
    return f"第{num}章{rest}"


def parse_txt(path: Path) -> tuple[str, str, list[Volume], list[str]]:
    text = _read_text(path)
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"empty txt file: {path}")

    title = path.stem
    author = ""
    match = TITLE_RE.match(lines[0].strip())
    if match:
        title = match.group("title").strip()
        author = match.group("author").strip()

    intro_lines: list[str] = []
    volumes: list[Volume] = []
    current_volume: Volume | None = None
    current_chapter: Chapter | None = None
    fanwai_volume: Volume | None = None

    for raw_line in lines[1:]:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("第一卷"):
            suffix = line.split("：", 1)[-1].strip()
            current_volume = Volume(title=suffix)
            volumes.append(current_volume)
            current_chapter = None
            continue

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            chapter_num = int(chapter_match.group("num"))
            if chapter_num >= FANWAI_START:
                if fanwai_volume is None:
                    fanwai_volume = Volume(title="番外")
                    volumes.append(fanwai_volume)
                current_volume = fanwai_volume
            if current_volume is None:
                current_volume = Volume(title="")
                volumes.append(current_volume)
            title_text = _shift_chapter_title(line)
            current_chapter = Chapter(title=title_text)
            current_volume.chapters.append(current_chapter)
            continue

        if current_chapter is None:
            intro_lines.append(line)
        else:
            current_chapter.paragraphs.append(line)

    return title, author, volumes, intro_lines


def build_epub(source: Path, out_path: Path) -> None:
    title, author, volumes, intro_lines = parse_txt(source)
    identifier = hashlib.sha1(f"{source.resolve()}::{title}::{author}".encode("utf-8")).hexdigest()
    write_epub(
        identifier=f"txt-{identifier[:16]}",
        title=title,
        author=author,
        volumes=volumes,
        out_path=out_path,
        intro_paragraphs=intro_lines or None,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert a chaptered TXT novel into an EPUB.")
    p.add_argument("source", type=Path, help="Input TXT file")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output EPUB path")
    args = p.parse_args(argv)

    source = args.source
    if not source.exists():
        print(f"[!] missing source file: {source}", file=sys.stderr)
        return 1

    title, _, _, _ = parse_txt(source)
    out_path = args.output
    if out_path is None:
        out_dir = Path(__file__).resolve().parent / "epub"
        out_path = out_dir / f"{title}.epub"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    build_epub(source, out_path)
    print(f"[+] wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
