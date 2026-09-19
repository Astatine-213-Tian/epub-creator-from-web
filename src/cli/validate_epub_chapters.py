"""Validate EPUB chapter numbering and navigation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.epub.validate import validate_epub


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("books")],
        help="EPUB files or directories containing EPUB files",
    )
    args = parser.parse_args(argv)

    epub_paths: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            epub_paths.extend(sorted(path.glob("*.epub")))
        else:
            epub_paths.append(path)

    if not epub_paths:
        print("No EPUB files found.", file=sys.stderr)
        return 2

    failed = 0
    for path in epub_paths:
        issues, chapter_count = validate_epub(path)
        if issues:
            failed += 1
            print(f"FAIL {path} ({chapter_count} numbered chapter entries)")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"OK   {path} ({chapter_count} numbered chapter entries)")

    print(f"\nValidated {len(epub_paths)} EPUB file(s); {failed} file(s) with issues.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
