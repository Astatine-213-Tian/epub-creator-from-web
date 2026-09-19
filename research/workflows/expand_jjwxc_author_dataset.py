#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import html
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = REPO_ROOT.parent
if str(PRODUCTION_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_ROOT))

from src.runtime.paths import dataset_txt_output_path, safe_path_name
from src.metadata.jjwxc import JjwxcTypeMetadata, parse_article_type
from src.crawler.search import build_previews, search_all
from src.crawler.search.orchestrator import normalize_query_text


BASE_URL = "https://www.jjwxc.net/"
DEFAULT_RANK_URL = (
    "https://www.jjwxc.net/bookbase.php?s_typeid=1&version=1&fw=0&yc=0&xx=2"
    "&mainview=0&sd=0&lx=0&bq=-1&submit=%C9%B8%D1%A1"
    "&sign=rf0gab7q72448caeb72a49216791c081b9a80c3a"
    "&time=1783631184590&jsver=20260522"
)
AUTHOR_ALIASES = {"顾雪柔": "非天夜翔"}
DEFAULT_SEARCH_PROVIDERS = "mgsf,quanben"


@dataclass(frozen=True)
class RankedBook:
    source_rank: int
    page: int
    page_position: int
    author: str
    author_id: str
    title: str
    novel_id: str
    jjwxc_url: str
    article_type: str
    status: str
    word_count: int
    points: int
    published_at: str


@dataclass(frozen=True)
class AuthorBook:
    author: str
    author_id: str
    title: str
    novel_id: str
    jjwxc_url: str
    article_type: str
    status: str
    word_count: int
    points: int
    published_at: str
    source_parser: str = ""
    source_url: str = ""
    source_title: str = ""
    source_author: str = ""
    source_chapter_count: int | None = None


@dataclass
class AuthorPlan:
    author: str
    author_id: str
    priority_rank: int
    existing_books: int
    pure_love_books: int
    selected_books: list[AuthorBook]
    needed_books: int = 0
    status: str = "selected"
    error: str = ""


@dataclass
class BookIngestEntry:
    author: str
    title: str
    novel_id: str
    jjwxc_url: str
    article_type: str
    time_area: str
    genre: str
    points: int
    word_count: int
    status: str = "pending"
    error: str = ""
    search_providers: list[str] = field(default_factory=list)
    source_parser: str = ""
    source_url: str = ""
    source_title: str = ""
    source_author: str = ""
    chapter_count: int | None = None
    epub_path: str = ""
    txt_path: str = ""
    command: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


def canonical_author(author: str) -> str:
    return AUTHOR_ALIASES.get(author.strip(), author.strip())


def fetch_html(url: str, *, timeout: int = 30) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    return urlopen(req, timeout=timeout).read().decode("gb18030", "replace")


def page_url(rank_url: str, page: int) -> str:
    parsed = urlparse(rank_url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("sortType", "0")
    query.setdefault("collectiontypes", "")
    query.setdefault("searchkeywords", "")
    query.setdefault("isfinish", "0")
    query["page"] = str(page)
    return urlunparse(parsed._replace(query=urlencode(query)))


def strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", "\t", value)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def extract_cells(row_html: str) -> list[str]:
    return [strip_tags(cell) for cell in re.findall(r"<td\b[^>]*>(.*?)</td>", row_html, re.I | re.S)]


def parse_int(value: str) -> int:
    digits = re.sub(r"[^\d]", "", value or "")
    return int(digits) if digits else 0


def clean_title(value: str) -> str:
    return re.sub(r"^〖[^〗]*〗\s*", "", value or "").strip()


def parse_provider_names(value: str) -> list[str] | None:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or any(name.lower() == "all" for name in names):
        return None
    return names


def parse_provider_groups(value: str) -> list[list[str] | None]:
    groups: list[list[str] | None] = []
    for raw_group in value.split(";"):
        raw_group = raw_group.strip()
        if not raw_group:
            continue
        groups.append(parse_provider_names(raw_group))
    return groups or [parse_provider_names(DEFAULT_SEARCH_PROVIDERS)]


def parse_rank_page(rank_url: str, page: int) -> list[RankedBook]:
    text = fetch_html(page_url(rank_url, page))
    books: list[RankedBook] = []
    for row in re.findall(r"<tr\b.*?</tr>", text, re.I | re.S):
        if "onebook.php?novelid=" not in row:
            continue
        cells = extract_cells(row)
        if len(cells) < 7:
            continue
        author_match = re.search(r"oneauthor\.php\?authorid=(\d+)", row)
        novel_match = re.search(r"onebook\.php\?novelid=(\d+)", row)
        if not author_match or not novel_match:
            continue
        novel_id = novel_match.group(1)
        author = canonical_author(cells[0])
        books.append(
            RankedBook(
                source_rank=0,
                page=page,
                page_position=len(books) + 1,
                author=author,
                author_id=author_match.group(1),
                title=clean_title(cells[1]),
                novel_id=novel_id,
                jjwxc_url=urljoin(BASE_URL, f"onebook.php?novelid={novel_id}"),
                article_type=cells[2],
                status=cells[3],
                word_count=parse_int(cells[4]),
                points=parse_int(cells[5]),
                published_at=cells[6],
            )
        )
    return books


def parse_author_page(author: str, author_id: str, *, min_word_count: int) -> list[AuthorBook]:
    text = fetch_html(urljoin(BASE_URL, f"oneauthor.php?authorid={author_id}"))
    books: dict[str, AuthorBook] = {}
    for row in re.findall(r"<tr\b.*?</tr>", text, re.I | re.S):
        if "onebook.php?novelid=" not in row:
            continue
        cells = extract_cells(row)
        if len(cells) < 6:
            continue
        novel_match = re.search(r"onebook\.php\?novelid=(\d+)", row)
        if not novel_match:
            continue
        article_type = cells[1]
        title = clean_title(cells[0])
        word_count = parse_int(cells[3])
        if "-纯爱-" not in article_type:
            continue
        if word_count < min_word_count:
            continue
        if not title or "[锁]" in title or "锁定" in title:
            continue
        novel_id = novel_match.group(1)
        books[novel_id] = AuthorBook(
            author=author,
            author_id=author_id,
            title=title,
            novel_id=novel_id,
            jjwxc_url=urljoin(BASE_URL, f"onebook.php?novelid={novel_id}"),
            article_type=article_type,
            status=cells[2],
            word_count=word_count,
            points=parse_int(cells[4]),
            published_at=cells[5] if len(cells) > 5 else "",
        )
    return sorted(books.values(), key=lambda item: (-item.points, -item.word_count, item.title))


def load_existing_counts(dataset_root: Path) -> tuple[dict[str, int], set[tuple[str, str]]]:
    manifest = dataset_root / "dataset_manifest.json"
    if not manifest.exists():
        return {}, set()
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else raw.get("books", [])
    counts: dict[str, int] = {}
    titles: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        author = canonical_author(str(row.get("author") or ""))
        title = str(row.get("title") or "")
        if not author or not title:
            continue
        counts[author] = counts.get(author, 0) + 1
        titles.add((author, title))
    return counts, titles


def load_failed_titles(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    blocked_statuses = {
        "no_search_results",
        "no_exact_author_title_match",
        "failed",
        "failed_ingest",
        "failed_missing_output",
        "skipped_partial_existing",
    }
    failed: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("status") not in blocked_statuses:
            continue
        author = canonical_author(str(row.get("author") or ""))
        title = str(row.get("title") or "")
        if author and title:
            failed.add((author, title))
    return failed


def discover_ranked_authors(
    *,
    rank_url: str,
    max_pages: int,
    rank_workers: int,
) -> tuple[list[RankedBook], dict[str, RankedBook]]:
    seen_novels: set[str] = set()
    ranked: list[RankedBook] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=rank_workers) as executor:
        futures = {executor.submit(parse_rank_page, rank_url, page): page for page in range(max_pages)}
        for future in concurrent.futures.as_completed(futures):
            page = futures[future]
            try:
                page_books = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[rank:fail] page={page}: {type(exc).__name__}: {exc}", flush=True)
                continue
            for book in page_books:
                if book.novel_id in seen_novels:
                    continue
                seen_novels.add(book.novel_id)
                ranked.append(book)
    ranked.sort(key=lambda item: (item.page, item.page_position, -item.points))
    ranked = [
        RankedBook(**{**asdict(book), "source_rank": index})
        for index, book in enumerate(ranked, start=1)
    ]
    best_by_author: dict[str, RankedBook] = {}
    for book in ranked:
        best_by_author.setdefault(book.author, book)
    return ranked, best_by_author


def build_author_plans(
    *,
    best_by_author: dict[str, RankedBook],
    existing_counts: dict[str, int],
    existing_titles: set[tuple[str, str]],
    skip_titles: set[tuple[str, str]],
    target_total_authors: int,
    books_per_author: int,
    min_word_count: int,
    author_workers: int,
    prefer_partial_authors: bool,
) -> list[AuthorPlan]:
    current_qualified = {author for author, count in existing_counts.items() if count >= books_per_author}
    needed = max(0, target_total_authors - len(current_qualified))
    candidate_books = [
        book for author, book in best_by_author.items()
        if author not in current_qualified
    ]
    if prefer_partial_authors:
        candidate_books.sort(key=lambda item: (existing_counts.get(item.author, 0) == 0, item.source_rank))
    else:
        candidate_books.sort(key=lambda item: item.source_rank)
    plans: list[AuthorPlan] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=author_workers) as executor:
        future_to_book = {
            executor.submit(parse_author_page, book.author, book.author_id, min_word_count=min_word_count): book
            for book in candidate_books
        }
        for future in concurrent.futures.as_completed(future_to_book):
            book = future_to_book[future]
            existing_book_count = existing_counts.get(book.author, 0)
            needed_books = max(0, books_per_author - existing_book_count)
            try:
                pure_books = future.result()
            except Exception as exc:  # noqa: BLE001
                plans.append(
                    AuthorPlan(
                        author=book.author,
                        author_id=book.author_id,
                        priority_rank=book.source_rank,
                        existing_books=existing_book_count,
                        pure_love_books=0,
                        selected_books=[],
                        needed_books=needed_books,
                        status="failed_author_page",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            if len(pure_books) < books_per_author:
                plans.append(
                    AuthorPlan(
                        author=book.author,
                        author_id=book.author_id,
                        priority_rank=book.source_rank,
                        existing_books=existing_book_count,
                        pure_love_books=len(pure_books),
                        selected_books=[],
                        needed_books=needed_books,
                        status="insufficient_pure_love_books",
                    )
                )
                continue
            selected: list[AuthorBook] = []
            for item in pure_books:
                if (item.author, item.title) in existing_titles or (item.author, item.title) in skip_titles:
                    continue
                selected.append(item)
                if len(selected) >= needed_books:
                    break
            if len(selected) < needed_books:
                plans.append(
                    AuthorPlan(
                        author=book.author,
                        author_id=book.author_id,
                        priority_rank=book.source_rank,
                        existing_books=existing_book_count,
                        pure_love_books=len(pure_books),
                        selected_books=selected,
                        needed_books=needed_books,
                        status="insufficient_new_candidate_books",
                    )
                )
                continue
            plans.append(
                AuthorPlan(
                    author=book.author,
                    author_id=book.author_id,
                    priority_rank=book.source_rank,
                    existing_books=existing_book_count,
                    pure_love_books=len(pure_books),
                    selected_books=selected,
                    needed_books=needed_books,
                )
            )
    plans.sort(
        key=lambda item: (
            item.status != "selected",
            0 if prefer_partial_authors and item.existing_books > 0 else 1,
            item.priority_rank,
        )
    )
    return [plan for plan in plans if plan.status == "selected"][:needed]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def discover(args: argparse.Namespace) -> int:
    existing_counts, existing_titles = load_existing_counts(args.dataset_root)
    skip_titles = load_failed_titles(args.exclude_failed_manifest)
    ranked, best_by_author = discover_ranked_authors(
        rank_url=args.rank_url,
        max_pages=args.max_pages,
        rank_workers=args.rank_workers,
    )
    plans = build_author_plans(
        best_by_author=best_by_author,
        existing_counts=existing_counts,
        existing_titles=existing_titles,
        skip_titles=skip_titles,
        target_total_authors=args.target_total_authors,
        books_per_author=args.books_per_author,
        min_word_count=args.min_word_count,
        author_workers=args.author_workers,
        prefer_partial_authors=args.prefer_partial_authors,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "ranked_books.json", [asdict(book) for book in ranked])
    write_json(args.output_dir / "selected_authors.json", [asdict(plan) for plan in plans])
    summary = {
        "existing_authors": len(existing_counts),
        "existing_qualified_authors_3plus": sum(1 for count in existing_counts.values() if count >= 3),
        "ranked_books": len(ranked),
        "ranked_authors": len(best_by_author),
        "selected_new_authors": len(plans),
        "target_total_authors": args.target_total_authors,
        "books_per_author": args.books_per_author,
        "prefer_partial_authors": args.prefer_partial_authors,
        "excluded_failed_titles": len(skip_titles),
        "selected_books": sum(len(plan.selected_books) for plan in plans),
    }
    write_json(args.output_dir / "discovery_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def metadata_for(article_type: str) -> JjwxcTypeMetadata:
    return parse_article_type(article_type, source="jjwxc-author-page") or JjwxcTypeMetadata(
        article_type,
        "近代现代",
        "爱情",
        "fallback",
    )


def final_paths(book: AuthorBook, dataset_root: Path) -> tuple[Path, Path, JjwxcTypeMetadata]:
    metadata = metadata_for(book.article_type)
    epub_path = PRODUCTION_ROOT / "books" / safe_path_name(book.author, "Unknown Author") / f"{safe_path_name(book.title, 'book')}.epub"
    txt_path = dataset_txt_output_path(
        dataset_root,
        title=book.title,
        author=book.author,
        time_area=metadata.time_area,
        genre=metadata.genre,
    )
    return epub_path, txt_path, metadata


@contextlib.contextmanager
def tee_to_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        yield handle


def select_source(
    book: AuthorBook,
    *,
    limit: int,
    providers: list[str] | None,
) -> tuple[Any | None, list[dict[str, Any]], str]:
    candidates: list[dict[str, Any]] = []
    exact: list[Any] = []
    provider_names: list[str | None] = providers or [None]
    for provider_name in provider_names:
        results = search_all(
            book.title,
            parser_name=provider_name,
            author=book.author,
            limit_per_provider=limit,
            verbose=False,
        )
        if not results:
            continue
        previews = build_previews(book.title, results, max_previews=limit, verbose=False)
        candidates.extend(
            {
                "parser": preview.parser,
                "title": preview.title,
                "author": preview.author,
                "url": preview.url,
                "chapter_count": preview.chapter_count,
                "status": preview.status,
                "source": preview.source,
            }
            for preview in previews
        )
        exact.extend(
            preview for preview in previews
            if normalize_query_text(preview.title) == normalize_query_text(book.title)
            and normalize_query_text(preview.author) == normalize_query_text(book.author)
        )
    if exact:
        return max(exact, key=lambda item: (item.chapter_count or -1, item.parser, item.url)), candidates, ""
    if not candidates:
        return None, [], "no_search_results"
    return None, candidates, "no_exact_author_title_match"


def ingest_one(book: AuthorBook, args: argparse.Namespace, index: int, total: int) -> BookIngestEntry:
    epub_path, txt_path, metadata = final_paths(book, args.dataset_root)
    entry = BookIngestEntry(
        author=book.author,
        title=book.title,
        novel_id=book.novel_id,
        jjwxc_url=book.jjwxc_url,
        article_type=book.article_type,
        time_area=metadata.time_area,
        genre=metadata.genre,
        points=book.points,
        word_count=book.word_count,
        epub_path=str(epub_path.relative_to(PRODUCTION_ROOT)),
        txt_path=str(txt_path.relative_to(PRODUCTION_ROOT)),
    )
    entry.search_providers = args.search_provider_names or ["all"]
    if epub_path.exists() and txt_path.exists():
        entry.status = "skipped_existing"
        entry.error = "epub and txt already exist"
        return entry
    if epub_path.exists() or txt_path.exists():
        entry.status = "skipped_partial_existing"
        entry.error = "one requested output already exists; not overwriting"
        return entry
    try:
        if book.source_url and book.source_parser:
            source_parser = book.source_parser
            source_url = book.source_url
            source_title = book.source_title
            source_author = book.source_author
            source_chapter_count = book.source_chapter_count
        else:
            selected, candidates, reason = select_source(
                book,
                limit=args.search_limit,
                providers=args.search_provider_names,
            )
            if selected is None:
                entry.status = reason
                entry.error = reason
                return entry
            source_parser = selected.parser
            source_url = selected.url
            source_title = selected.title
            source_author = selected.author
            source_chapter_count = selected.chapter_count
        entry.source_parser = source_parser
        entry.source_url = source_url
        entry.source_title = source_title
        entry.source_author = source_author
        entry.chapter_count = source_chapter_count
        cmd = [
            "uv",
            "run",
            "book-ingest",
            source_url,
            "--parser",
            source_parser,
            "--mode",
            "both",
            "--output",
            str(epub_path),
            "--txt-output",
            str(txt_path),
            "--dataset-root",
            str(args.dataset_root),
            "--time-area",
            metadata.time_area,
            "--genre",
            metadata.genre,
            "--headless",
            "--concurrency",
            str(args.book_concurrency),
            "--no-codex-classify",
            "--no-fetch-jjwxc",
        ]
        entry.command = cmd
        if args.dry_run:
            entry.status = "selected_dry_run"
            return entry
        log_path = args.output_dir / "logs" / f"{index:03d}_{safe_path_name(book.author, 'author')}_{safe_path_name(book.title, 'book')}.log"
        with tee_to_log(log_path) as log_file:
            log_file.write(f"### {index}/{total} {book.author} / {book.title}\n")
            log_file.write(" ".join(cmd) + "\n")
            log_file.flush()
            result = subprocess.run(
                cmd,
                cwd=PRODUCTION_ROOT,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=args.ingest_timeout,
                check=False,
            )
        if result.returncode != 0:
            entry.status = "failed_ingest"
            entry.error = f"book-ingest exited {result.returncode}; see {log_path.relative_to(REPO_ROOT)}"
            return entry
        if not epub_path.exists() or not txt_path.exists():
            entry.status = "failed_missing_output"
            entry.error = "book-ingest exited successfully but requested outputs are missing"
            return entry
        entry.status = "downloaded"
        return entry
    except Exception as exc:  # noqa: BLE001
        entry.status = "failed"
        entry.error = f"{type(exc).__name__}: {exc}"
        trace_path = args.output_dir / "logs" / f"{index:03d}_{safe_path_name(book.author, 'author')}_{safe_path_name(book.title, 'book')}.trace.log"
        trace_path.write_text(traceback.format_exc(), encoding="utf-8")
        return entry
    finally:
        entry.updated_at = time.time()


def selected_authors_path(args: argparse.Namespace) -> Path:
    path = args.selected_authors_file or args.output_dir / "selected_authors.json"
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def load_author_plans(path: Path) -> list[AuthorPlan]:
    return [
        AuthorPlan(
            **{
                **item,
                "selected_books": [AuthorBook(**book) for book in item.get("selected_books", [])],
            }
        )
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]


def load_selected_books(path: Path) -> list[AuthorBook]:
    books: list[AuthorBook] = []
    for plan in load_author_plans(path):
        books.extend(plan.selected_books)
    return books


def preflight_book_source(
    book: AuthorBook,
    *,
    provider_groups: list[list[str] | None],
    search_limit: int,
) -> tuple[AuthorBook | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for providers in provider_groups:
        selected, candidates, reason = select_source(book, limit=search_limit, providers=providers)
        attempt = {
            "author": book.author,
            "title": book.title,
            "novel_id": book.novel_id,
            "jjwxc_url": book.jjwxc_url,
            "points": book.points,
            "word_count": book.word_count,
            "search_providers": providers or ["all"],
            "candidate_count": len(candidates),
            "status": reason or "source_selected",
            "updated_at": time.time(),
        }
        if selected is not None:
            confirmed = replace(
                book,
                source_parser=selected.parser,
                source_url=selected.url,
                source_title=selected.title,
                source_author=selected.author,
                source_chapter_count=selected.chapter_count,
            )
            attempt.update(
                {
                    "source_parser": selected.parser,
                    "source_url": selected.url,
                    "source_title": selected.title,
                    "source_author": selected.author,
                    "chapter_count": selected.chapter_count,
                }
            )
            attempts.append(attempt)
            return confirmed, attempts
        attempts.append(attempt)
    return None, attempts


def preflight(args: argparse.Namespace) -> int:
    selected_path = selected_authors_path(args)
    if not selected_path.exists():
        raise SystemExit(f"missing {selected_path}; run discover first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_plan_path = args.preflight_output or args.output_dir / "selected_authors.preflight.json"
    if not source_plan_path.is_absolute():
        source_plan_path = REPO_ROOT / source_plan_path
    manifest_path = args.output_dir / "source_preflight_manifest.json"
    summary_path = args.output_dir / "source_preflight_summary.json"

    existing_counts, existing_titles = load_existing_counts(args.dataset_root)
    provider_groups = args.preflight_provider_name_groups
    output_plans: list[AuthorPlan] = []
    attempts: list[dict[str, Any]] = []

    plans = load_author_plans(selected_path)
    print(
        f"[preflight] authors={len(plans)} provider_groups={args.preflight_provider_groups} "
        f"candidate_books_per_author={args.candidate_books_per_author}",
        flush=True,
    )
    for index, plan in enumerate(plans, start=1):
        existing_book_count = existing_counts.get(plan.author, plan.existing_books)
        needed_books = max(0, args.books_per_author - existing_book_count)
        if needed_books <= 0:
            output_plans.append(
                AuthorPlan(
                    author=plan.author,
                    author_id=plan.author_id,
                    priority_rank=plan.priority_rank,
                    existing_books=existing_book_count,
                    pure_love_books=plan.pure_love_books,
                    selected_books=[],
                    needed_books=0,
                    status="already_qualified",
                )
            )
            continue
        try:
            pure_books = parse_author_page(plan.author, plan.author_id, min_word_count=args.min_word_count)
        except Exception as exc:  # noqa: BLE001
            output_plans.append(
                AuthorPlan(
                    author=plan.author,
                    author_id=plan.author_id,
                    priority_rank=plan.priority_rank,
                    existing_books=existing_book_count,
                    pure_love_books=0,
                    selected_books=[],
                    needed_books=needed_books,
                    status="failed_author_page",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            write_json(source_plan_path, [asdict(item) for item in output_plans])
            continue
        candidates = [book for book in pure_books if (book.author, book.title) not in existing_titles]
        candidate_pool = candidates[: args.candidate_books_per_author]
        selected_books: list[AuthorBook] = []
        selected_titles: set[tuple[str, str]] = set()
        checked = 0
        for providers in provider_groups:
            if len(selected_books) >= needed_books:
                break
            provider_label = ",".join(providers) if providers else "all"
            print(
                f"[preflight:providers] {index}/{len(plans)} {plan.author}: "
                f"{provider_label} across {len(candidate_pool)} candidate(s)",
                flush=True,
            )
            for book in candidate_pool:
                if len(selected_books) >= needed_books:
                    break
                if (book.author, book.title) in selected_titles:
                    continue
                checked += 1
                confirmed, book_attempts = preflight_book_source(
                    book,
                    provider_groups=[providers],
                    search_limit=args.search_limit,
                )
                attempts.extend(book_attempts)
                write_json(manifest_path, attempts)
                if confirmed is not None:
                    selected_books.append(confirmed)
                    selected_titles.add((confirmed.author, confirmed.title))
                    print(
                        f"[source] {index}/{len(plans)} {plan.author}: "
                        f"{len(selected_books)}/{needed_books} {confirmed.title} -> {confirmed.source_parser}",
                        flush=True,
                    )
        status = "source_ready" if len(selected_books) >= needed_books else "insufficient_source_matches"
        error = "" if status == "source_ready" else f"confirmed {len(selected_books)}/{needed_books} after {checked} candidate book(s)"
        output_plans.append(
            AuthorPlan(
                author=plan.author,
                author_id=plan.author_id,
                priority_rank=plan.priority_rank,
                existing_books=existing_book_count,
                pure_love_books=len(pure_books),
                selected_books=selected_books[:needed_books],
                needed_books=needed_books,
                status=status,
                error=error,
            )
        )
        write_json(source_plan_path, [asdict(item) for item in output_plans])
        print(
            f"[preflight:author] {index}/{len(plans)} {plan.author}: {status} "
            f"{len(selected_books)}/{needed_books}",
            flush=True,
        )
    ready_plans = [plan for plan in output_plans if plan.status == "source_ready"]
    summary = {
        "input": str(selected_path.relative_to(REPO_ROOT) if selected_path.is_relative_to(REPO_ROOT) else selected_path),
        "source_plan": str(source_plan_path.relative_to(REPO_ROOT) if source_plan_path.is_relative_to(REPO_ROOT) else source_plan_path),
        "authors": len(output_plans),
        "source_ready_authors": len(ready_plans),
        "selected_books": sum(len(plan.selected_books) for plan in ready_plans),
        "provider_groups": [group or ["all"] for group in provider_groups],
        "attempts": len(attempts),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def source_discover(args: argparse.Namespace) -> int:
    existing_counts, existing_titles = load_existing_counts(args.dataset_root)
    skip_titles = load_failed_titles(args.exclude_failed_manifest)
    ranked, best_by_author = discover_ranked_authors(
        rank_url=args.rank_url,
        max_pages=args.max_pages,
        rank_workers=args.rank_workers,
    )
    current_qualified = {author for author, count in existing_counts.items() if count >= args.books_per_author}
    needed_authors = max(0, args.target_total_authors - len(current_qualified))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_plan_path = args.preflight_output or args.output_dir / "selected_authors.source_ready.json"
    if not source_plan_path.is_absolute():
        source_plan_path = REPO_ROOT / source_plan_path
    manifest_path = args.output_dir / "source_discovery_manifest.json"
    summary_path = args.output_dir / "source_discovery_summary.json"
    provider_groups = args.preflight_provider_name_groups

    candidate_books = [
        book for author, book in best_by_author.items()
        if author not in current_qualified
        and (not args.new_authors_only or existing_counts.get(author, 0) == 0)
    ]
    candidate_books.sort(key=lambda item: item.source_rank)

    ready_plans: list[AuthorPlan] = []
    processed_plans: list[AuthorPlan] = []
    attempts: list[dict[str, Any]] = []
    print(
        f"[source-discover] qualified={len(current_qualified)} needed={needed_authors} "
        f"candidates={len(candidate_books)} new_only={args.new_authors_only}",
        flush=True,
    )
    for index, ranked_author in enumerate(candidate_books, start=1):
        if len(ready_plans) >= needed_authors:
            break
        author = ranked_author.author
        existing_book_count = existing_counts.get(author, 0)
        needed_books = max(0, args.books_per_author - existing_book_count)
        if needed_books <= 0:
            continue
        try:
            pure_books = parse_author_page(author, ranked_author.author_id, min_word_count=args.min_word_count)
        except Exception as exc:  # noqa: BLE001
            plan = AuthorPlan(
                author=author,
                author_id=ranked_author.author_id,
                priority_rank=ranked_author.source_rank,
                existing_books=existing_book_count,
                pure_love_books=0,
                selected_books=[],
                needed_books=needed_books,
                status="failed_author_page",
                error=f"{type(exc).__name__}: {exc}",
            )
            processed_plans.append(plan)
            continue
        candidates = [
            book for book in pure_books
            if (book.author, book.title) not in existing_titles
            and (book.author, book.title) not in skip_titles
        ]
        candidate_pool = candidates[: args.candidate_books_per_author]
        selected_books: list[AuthorBook] = []
        selected_titles: set[tuple[str, str]] = set()
        checked = 0
        print(
            f"[source-discover:author] {index}/{len(candidate_books)} {author}: "
            f"need={needed_books} candidates={len(candidate_pool)} rank={ranked_author.source_rank}",
            flush=True,
        )
        for providers in provider_groups:
            if len(selected_books) >= needed_books:
                break
            provider_label = ",".join(providers) if providers else "all"
            print(
                f"[source-discover:providers] {author}: {provider_label}",
                flush=True,
            )
            for book in candidate_pool:
                if len(selected_books) >= needed_books:
                    break
                if (book.author, book.title) in selected_titles:
                    continue
                checked += 1
                confirmed, book_attempts = preflight_book_source(
                    book,
                    provider_groups=[providers],
                    search_limit=args.search_limit,
                )
                attempts.extend(book_attempts)
                write_json(manifest_path, attempts)
                if confirmed is None:
                    continue
                selected_books.append(confirmed)
                selected_titles.add((confirmed.author, confirmed.title))
                print(
                    f"[source-ready-book] {author}: {len(selected_books)}/{needed_books} "
                    f"{confirmed.title} -> {confirmed.source_parser}",
                    flush=True,
                )
        status = "source_ready" if len(selected_books) >= needed_books else "insufficient_source_matches"
        error = "" if status == "source_ready" else f"confirmed {len(selected_books)}/{needed_books} after {checked} attempt(s)"
        plan = AuthorPlan(
            author=author,
            author_id=ranked_author.author_id,
            priority_rank=ranked_author.source_rank,
            existing_books=existing_book_count,
            pure_love_books=len(pure_books),
            selected_books=selected_books[:needed_books],
            needed_books=needed_books,
            status=status,
            error=error,
        )
        processed_plans.append(plan)
        if status == "source_ready":
            ready_plans.append(plan)
            print(
                f"[source-ready-author] {len(ready_plans)}/{needed_authors} {author}",
                flush=True,
            )
        else:
            print(f"[source-reject-author] {author}: {error}", flush=True)
        write_json(source_plan_path, [asdict(item) for item in ready_plans])
    summary = {
        "existing_authors": len(existing_counts),
        "existing_qualified_authors_3plus": len(current_qualified),
        "target_total_authors": args.target_total_authors,
        "needed_authors": needed_authors,
        "ranked_books": len(ranked),
        "ranked_authors": len(best_by_author),
        "candidate_authors": len(candidate_books),
        "processed_authors": len(processed_plans),
        "source_ready_authors": len(ready_plans),
        "selected_books": sum(len(plan.selected_books) for plan in ready_plans),
        "provider_groups": [group or ["all"] for group in provider_groups],
        "new_authors_only": args.new_authors_only,
        "attempts": len(attempts),
        "source_plan": str(source_plan_path.relative_to(REPO_ROOT) if source_plan_path.is_relative_to(REPO_ROOT) else source_plan_path),
    }
    write_json(args.output_dir / "source_discovery_processed_authors.json", [asdict(item) for item in processed_plans])
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def ingest(args: argparse.Namespace) -> int:
    selected_path = selected_authors_path(args)
    if not selected_path.exists():
        raise SystemExit(f"missing {selected_path}; run discover first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    books = load_selected_books(selected_path)
    manifest_path = args.output_dir / "ingest_manifest.json"
    existing: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        existing = {
            f"{item['author']}::{item['title']}": item
            for item in json.loads(manifest_path.read_text(encoding="utf-8"))
        }
    completed = {"downloaded", "skipped_existing", "selected_dry_run"}
    pending = [
        book for book in books
        if existing.get(f"{book.author}::{book.title}", {}).get("status") not in completed
    ]
    total = len(books)
    print(f"[ingest] total={total} pending={len(pending)} workers={args.ingest_workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.ingest_workers) as executor:
        future_to_item = {
            executor.submit(ingest_one, book, args, index, total): (index, book)
            for index, book in enumerate(pending, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_item):
            index, book = future_to_item[future]
            try:
                entry = future.result()
            except Exception as exc:  # noqa: BLE001
                entry = BookIngestEntry(
                    author=book.author,
                    title=book.title,
                    novel_id=book.novel_id,
                    jjwxc_url=book.jjwxc_url,
                    article_type=book.article_type,
                    time_area="",
                    genre="",
                    points=book.points,
                    word_count=book.word_count,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            existing[f"{entry.author}::{entry.title}"] = asdict(entry)
            write_json(manifest_path, list(existing.values()))
            print(f"[done] {index}/{len(pending)} {entry.author} / {entry.title}: {entry.status} {entry.error}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand the author-style dataset from JJWXC ranked pure-love authors.")
    parser.add_argument("command", choices=["discover", "preflight", "source-discover", "ingest", "all"])
    parser.add_argument("--rank-url", default=DEFAULT_RANK_URL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/corpus_acquisition/runs/author50_expansion"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--selected-authors-file",
        type=Path,
        help="Author plan file to preflight or ingest; defaults to output-dir/selected_authors.json.",
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        help="Where to write source-confirmed author plans; defaults to output-dir/selected_authors.preflight.json.",
    )
    parser.add_argument(
        "--exclude-failed-manifest",
        type=Path,
        help="Prior ingest manifest whose failed title attempts should be skipped during discovery.",
    )
    parser.add_argument("--target-total-authors", type=int, default=50)
    parser.add_argument("--books-per-author", type=int, default=3)
    parser.add_argument(
        "--prefer-partial-authors",
        action="store_true",
        help="Prefer ranked authors that already have 1-2 dataset books, and select only missing replacement books.",
    )
    parser.add_argument(
        "--new-authors-only",
        action="store_true",
        help="For source-discover, skip authors that already have any dataset books.",
    )
    parser.add_argument("--min-word-count", type=int, default=50_000)
    parser.add_argument("--max-pages", type=int, default=8)
    parser.add_argument("--rank-workers", type=int, default=8)
    parser.add_argument("--author-workers", type=int, default=16)
    parser.add_argument("--ingest-workers", type=int, default=6)
    parser.add_argument("--book-concurrency", type=int, default=8)
    parser.add_argument("--search-limit", type=int, default=10)
    parser.add_argument(
        "--candidate-books-per-author",
        type=int,
        default=12,
        help="Maximum non-existing JJWXC books to source-preflight per author.",
    )
    parser.add_argument(
        "--preflight-provider-groups",
        default="mgsf,quanben;xfxs",
        help="Semicolon-separated provider groups tried author-by-author during preflight.",
    )
    parser.add_argument(
        "--search-providers",
        default=DEFAULT_SEARCH_PROVIDERS,
        help="Comma-separated provider names to search during ingest, or 'all'. Defaults to fast bulk providers.",
    )
    parser.add_argument("--ingest-timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dataset_root.is_absolute():
        args.dataset_root = REPO_ROOT / args.dataset_root
    if not args.output_dir.is_absolute():
        args.output_dir = REPO_ROOT / args.output_dir
    if args.exclude_failed_manifest is not None and not args.exclude_failed_manifest.is_absolute():
        args.exclude_failed_manifest = REPO_ROOT / args.exclude_failed_manifest
    if args.selected_authors_file is not None and not args.selected_authors_file.is_absolute():
        args.selected_authors_file = REPO_ROOT / args.selected_authors_file
    if args.preflight_output is not None and not args.preflight_output.is_absolute():
        args.preflight_output = REPO_ROOT / args.preflight_output
    args.search_provider_names = parse_provider_names(args.search_providers)
    args.preflight_provider_name_groups = parse_provider_groups(args.preflight_provider_groups)

    if args.command in {"discover", "all"}:
        code = discover(args)
        if code:
            return code
    if args.command == "preflight":
        return preflight(args)
    if args.command == "source-discover":
        return source_discover(args)
    if args.command in {"ingest", "all"}:
        return ingest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
