"""Look up authoritative book metadata, independent of any output destination."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, quote_from_bytes, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from opencc import OpenCC

from src.metadata.jjwxc import parse_article_type

DEFAULT_PRIMARY_SUBJECT = "耽美"


IGNORED_JJWXC_SERIES = {
    "内些冗长拖沓的裹脚布",
    "内容有些拖沓的裹脚布",
    "角落里的事",
    "角落里的那些事",
    "无所属系列",
}


GENERIC_JJWXC_SECTIONS = {
    "正在连载",
    "言情",
    "纯爱",
    "耽美",
    "百合",
    "无CP",
    "全年龄向",
    "短篇",
    "暂时休息的文案们",
    "已经锁了的文",
    "评论文章",
}


TITLE_SUFFIX_RE = re.compile(r"(?:[（(【\[][^）)】\]]+[）)】\]]|[（(【\[]坑[）)】\]])$")


MASKED_TITLE_RE = re.compile(r"[*＊]{2,}")


JJWXC_BOOK_LINK_RE = re.compile(r"jump_book2\((\d+)")


JJWXC_TITLE_RE = re.compile(r"《([^》]+)》")


JJWXC_SERIES_RE = re.compile(r"〖([^〗]+)〗")


JJWXC_SECTION_RE = re.compile(r"【([^】]+)】")


JJWXC_DATE_RE = re.compile(r"发表时间：(\d{4}-\d{2}-\d{2})")


JJWXC_TYPE_RE = re.compile(r"类型:([^\n]+?)(?:发表时间|进度|$)")


T2S = OpenCC("t2s")


S2T = OpenCC("s2t")


@dataclass(frozen=True)
class PackageMetadata:
    title: str
    author: str
    language: str
    date: str
    source: str
    description: str
    subjects: tuple[str, ...]
    series: str | None
    series_position: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceMetadata:
    provider: str
    title: str
    author: str
    language: str
    date: str
    source: str
    description: str
    subjects: tuple[str, ...]
    series: str | None = None
    series_position: int | None = None
    series_verified: bool = False


@dataclass
class MetadataEnrichmentReport:
    path: Path
    applied: bool
    status: str
    provider: str | None = None
    opf_member: str | None = None
    changed_fields: list[str] = field(default_factory=list)
    before: dict[str, object] = field(default_factory=dict)
    after: dict[str, object] = field(default_factory=dict)
    source_url: str | None = None
    publication_date: str | None = None
    backup: Path | None = None
    error: str | None = None

    @property
    def changed(self) -> bool:
        return bool(self.changed_fields)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "applied": self.applied,
            "status": self.status,
            "provider": self.provider,
            "opf_member": self.opf_member,
            "changed_fields": self.changed_fields,
            "before": self.before,
            "after": self.after,
            "source_url": self.source_url,
            "publication_date": self.publication_date,
            "backup": str(self.backup) if self.backup else None,
            "error": self.error,
        }

    def format_text(self) -> str:
        provider = f" via {self.provider}" if self.provider else ""
        line = f"METADATA {self.path}: {self.status}{provider}"
        if self.changed_fields:
            mode = "applied" if self.applied else "proposed"
            line += f"; {mode}={','.join(self.changed_fields)}"
        if self.publication_date:
            line += f"; date={self.publication_date}"
        if self.error:
            line += f"; error={self.error}"
        return line


@dataclass(frozen=True)
class _JjwxcCatalogEntry:
    title: str
    novel_id: str
    date: str
    raw_type: str
    locked: bool
    prefix: str | None
    section: str | None


@dataclass(frozen=True)
class _JjwxcCandidate:
    title: str
    novel_id: str
    date: str
    raw_type: str
    locked: bool
    series: str | None
    series_position: int | None


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _normalized_title(value: str) -> str:
    simplified = T2S.convert(value or "")
    return re.sub(r"[\W_]+", "", simplified, flags=re.UNICODE).casefold()


def _title_variants(value: str) -> set[str]:
    simplified = T2S.convert(value or "").strip()
    variants = {_normalized_title(simplified)}
    without_suffix = TITLE_SUFFIX_RE.sub("", simplified).strip()
    if without_suffix:
        variants.add(_normalized_title(without_suffix))
    return {variant for variant in variants if variant}


def _masked_title_matches(source_title: str, local_title: str) -> bool:
    simplified_source = T2S.convert(source_title)
    if not MASKED_TITLE_RE.search(simplified_source):
        return False
    pieces = MASKED_TITLE_RE.split(simplified_source)
    pattern = "^" + ".+".join(re.escape(piece) for piece in pieces) + "$"
    return any(
        re.fullmatch(pattern, T2S.convert(variant)) is not None
        for variant in {local_title, TITLE_SUFFIX_RE.sub("", local_title)}
    )


def _title_matches(source_title: str, local_title: str) -> bool:
    if _title_variants(source_title) & _title_variants(local_title):
        return True
    return _masked_title_matches(source_title, local_title)


def _clean_lines(text: str) -> str:
    lines: list[str] = []
    for raw_line in text.replace("\r", "\n").splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", raw_line).strip()
        if line and line not in {"简介", "（查看全部）"}:
            lines.append(line)
    return "\n".join(lines)


def _response_text(response: requests.Response, encoding: str) -> str:
    response.raise_for_status()
    return response.content.decode(encoding, "replace")


def _author_id_from_href(href: str) -> str:
    query = parse_qs(urlparse(href).query)
    return query.get("authorid", [""])[0]


def parse_jjwxc_author_search(html: str, author: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    matches = {
        _author_id_from_href(str(link.get("href", "")))
        for link in soup.select("a[href*='authorid=']")
        if _normalized_title(link.get_text(" ", strip=True))
        == _normalized_title(author)
    }
    matches.discard("")
    if len(matches) == 1:
        return next(iter(matches))
    return None


def parse_jjwxc_author_catalog(html: str) -> list[_JjwxcCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    entries: list[_JjwxcCatalogEntry] = []
    for table in soup.select("table.author.novel"):
        text = " ".join(table.stripped_strings)
        link = table.find("a", onclick=JJWXC_BOOK_LINK_RE)
        if not isinstance(link, Tag):
            continue
        link_text = link.get_text(" ", strip=True)
        id_match = JJWXC_BOOK_LINK_RE.search(str(link.get("onclick", "")))
        title_match = JJWXC_TITLE_RE.search(link_text)
        date_match = JJWXC_DATE_RE.search(text)
        if not id_match or not title_match or not date_match:
            continue
        type_match = JJWXC_TYPE_RE.search(text)
        prefix_match = JJWXC_SERIES_RE.search(link_text)
        prefix = prefix_match.group(1).strip() if prefix_match else None
        if prefix in IGNORED_JJWXC_SERIES:
            prefix = None
        section_table = table.find_previous("table", class_="series")
        section_match = (
            JJWXC_SECTION_RE.search(section_table.get_text(" ", strip=True))
            if isinstance(section_table, Tag)
            else None
        )
        section = section_match.group(1).strip() if section_match else None
        source_title = title_match.group(1).strip()
        entries.append(
            _JjwxcCatalogEntry(
                title=source_title,
                novel_id=id_match.group(1),
                date=date_match.group(1),
                raw_type=type_match.group(1).strip() if type_match else "",
                locked=bool(MASKED_TITLE_RE.search(source_title) or "[锁]" in text),
                prefix=prefix,
                section=section,
            )
        )

    prefix_counts = Counter(
        (entry.section, entry.prefix) for entry in entries if entry.prefix is not None
    )
    positions: Counter[str] = Counter()
    candidates: list[_JjwxcCandidate] = []
    for entry in entries:
        if entry.section in IGNORED_JJWXC_SERIES:
            series = None
        else:
            section_series = (
                entry.section
                if entry.section and entry.section not in GENERIC_JJWXC_SECTIONS
                else None
            )
            repeated_prefix = (
                entry.prefix is not None
                and prefix_counts[(entry.section, entry.prefix)] > 1
            )
            series = entry.prefix if repeated_prefix else section_series or entry.prefix
        position: int | None = None
        if series:
            positions[series] += 1
            position = positions[series]
        candidates.append(
            _JjwxcCandidate(
                title=entry.title,
                novel_id=entry.novel_id,
                date=entry.date,
                raw_type=entry.raw_type,
                locked=entry.locked,
                series=series,
                series_position=position,
            )
        )
    return candidates


def parse_jjwxc_description(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    whole = soup.select_one("#novelintro_whole")
    intro = soup.select_one("#novelintro")
    description_node = whole if whole is not None else intro
    description = _clean_lines(
        description_node.get_text("\n", strip=True)
        if description_node is not None
        else ""
    )
    one_line = ""
    for item in soup.select("li"):
        text = _clean_lines(item.get_text("\n", strip=True))
        if text.startswith("一句话简介："):
            one_line = text
            break
    if one_line and one_line not in description:
        description = f"{one_line}\n{description}" if description else one_line
    return description


class MetadataLookup:
    def __init__(
        self,
        *,
        author_ids: Mapping[str, str] | None = None,
        kadokado: bool = True,
        timeout: int = 20,
        session: requests.Session | None = None,
    ) -> None:
        self.author_ids = dict(author_ids or {})
        self.kadokado = kadokado
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent",
            "Mozilla/5.0 epub-creator-from-web metadata-enricher",
        )
        self._author_id_cache: dict[str, str | None] = {}
        self._catalog_cache: dict[str, list[_JjwxcCandidate]] = {}

    def find(
        self,
        package: PackageMetadata,
        *,
        primary_subject: str = DEFAULT_PRIMARY_SUBJECT,
    ) -> SourceMetadata | None:
        jjwxc = self._find_jjwxc(package, primary_subject=primary_subject)
        if jjwxc is not None:
            return jjwxc
        if not self.kadokado:
            return None
        return self._find_kadokado(package)

    def _find_jjwxc_author_id(self, author: str) -> str | None:
        if author in self.author_ids:
            return self.author_ids[author]
        if author in self._author_id_cache:
            return self._author_id_cache[author]
        encoded = quote_from_bytes(author.encode("gb18030"))
        url = f"https://www.jjwxc.net/search.php?kw={encoded}&t=2"
        html = _response_text(
            self.session.get(url, timeout=self.timeout),
            "gb18030",
        )
        author_id = parse_jjwxc_author_search(html, author)
        self._author_id_cache[author] = author_id
        return author_id

    def _jjwxc_catalog(self, author_id: str) -> list[_JjwxcCandidate]:
        if author_id not in self._catalog_cache:
            url = f"https://m.jjwxc.net/wapauthor/{author_id}"
            html = _response_text(
                self.session.get(url, timeout=self.timeout),
                "gb18030",
            )
            self._catalog_cache[author_id] = parse_jjwxc_author_catalog(html)
        return self._catalog_cache[author_id]

    def _find_jjwxc(
        self,
        package: PackageMetadata,
        *,
        primary_subject: str,
    ) -> SourceMetadata | None:
        author_id = self._find_jjwxc_author_id(package.author)
        if not author_id:
            return None
        matches = [
            candidate
            for candidate in self._jjwxc_catalog(author_id)
            if _title_matches(candidate.title, package.title)
        ]
        if len(matches) != 1:
            return None
        candidate = matches[0]
        source = f"https://m.jjwxc.net/book2/{candidate.novel_id}"
        description = package.description
        if not candidate.locked:
            book_html = _response_text(
                self.session.get(source, timeout=self.timeout),
                "gb18030",
            )
            description = parse_jjwxc_description(book_html) or description
        parsed_type = parse_article_type(candidate.raw_type)
        subjects = package.subjects
        if parsed_type and parsed_type.genre and parsed_type.time_area:
            subjects = _unique(
                [
                    primary_subject,
                    parsed_type.genre,
                    parsed_type.time_area,
                ]
            )
        if "耽美" in subjects:
            subjects = tuple(subject for subject in subjects if subject != "爱情")
        return SourceMetadata(
            provider="jinjiang",
            title=(package.title if candidate.locked else candidate.title),
            author=package.author,
            language="zh-CN",
            date=candidate.date,
            source=source,
            description=description,
            subjects=subjects,
            series=candidate.series,
            series_position=candidate.series_position,
            series_verified=True,
        )

    def _find_kadokado(
        self,
        package: PackageMetadata,
    ) -> SourceMetadata | None:
        response = self.session.get(
            "https://api.kadokado.com.tw/v3/search",
            params={
                "keyword": S2T.convert(package.title),
                "current": 1,
                "limit": 100,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        candidates = response.json().get("data", [])
        matches = [
            candidate
            for candidate in candidates
            if _title_matches(
                str(candidate.get("displayName", "")),
                package.title,
            )
            and any(
                _normalized_title(str(author)) == _normalized_title(package.author)
                for author in (
                    candidate.get("authorsDisplayNames")
                    or [candidate.get("ownerDisplayName", "")]
                )
            )
        ]
        if len(matches) != 1:
            return None
        title_id = str(matches[0]["id"])
        detail_response = self.session.get(
            f"https://api.kadokado.com.tw/v2/titles/{title_id}",
            timeout=self.timeout,
        )
        detail_response.raise_for_status()
        details = detail_response.json()
        collection_response = self.session.get(
            f"https://api.kadokado.com.tw/v3/title/{title_id}/collection",
            timeout=self.timeout,
        )
        collection_response.raise_for_status()
        listing_dates = [
            str(chapter.get("listingFrom", ""))[:10]
            for collection in collection_response.json()
            for chapter in collection.get("chapters", [])
            if chapter.get("listingFrom")
        ]
        if not listing_dates:
            return None
        description_parts = [
            str(details.get("oneLineIntro", "")).strip(),
            str(details.get("logline", "")).strip(),
        ]
        description = T2S.convert("\n".join(part for part in description_parts if part))
        return SourceMetadata(
            provider="kadokado",
            title=T2S.convert(
                str(details.get("displayName") or matches[0]["displayName"])
            ),
            author=package.author,
            language="zh-CN",
            date=min(listing_dates),
            source=f"https://www.kadokado.com.tw/book/{title_id}",
            description=description or package.description,
            subjects=package.subjects,
            series=package.series,
            series_position=(
                int(package.series_position)
                if package.series_position and package.series_position.isdigit()
                else None
            ),
            series_verified=False,
        )
