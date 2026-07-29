from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, quote_from_bytes, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from lxml import etree
from opencc import OpenCC

from src.metadata.jjwxc import parse_article_type


OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
OPF_META = f"{{{OPF_NS}}}meta"
DEFAULT_BACKUP_DIR = Path(
    "/private/tmp/epub-creator-from-web-codex-backups"
)
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
TARGET_DC_TAGS = {
    f"{{{DC_NS}}}title",
    f"{{{DC_NS}}}creator",
    f"{{{DC_NS}}}language",
    f"{{{DC_NS}}}date",
    f"{{{DC_NS}}}source",
    f"{{{DC_NS}}}description",
    f"{{{DC_NS}}}subject",
}
TITLE_SUFFIX_RE = re.compile(
    r"(?:[（(【\[][^）)】\]]+[）)】\]]|[（(【\[]坑[）)】\]])$"
)
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
        (entry.section, entry.prefix)
        for entry in entries
        if entry.prefix is not None
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
            title=(
                package.title if candidate.locked else candidate.title
            ),
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
                _normalized_title(str(author))
                == _normalized_title(package.author)
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
        description = T2S.convert(
            "\n".join(part for part in description_parts if part)
        )
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
                if package.series_position
                and package.series_position.isdigit()
                else None
            ),
            series_verified=False,
        )


def _find_opf_member(archive: zipfile.ZipFile) -> str:
    if "EPUB/content.opf" in archive.namelist():
        return "EPUB/content.opf"
    try:
        container = etree.fromstring(archive.read("META-INF/container.xml"))
    except KeyError as error:
        raise ValueError("EPUB has no META-INF/container.xml") from error
    rootfile = container.find(f".//{{{CONTAINER_NS}}}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ValueError("EPUB container has no package rootfile")
    return str(rootfile.get("full-path"))


def _dc_texts(metadata: etree._Element, local_name: str) -> list[str]:
    return [
        (element.text or "").strip()
        for element in metadata.findall(f"{{{DC_NS}}}{local_name}")
    ]


def _package_metadata(opf_data: bytes) -> PackageMetadata:
    root = etree.fromstring(
        opf_data,
        etree.XMLParser(remove_blank_text=False, resolve_entities=False),
    )
    metadata = root.find(f"{{{OPF_NS}}}metadata")
    if metadata is None:
        raise ValueError("EPUB package has no metadata element")
    series = next(
        (
            (element.text or "").strip()
            for element in metadata.findall(OPF_META)
            if element.get("property") == "belongs-to-collection"
        ),
        "",
    )
    collection_ids = {
        element.get("id")
        for element in metadata.findall(OPF_META)
        if element.get("property") == "belongs-to-collection"
        and element.get("id")
    }
    position = next(
        (
            (element.text or "").strip()
            for element in metadata.findall(OPF_META)
            if element.get("property") == "group-position"
            and element.get("refines", "").removeprefix("#")
            in collection_ids
        ),
        "",
    )

    def first(name: str, default: str = "") -> str:
        values = _dc_texts(metadata, name)
        return values[0] if values else default

    return PackageMetadata(
        title=first("title"),
        author=first("creator"),
        language=first("language", "zh-CN"),
        date=first("date"),
        source=first("source"),
        description=first("description"),
        subjects=tuple(_dc_texts(metadata, "subject")),
        series=series or None,
        series_position=position or None,
    )


def _desired_package(
    current: PackageMetadata,
    source: SourceMetadata,
) -> PackageMetadata:
    replace_series = source.series_verified
    return PackageMetadata(
        title=source.title or current.title,
        author=source.author or current.author,
        language=source.language or current.language or "zh-CN",
        date=source.date,
        source=source.source,
        description=source.description or current.description,
        subjects=source.subjects or current.subjects,
        series=source.series if replace_series else current.series,
        series_position=(
            str(source.series_position)
            if replace_series and source.series_position is not None
            else (None if replace_series else current.series_position)
        ),
    )


def _changed_fields(
    before: PackageMetadata,
    after: PackageMetadata,
) -> list[str]:
    return [
        name
        for name in (
            "title",
            "author",
            "language",
            "date",
            "source",
            "description",
            "subjects",
            "series",
            "series_position",
        )
        if getattr(before, name) != getattr(after, name)
    ]


def _make_dc(
    local_name: str,
    text: str,
    **attributes: str,
) -> etree._Element:
    element = etree.Element(f"{{{DC_NS}}}{local_name}", **attributes)
    element.text = text
    return element


def _updated_opf(
    opf_data: bytes,
    desired: PackageMetadata,
    *,
    replace_series: bool,
    modified: str,
) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    root = etree.fromstring(opf_data, parser)
    metadata = root.find(f"{{{OPF_NS}}}metadata")
    if metadata is None:
        raise ValueError("EPUB package has no metadata element")
    collection_ids = {
        child.get("id")
        for child in metadata.findall(OPF_META)
        if child.get("property") == "belongs-to-collection" and child.get("id")
    }
    for child in list(metadata):
        if child.tag in TARGET_DC_TAGS:
            metadata.remove(child)
            continue
        if child.tag != OPF_META or not replace_series:
            continue
        if child.get("property") == "belongs-to-collection":
            metadata.remove(child)
            continue
        refines = child.get("refines", "").removeprefix("#")
        if refines in collection_ids:
            metadata.remove(child)

    modified_meta = next(
        (
            child
            for child in metadata.findall(OPF_META)
            if child.get("property") == "dcterms:modified"
        ),
        None,
    )
    if modified_meta is None:
        modified_meta = etree.Element(OPF_META, property="dcterms:modified")
        metadata.insert(0, modified_meta)
    modified_meta.text = modified

    identifier_index = next(
        (
            index
            for index, child in enumerate(metadata)
            if child.tag == f"{{{DC_NS}}}identifier"
        ),
        len(metadata) - 1,
    )
    insert_at = identifier_index + 1
    dc_elements = [
        _make_dc("title", desired.title),
        _make_dc("creator", desired.author, id="creator"),
        _make_dc("language", desired.language),
        _make_dc("date", desired.date),
        _make_dc("source", desired.source),
    ]
    if desired.description:
        dc_elements.append(_make_dc("description", desired.description))
    dc_elements.extend(
        _make_dc("subject", subject) for subject in desired.subjects
    )
    for element in dc_elements:
        metadata.insert(insert_at, element)
        insert_at += 1

    if replace_series and desired.series:
        existing_ids = {
            element.get("id")
            for element in metadata.findall(OPF_META)
            if element.get("id")
        }
        collection_id = "c01"
        suffix = 1
        while collection_id in existing_ids:
            suffix += 1
            collection_id = f"c{suffix:02d}"
        collection = etree.Element(
            OPF_META,
            property="belongs-to-collection",
            id=collection_id,
        )
        collection.text = desired.series
        metadata.append(collection)
        collection_type = etree.Element(
            OPF_META,
            refines=f"#{collection_id}",
            property="collection-type",
        )
        collection_type.text = "series"
        metadata.append(collection_type)
        group_position = etree.Element(
            OPF_META,
            refines=f"#{collection_id}",
            property="group-position",
        )
        group_position.text = desired.series_position or "1"
        metadata.append(group_position)

    etree.indent(metadata, space="  ", level=1)
    output = etree.tostring(
        root.getroottree(),
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
    )
    if opf_data.endswith(b"\n"):
        output += b"\n"
    return output


def _backup_target(path: Path, backup_dir: Path) -> Path:
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve() / "books")
    except ValueError:
        relative = Path(path.name)
    return backup_dir / relative


def _rewrite_epub_member(
    path: Path,
    member: str,
    data: bytes,
) -> None:
    with zipfile.ZipFile(path, "r") as source:
        infos = source.infolist()
        members = {info.filename: source.read(info.filename) for info in infos}
    members[member] = data
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}-metadata-",
        suffix=".epub",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temp_name)
    try:
        with zipfile.ZipFile(temporary, "w") as destination:
            for info in infos:
                destination.writestr(info, members[info.filename])
        with zipfile.ZipFile(temporary, "r") as check:
            bad_member = check.testzip()
            if bad_member:
                raise ValueError(f"Corrupt rewritten EPUB member: {bad_member}")
            etree.fromstring(check.read(member))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def enrich_epub_metadata(
    path: Path,
    *,
    apply: bool = True,
    backup_dir: Path | None = DEFAULT_BACKUP_DIR,
    overwrite_backup: bool = False,
    primary_subject: str = DEFAULT_PRIMARY_SUBJECT,
    lookup: MetadataLookup | None = None,
) -> MetadataEnrichmentReport:
    path = Path(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            opf_member = _find_opf_member(archive)
            opf_data = archive.read(opf_member)
        current = _package_metadata(opf_data)
        if not current.title or not current.author:
            return MetadataEnrichmentReport(
                path=path,
                applied=apply,
                status="unmatched",
                opf_member=opf_member,
                error="EPUB metadata must contain dc:title and dc:creator",
            )
        source = (lookup or MetadataLookup()).find(
            current,
            primary_subject=primary_subject,
        )
        if source is None:
            return MetadataEnrichmentReport(
                path=path,
                applied=apply,
                status="unmatched",
                opf_member=opf_member,
                before=current.to_dict(),
                after=current.to_dict(),
            )
        desired = _desired_package(current, source)
        changes = _changed_fields(current, desired)
        status = "enriched" if changes else "already_complete"
        report = MetadataEnrichmentReport(
            path=path,
            applied=apply,
            status=status,
            provider=source.provider,
            opf_member=opf_member,
            changed_fields=changes,
            before=current.to_dict(),
            after=desired.to_dict(),
            source_url=source.source,
            publication_date=source.date,
        )
        if not changes or not apply:
            return report
        if backup_dir is not None:
            backup = _backup_target(path, backup_dir)
            backup.parent.mkdir(parents=True, exist_ok=True)
            if overwrite_backup or not backup.exists():
                shutil.copy2(path, backup)
            report.backup = backup
        modified = (
            datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        updated = _updated_opf(
            opf_data,
            desired,
            replace_series=source.series_verified,
            modified=modified,
        )
        _rewrite_epub_member(path, opf_member, updated)
        return report
    except Exception as error:
        return MetadataEnrichmentReport(
            path=path,
            applied=apply,
            status="error",
            error=f"{type(error).__name__}: {error}",
        )
