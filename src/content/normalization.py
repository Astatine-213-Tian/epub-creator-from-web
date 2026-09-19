"""Source text and XHTML cleanup; never applied by presentation renderers."""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable
from xml.etree import ElementTree as ET

from src.metadata.catalog import MetadataEnrichmentReport

HAN_CLASS = (
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef\U0002f800-\U0002fa1f"
)


HAN_RE = re.compile(f"[{HAN_CLASS}]")


HAN_INTERNAL_SPACE_RE = re.compile(
    f"(?<=[{HAN_CLASS}])[\\t \\u00a0\\u3000]+(?=[{HAN_CLASS}])"
)


HAN_INTERNAL_BACKSLASH_RE = re.compile(f"(?<=[{HAN_CLASS}])\\\\+(?=[{HAN_CLASS}])")


SPACE_BEFORE_CJK_PUNCT_RE = re.compile(r"[\t \u00a0\u3000]+(?=[，。！？；：、])")


SPACE_BEFORE_CLOSING_QUOTE_RE = re.compile(
    rf"(?<=[{HAN_CLASS}A-Za-z0-9，。！？；：、…])"
    r"[\t \u00a0\u3000]+(?=[”）》】」』])"
)


SPACE_AFTER_CJK_PUNCT_RE = re.compile(r"(?<=[，。！？；：、])[\t \u00a0\u3000]+")


SPACE_AFTER_CJK_OPEN_RE = re.compile(
    rf"(?<=[“《【「『])[\t \u00a0\u3000]+(?=[{HAN_CLASS}A-Za-z0-9])"
)


CJK_SPACE_BEFORE_OPEN_RE = re.compile(
    f"(?<=[{HAN_CLASS}])[\t \u00a0\u3000]+(?=[“‘《【「『])"
)


ORDINAL_RE = re.compile(
    r"第\s*(?:\d+|[零〇一二三四五六七八九十百千万两]+)\s*[章回节卷部]"
)


THOUSANDS_RE = re.compile(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)")


FILENAME_EXTENSION_RE = re.compile(
    r"(?:jpe?g|png|gif|webp|avi|mp4|mp3|txt|pdf|docx?|xlsx?|zip|rar|epub|html?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


ASCII_ELLIPSIS_RE = re.compile(r"\.{3,}")


ASCII_DASH_RE = re.compile(r"-{2,}")


ASCII_PAREN_WITH_HAN_RE = re.compile(rf"\((?P<body>[^()\n]*[{HAN_CLASS}][^()\n]*)\)")


ASCII_BRACKET_WITH_HAN_RE = re.compile(
    rf"\[(?P<body>[^\[\]\n]*[{HAN_CLASS}][^\[\]\n]*)\]"
)


REPEATED_CHINESE_COMMA_RE = re.compile(r"，{2,}")


CHINESE_DIGIT_RUN_RE = re.compile(r"[零〇○一二三四五六七八九]{2,}")


RESOURCE_GROUP_AD_RE = re.compile(
    r"(?:耽美小说)?资源群\s*[:：]\s*\d{5,}",
    re.IGNORECASE,
)


AUTHOR_NOTE_MARKER_RE = re.compile(r"作者有话(?:要)?说")


AUTHOR_NOTE_HEADING_RE = re.compile(r"作者有话(?:要)?说(?=\s*[:：])")


AUTHOR_NOTE_PARAGRAPH_RE = re.compile(r"^\s*作者有话(?:要)?说(?:\s*[:：]|\s+|$)")


AUTHOR_NOTE_SIGNAL_RE = re.compile(
    r"(?:感谢大家.{0,20}(?:陪伴|支持)|"
    r"本书连载期间|个人志|预售|下本(?:书)?见|晋江同步连载)"
)


KNOWN_SPLIT_HAN_WORDS = (
    "体液",
    "性梦",
    "胯下",
    "硬挺",
    "赤裸",
    "勃起",
    "性具",
    "后庭",
    "阳具",
    "抽插",
)


KNOWN_SPLIT_HAN_FRAGMENTS = (
    ("要", "让"),
    ("是", "目前"),
    ("会", "通过"),
    ("无非", "是"),
    ("要", "去"),
    ("坚", "挺"),
    ("下", "身"),
    ("光", "裸"),
    ("臀", "股"),
    ("屁", "股"),
)


CHAPTER_LIKE_TITLE_RE = re.compile(
    r"^\s*(?:"
    r"第\s*(?:\d+|[零〇一二三四五六七八九十百千万两]+)\s*[章回节卷部]"
    r"|卷[零〇一二三四五六七八九十百千万两]+"
    r"|序言|序章|序卷|终章|终卷|尾声|后记|番外"
    r"|Chapter\s+\d+"
    r")",
    re.IGNORECASE,
)


CHAPTER_NUMBER_PREFIX_RE = re.compile(
    r"^\s*第\s*(?:\d+|[零〇一二三四五六七八九十百千万两]+)\s*章[\s　]*"
)


FANWAI_CHAPTER_NUMBER_RE = re.compile(
    CHAPTER_NUMBER_PREFIX_RE.pattern + r"(?=[^\n]*番外)"
)


FANWAI_NUMBERED_TITLE_RE = re.compile(
    r"^(?P<prefix>\s*番外)(?P<number_spacing>[\s　]*)"
    r"(?P<number>"
    r"(?:\d+)(?!\d)"
    r"|(?:[零〇一二三四五六七八九十百千万两]+)"
    r"(?![零〇一二三四五六七八九十百千万两])"
    r")"
    r"(?P<separator>[\s　·.：:]*)"
    r"(?P<title>\S.*)?$"
)


FANWAI_TRAILING_SEPARATOR_RE = re.compile(r"(?<![.·])[.·]$")


FANWAI_COLON_SEPARATOR_RE = re.compile(r"(?<=番外)[\s　]*[：:][\s　]*(?=\S)")


FANWAI_MISSING_YEAR_RE = re.compile(
    r"(?<!\d)(?P<year>(?:19|20)\d{2})[\s　]*(?=中秋(?:节)?番外)"
)


GROUPED_FANWAI_PREFIX_RE = re.compile(r"^\s*番外[\s　·.：:—–－-]*")


MIDDLE_AUTUMN_FANWAI_INTERNAL_SEPARATOR_RE = re.compile(
    r"(?P<prefix>中秋(?:节)?)[\s　·]+(?=番外)"
)


MIDDLE_AUTUMN_FANWAI_TITLE_SEPARATOR_RE = re.compile(
    r"(?P<prefix>中秋(?:节)?番外)[\s　：:]*(?=[^·\s])"
)


EPUB_OPS_NAMESPACE_DECL_RE = re.compile(
    r"xmlns:(?P<prefix>[A-Za-z_][\w.-]*)="
    r'(?P<quote>["\'])http://www\.idpf\.org/2007/ops(?P=quote)'
)


DECORATIVE_END_MARKER_RE = re.compile(
    r"^\s*(?:-{2,}|—{2,})\s*"
    r"(?P<label>[^—\-\n]{1,50}?)"
    r"\s*(?:-{2,}|—{2,})\s*$"
)


ENDING_TERMS = (
    "正文部分完",
    "正文完",
    "全文完",
    "全剧终",
    "上册完",
    "本卷完",
    "番外完",
    "终",
    "完",
)


STRUCTURAL_MARKER_RE = re.compile(
    r"^\s*(?:#\s*)?(?:"
    r"(?:第?[零〇一二三四五六七八九十百千万两]+卷|"
    r"卷[零〇一二三四五六七八九十百千万两]+)"
    r"(?:[·：: \u3000].{0,35})?"
    r"|序言|序章|序卷|终章|终卷|尾声|后记"
    r"|番外卷.{0,35}|其他番外"
    r")\s*$"
)


INTRO_BOILERPLATE_RE = re.compile(r"(?:《[^》]{1,80}》\s*)?作者\s*[:：]\s*\S+")


TOKEN_RE = re.compile(r"(<[^>]+>|&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]+);)")


PARAGRAPH_RE = re.compile(
    r"(?P<open><(?:[A-Za-z_][\w.-]*:)?p\b[^>]*>)"
    r"(?P<body>.*?)"
    r"(?P<close></(?:[A-Za-z_][\w.-]*:)?p>)",
    re.DOTALL,
)


ZH_TRANSLATION_CLASS_RE = re.compile(
    r"\bclass\s*=\s*([\"'])[^\"']*\bzh-translation\b[^\"']*\1",
    re.IGNORECASE,
)


STRAIGHT_DOUBLE_QUOTE_RE = re.compile(rf'"([^"\n]*[{HAN_CLASS}][^"\n]*)"')


STRAIGHT_SINGLE_QUOTE_RE = re.compile(
    rf"(?<![A-Za-z])'([^'\n]*[{HAN_CLASS}][^'\n]*)'(?![A-Za-z])"
)


REVERSED_SINGLE_QUOTE_PAIR_RE = re.compile(
    rf"’(?P<body>(?=[^’‘\n，。！？；：、“”]{{0,23}}"
    rf"[{HAN_CLASS}])[^’‘\n，。！？；：、“”]{{1,24}})‘"
    r"(?=[\s，。！？；：、）》】」』]|$)"
)


DIALOGUE_ATTRIBUTION_RE = re.compile(
    r"(?P<close_punct>[，。！？])“"
    r"(?P<attribution>[^“”\n]{0,40}?"
    r"(?:说道|问道|答道|笑道|反问|解释|提醒|表示|说|道|问|答|叫|喊|吼|骂)"
    r"[^“”\n]{0,12})"
    r"(?P<resume_punct>[，：])”"
)


DUPLICATE_OPEN_AFTER_COLON_RE = re.compile(r"(?<=[:：])“{2,}")


DUPLICATE_OPEN_AT_START_RE = re.compile(r"^“{2,}")


DIALOGUE_BOUNDARY_DOUBLE_RE = re.compile(
    rf"(?<=[。！？…])(?:“{{2,}}|”{{2,}})(?=[{HAN_CLASS}])"
)


DUPLICATE_CLOSE_AT_END_RE = re.compile(r"”{2,}(?=\s*$)")


AMBIGUOUS_DUPLICATE_QUOTE_RE = re.compile(r"“{2,}|”{2,}|‘{2,}|’{2,}")


ORPHAN_ATTRIBUTION_OPEN_RE = re.compile(
    r"”[^“”\n]{0,40}?"
    r"(?:说道|问道|答道|笑道|说|道|问|答)"
    r"\s*[:：]\s*“\s*$"
)


METADATA_SPACING_RE = re.compile(
    r"(?:内容标签|搜索关键字|标签|TAG)\s*[:：]",
    re.IGNORECASE,
)


SUSPICIOUS_AD_PATTERNS = (
    (
        "website or URL",
        re.compile(
            r"(?:https?://|www\.)[^\s<>{}]+"
            r"|(?<![@\w])(?:[A-Za-z0-9-]+\.)+"
            r"(?:com|cn|net|org|cc|me|io)(?!\w)",
            re.IGNORECASE,
        ),
    ),
    (
        "social account promotion",
        re.compile(
            r"(?:关注|搜索|添加|扫码|扫描).{0,12}"
            r"(?:公众号|微信|微博|抖音|快手)"
            r"|(?:微信公众号|微信号|微博号)\s*[:：]?\s*[A-Za-z0-9_-]+"
        ),
    ),
    (
        "reader group promotion",
        re.compile(
            r"(?:QQ|微信)?(?:读者)?群(?:号)?\s*[:：]?\s*\d{5,}"
            r"|(?:加群|进群|入群).{0,16}(?:QQ|微信|群)"
        ),
    ),
    (
        "site or download promotion",
        re.compile(
            r"(?:最新网址|最新地址|备用网址|首发网站|首发站|手机用户请访问"
            r"|更多精彩.{0,12}(?:访问|下载)|下载.{0,8}(?:APP|客户端)"
            r"|本书由.{0,30}(?:整理|校对|制作))",
            re.IGNORECASE,
        ),
    ),
)


KNOWN_MOJIBAKE = ("锟斤拷", "鏂囧", "鈥", "�")


BODY_TAGS = {"p"}


XHTML_TITLE_TAGS = {"title", "h1", "h2", "h3"}


NAV_TAGS = {"a", "span"}


NCX_TAGS = {"text"}


OPF_TAGS = {"title"}


QUOTE_PAIRS = (("“", "”"), ("‘", "’"), ("「", "」"), ("『", "』"))


@dataclass(frozen=True)
class NormalizationIssue:
    kind: str
    member: str
    message: str
    excerpt: str
    count: int = 1
    recommended_action: str = "Have Codex review the source context before editing."
    requires_codex_review: bool = True
    requires_user_review: bool = False
    review_verdict: str | None = None
    review_reason: str | None = None
    review_confidence: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "member": self.member,
            "message": self.message,
            "excerpt": self.excerpt,
            "count": self.count,
            "recommended_action": self.recommended_action,
            "requires_codex_review": self.requires_codex_review,
            "requires_user_review": self.requires_user_review,
            "review_verdict": self.review_verdict,
            "review_reason": self.review_reason,
            "review_confidence": self.review_confidence,
        }


@dataclass
class NormalizationReport:
    path: Path
    applied: bool
    change_counts: Counter[str] = field(default_factory=Counter)
    member_changes: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    samples: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    issues: list[NormalizationIssue] = field(default_factory=list)
    xml_members_checked: int = 0
    codex_review_status: str = "not_requested"
    codex_review_error: str | None = None
    codex_review_decisions: list[dict[str, object]] = field(default_factory=list)
    metadata_enrichment: MetadataEnrichmentReport | None = None

    @property
    def total_changes(self) -> int:
        return sum(self.change_counts.values())

    @property
    def changed_members(self) -> int:
        return len(self.member_changes)

    def record_change(
        self,
        kind: str,
        member: str,
        count: int,
        before: str,
        after: str,
    ) -> None:
        if not count:
            return
        self.change_counts[kind] += count
        self.member_changes[member][kind] += count
        if len(self.samples[kind]) < 5:
            self.samples[kind].append(
                {
                    "member": member,
                    "before": _excerpt(before),
                    "after": _excerpt(after),
                }
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "applied": self.applied,
            "total_changes": self.total_changes,
            "changed_members": self.changed_members,
            "change_counts": dict(sorted(self.change_counts.items())),
            "member_changes": {
                member: dict(sorted(counts.items()))
                for member, counts in sorted(self.member_changes.items())
            },
            "samples": dict(sorted(self.samples.items())),
            "issues": [issue.to_dict() for issue in self.issues],
            "xml_members_checked": self.xml_members_checked,
            "metadata_enrichment": (
                self.metadata_enrichment.to_dict() if self.metadata_enrichment else None
            ),
            "codex_review": {
                "status": self.codex_review_status,
                "error": self.codex_review_error,
                "reviewed": sum(
                    not issue.requires_codex_review for issue in self.issues
                ),
                "pending": sum(issue.requires_codex_review for issue in self.issues),
                "needs_user_review": sum(
                    issue.requires_user_review for issue in self.issues
                ),
                "decisions": self.codex_review_decisions,
            },
        }

    def format_text(self, *, include_members: bool = False) -> str:
        mode = "applied" if self.applied else "check"
        lines = [
            (
                f"NORMALIZE {self.path} ({mode}): "
                f"{self.total_changes} fixes in {self.changed_members} members; "
                f"{len(self.issues)} reported issues"
            )
        ]
        for kind, count in sorted(self.change_counts.items()):
            lines.append(f"  fix {kind}: {count}")
        if self.metadata_enrichment is not None:
            lines.append(f"  {self.metadata_enrichment.format_text()}")
        if self.codex_review_status != "not_requested":
            reviewed = sum(not issue.requires_codex_review for issue in self.issues)
            pending = sum(issue.requires_codex_review for issue in self.issues)
            needs_user = sum(issue.requires_user_review for issue in self.issues)
            lines.append(
                "  codex review "
                f"{self.codex_review_status}: reviewed={reviewed}, "
                f"pending={pending}, needs_user={needs_user}"
            )
            if self.codex_review_error:
                lines.append(f"    error: {self.codex_review_error}")
        if include_members:
            for member, counts in sorted(self.member_changes.items()):
                details = ", ".join(
                    f"{kind}={count}" for kind, count in sorted(counts.items())
                )
                lines.append(f"  member {member}: {details}")
        for issue in self.issues:
            lines.append(
                f"  report {issue.kind} {issue.member}: "
                f"{issue.message} [{issue.excerpt}]"
            )
            if issue.requires_codex_review:
                lines.append("    review: Codex required")
            elif issue.requires_user_review:
                lines.append("    review: user decision required")
            elif issue.review_verdict:
                lines.append(
                    "    review: "
                    f"{issue.review_verdict} ({issue.review_confidence or 'unknown'})"
                )
                if issue.review_reason:
                    lines.append(f"    reason: {issue.review_reason}")
            if issue.recommended_action:
                lines.append(f"    action: {issue.recommended_action}")
        return "\n".join(lines)


def _excerpt(text: str, width: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= width:
        return compact
    return compact[: width - 1] + "…"


def _quote_marks(
    text: str,
    opening: str,
    closing: str,
) -> list[tuple[int, str]]:
    marks: list[tuple[int, str]] = []
    local_depth = 0
    for index, char in enumerate(text):
        if char == opening:
            marks.append((index, char))
            local_depth += 1
            continue
        if char != closing:
            continue
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        corner_bracket_emoticon = (
            closing == "」" and previous.isdigit() and following == "∠"
        )
        if corner_bracket_emoticon:
            continue
        latin_apostrophe = (
            opening == "‘"
            and local_depth == 0
            and (
                previous.isascii()
                and previous.isalpha()
                or following.isascii()
                and following.isalpha()
            )
        )
        if latin_apostrophe:
            continue
        marks.append((index, char))
        local_depth = max(0, local_depth - 1)
    return marks


def _quote_review_context(paragraphs: list[str], index: int) -> str:
    context: list[str] = []
    if index > 0:
        context.append(f"previous: {paragraphs[index - 1]}")
    context.append(f"paragraph {index + 1}: {paragraphs[index]}")
    if index + 1 < len(paragraphs):
        context.append(f"next: {paragraphs[index + 1]}")
    return _excerpt(" | ".join(context), width=420)


def _quote_mismatch_indices(
    paragraphs: list[str],
    opening: str,
    closing: str,
) -> list[tuple[int, int, int]]:
    counts: list[tuple[int, int]] = []
    paragraph_marks: list[list[tuple[int, str]]] = []
    for paragraph in paragraphs:
        marks = _quote_marks(paragraph, opening, closing)
        paragraph_marks.append(marks)
        counts.append(
            (
                sum(char == opening for _index, char in marks),
                sum(char == closing for _index, char in marks),
            )
        )

    mismatches: dict[int, tuple[int, int]] = {}
    depth = 0
    active_start: int | None = None
    for paragraph_index, (paragraph, marks) in enumerate(
        zip(paragraphs, paragraph_marks)
    ):
        if not marks:
            continue
        first_position, first_mark = marks[0]
        continuation_open = (
            depth > 0
            and first_mark == opening
            and not paragraph[:first_position].strip()
        )
        for mark_index, (_position, mark) in enumerate(marks):
            if continuation_open and mark_index == 0:
                continue
            if mark == opening:
                if depth == 0:
                    active_start = paragraph_index
                depth += 1
                continue
            if depth:
                depth -= 1
                if depth == 0:
                    active_start = None
                continue
            mismatches[paragraph_index] = counts[paragraph_index]

    if depth and active_start is not None:
        mismatches[active_start] = counts[active_start]
    return [
        (index, opening_count, closing_count)
        for index, (opening_count, closing_count) in sorted(mismatches.items())
    ]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _is_han(char: str) -> bool:
    return bool(HAN_RE.fullmatch(char))


def _is_narrow_alnum(char: str) -> bool:
    return (char.isascii() and char.isalnum()) or (
        char.isalpha() and "GREEK" in unicodedata.name(char, "")
    )


def _is_chinese_context(text: str) -> bool:
    return bool(HAN_RE.search(text))


def _surface_tags(member: str) -> set[str]:
    basename = Path(member).name
    if member.endswith(".xhtml"):
        return BODY_TAGS | XHTML_TITLE_TAGS
    if basename == "toc.ncx":
        return NCX_TAGS
    if basename == "content.opf":
        return OPF_TAGS
    return set()


def _tag_pattern(tags: Iterable[str]) -> re.Pattern[str]:
    alternatives = "|".join(sorted(map(re.escape, tags), key=len, reverse=True))
    return re.compile(
        rf"(<(?:[A-Za-z_][\w.-]*:)?(?:{alternatives})\b[^>]*>)"
        rf"(.*?)"
        rf"(</(?:[A-Za-z_][\w.-]*:)?(?:{alternatives})>)",
        re.DOTALL,
    )


def _apply_regex(
    text: str,
    pattern: re.Pattern[str],
    replacement: str | Callable[[re.Match[str]], str],
    *,
    kind: str,
    member: str,
    report: NormalizationReport,
) -> str:
    before = text
    text, count = pattern.subn(replacement, text)
    report.record_change(kind, member, count, before, text)
    return text


def _normalize_chinese_numeral_zero(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    before = text
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group(0)
        if "○" not in value or value == "○" * len(value):
            return value
        count += value.count("○")
        return value.replace("○", "〇")

    text = CHINESE_DIGIT_RUN_RE.sub(replace, text)
    report.record_change(
        "white_circle_to_ideographic_zero",
        member,
        count,
        before,
        text,
    )
    return text


def _normalize_contextual_quote_directions(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    before = text
    output: list[str] = []
    depth = 0
    changes = 0
    seen_non_whitespace = False
    for index, char in enumerate(text):
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if char == "“":
            if depth > 0 and previous in "，。！？…—":
                output.append("”")
                depth -= 1
                changes += 1
            else:
                output.append(char)
                depth += 1
            seen_non_whitespace = True
            continue
        if char == "”":
            wrong_opening_context = depth == 0 and (
                not seen_non_whitespace
                or previous in ":："
                and bool(HAN_RE.match(following))
            )
            if wrong_opening_context:
                output.append("“")
                depth += 1
                changes += 1
            else:
                output.append(char)
                depth = max(0, depth - 1)
            seen_non_whitespace = True
            continue
        output.append(char)
        if not char.isspace():
            seen_non_whitespace = True
    normalized = "".join(output)
    report.record_change(
        "contextual_quote_direction_fixed",
        member,
        changes,
        before,
        normalized,
    )
    return normalized


def _thousands_comma_positions(text: str) -> set[int]:
    positions: set[int] = set()
    for match in THOUSANDS_RE.finditer(text):
        positions.update(
            index for index in range(match.start(), match.end()) if text[index] == ","
        )
    return positions


def _chinese_punctuation_context(text: str, index: int) -> bool:
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if _is_han(previous) or _is_han(following):
        return True
    boundaries = "\n。！？!?；;：:"
    start = max((text.rfind(char, 0, index) for char in boundaries), default=-1)
    following_boundaries = [
        position for char in boundaries if (position := text.find(char, index + 1)) >= 0
    ]
    end = min(following_boundaries, default=len(text))
    clause = text[start + 1 : end]
    han_count = len(HAN_RE.findall(clause))
    if not han_count:
        return False
    latin_count = sum(char.isascii() and char.isalpha() for char in clause)
    if latin_count >= 4 and latin_count > han_count * 2:
        return False
    return True


def _normalize_ascii_pairs(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    text = _apply_regex(
        text,
        ASCII_PAREN_WITH_HAN_RE,
        r"（\g<body>）",
        kind="ascii_parentheses_to_chinese",
        member=member,
        report=report,
    )
    return _apply_regex(
        text,
        ASCII_BRACKET_WITH_HAN_RE,
        r"［\g<body>］",
        kind="ascii_brackets_to_chinese",
        member=member,
        report=report,
    )


def _normalize_ascii_runs(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    before = text
    ellipsis_count = 0

    def replace_ellipsis(match: re.Match[str]) -> str:
        nonlocal ellipsis_count
        if not _chinese_punctuation_context(text, match.start()):
            return match.group(0)
        ellipsis_count += 1
        return "……"

    text = ASCII_ELLIPSIS_RE.sub(replace_ellipsis, text)
    report.record_change(
        "ascii_ellipsis_to_chinese",
        member,
        ellipsis_count,
        before,
        text,
    )

    before = text
    dash_count = 0

    def replace_dash(match: re.Match[str]) -> str:
        nonlocal dash_count
        if not _chinese_punctuation_context(text, match.start()):
            return match.group(0)
        dash_count += 1
        return "——"

    text = ASCII_DASH_RE.sub(replace_dash, text)
    report.record_change(
        "ascii_dash_to_chinese",
        member,
        dash_count,
        before,
        text,
    )
    return text


def _han_run_before(text: str, index: int) -> int:
    count = 0
    for char in reversed(text[:index]):
        if not _is_han(char):
            break
        count += 1
    return count


def _should_convert_ascii_question(text: str, index: int) -> bool:
    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous == "?" or following == "?":
        return False
    if not _chinese_punctuation_context(text, index):
        return False
    if following and (_is_narrow_alnum(following) or following == "_"):
        return False
    if _is_han(previous) and _is_han(following):
        return _han_run_before(text, index) >= 2
    return True


def _normalize_ascii_punctuation(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    if not text:
        return text
    text = _normalize_ascii_pairs(text, member=member, report=report)
    text = _normalize_ascii_runs(text, member=member, report=report)
    thousands = _thousands_comma_positions(text)
    chinese_context = _is_chinese_context(text)
    output: list[str] = []
    comma_count = 0
    period_count = 0
    question_count = 0
    exclamation_count = 0
    colon_count = 0
    semicolon_count = 0
    for index, char in enumerate(text):
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if char == "," and index not in thousands:
            if not (_is_narrow_alnum(previous) and _is_narrow_alnum(following)) and (
                _is_han(previous)
                or _is_han(following)
                or previous in ("”", "’", "》", "）", "】", "」", "』")
                or following in ("“", "‘", "《", "【", "「", "『")
            ):
                char = "，"
                comma_count += 1
        elif char == "." and chinese_context:
            is_decimal = previous.isdigit() and following.isdigit()
            is_ascii_token = _is_narrow_alnum(previous) and _is_narrow_alnum(following)
            is_filename_extension = bool(FILENAME_EXTENSION_RE.match(text, index + 1))
            is_ellipsis = previous == "." or following == "."
            url_tail = not following and bool(
                re.search(r"(?:https?://|www\.)\S+\.$", text, re.IGNORECASE)
            )
            punctuation_context = _chinese_punctuation_context(text, index)
            if (
                not (
                    is_decimal
                    or is_ascii_token
                    or is_filename_extension
                    or is_ellipsis
                    or url_tail
                )
                and punctuation_context
                and (
                    _is_han(previous)
                    or _is_han(following)
                    or previous in "”’）》】」』"
                    or not following
                )
            ):
                char = "。"
                period_count += 1
        elif char == "?" and _should_convert_ascii_question(text, index):
            char = "？"
            question_count += 1
        elif char == "!" and _chinese_punctuation_context(text, index):
            char = "！"
            exclamation_count += 1
        elif char == ":" and _chinese_punctuation_context(text, index):
            is_time = previous.isdigit() and following.isdigit()
            is_ascii_token = (previous.isascii() and previous.isalnum()) and (
                following.isascii() and following.isalnum()
            )
            is_emoticon = following.isdigit() and previous in "(_"
            is_path_or_url = following in "/\\" or text[
                max(0, index - 6) : index
            ].endswith(("http", "https"))
            if not (is_time or is_ascii_token or is_emoticon or is_path_or_url):
                char = "："
                colon_count += 1
        elif char == ";" and _chinese_punctuation_context(text, index):
            is_broken_entity = text[max(0, index - 4) : index].lower() == "amp"
            if not is_broken_entity:
                char = "；"
                semicolon_count += 1
        output.append(char)
    normalized = "".join(output)
    report.record_change(
        "ascii_comma_to_chinese",
        member,
        comma_count,
        text,
        normalized,
    )
    report.record_change(
        "ascii_period_to_chinese",
        member,
        period_count,
        text,
        normalized,
    )
    report.record_change(
        "ascii_question_to_chinese",
        member,
        question_count,
        text,
        normalized,
    )
    report.record_change(
        "ascii_exclamation_to_chinese",
        member,
        exclamation_count,
        text,
        normalized,
    )
    report.record_change(
        "ascii_colon_to_chinese",
        member,
        colon_count,
        text,
        normalized,
    )
    report.record_change(
        "ascii_semicolon_to_chinese",
        member,
        semicolon_count,
        text,
        normalized,
    )
    return normalized


def _arabic_to_chinese_numeral(value: int) -> str:
    if value < 0 or value > 9999:
        return str(value)
    if value == 0:
        return "零"

    digits = "零一二三四五六七八九"
    output: list[str] = []
    zero_pending = False
    remaining = value
    for place, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        digit, remaining = divmod(remaining, place)
        if digit:
            if zero_pending:
                output.append("零")
            output.extend((digits[digit], unit))
            zero_pending = False
        elif output and remaining:
            zero_pending = True
    normalized = "".join(output)
    return normalized[1:] if normalized.startswith("一十") else normalized


def _normalize_fanwai_title(
    text: str,
    *,
    force: bool,
    member: str,
    report: NormalizationReport,
) -> str:
    prefix_pattern = CHAPTER_NUMBER_PREFIX_RE if force else FANWAI_CHAPTER_NUMBER_RE
    without_number, removed = prefix_pattern.subn("", text, count=1)
    report.record_change(
        "fanwai_chapter_number_removed",
        member,
        removed,
        text,
        without_number,
    )
    text = without_number

    text = _apply_regex(
        text,
        FANWAI_COLON_SEPARATOR_RE,
        "·",
        kind="fanwai_colon_to_middle_dot",
        member=member,
        report=report,
    )

    text = _apply_regex(
        text,
        MIDDLE_AUTUMN_FANWAI_INTERNAL_SEPARATOR_RE,
        r"\g<prefix>",
        kind="fanwai_middle_autumn_internal_separator_removed",
        member=member,
        report=report,
    )

    without_trailing_separator, trailing_separator_removed = (
        FANWAI_TRAILING_SEPARATOR_RE.subn("", text, count=1)
    )
    report.record_change(
        "fanwai_trailing_separator_removed",
        member,
        trailing_separator_removed,
        text,
        without_trailing_separator,
    )
    text = without_trailing_separator

    text = _apply_regex(
        text,
        FANWAI_MISSING_YEAR_RE,
        r"\g<year> 年",
        kind="fanwai_middle_autumn_year_added",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        MIDDLE_AUTUMN_FANWAI_TITLE_SEPARATOR_RE,
        r"\g<prefix>·",
        kind="fanwai_middle_autumn_title_separator_added",
        member=member,
        report=report,
    )

    match = FANWAI_NUMBERED_TITLE_RE.fullmatch(text)
    if match is not None:
        if text == "番外六一快乐" or match.group("title") in {"（完）", "(完)"}:
            title = match.group("title") or ""
            separator = match.group("separator")
        else:
            title = match.group("title") or ""
            separator = "·" if title else ""

        number = match.group("number")
        report.record_change(
            "fanwai_number_spacing_removed",
            member,
            int(bool(match.group("number_spacing"))),
            text,
            text.replace(
                f"{match.group('prefix')}{match.group('number_spacing')}{number}",
                f"{match.group('prefix')}{number}",
                1,
            ),
        )
        normalized_number = (
            _arabic_to_chinese_numeral(int(number))
            if number.isascii() and number.isdigit()
            else number
        )
        report.record_change(
            "fanwai_number_to_chinese",
            member,
            int(normalized_number != number),
            number,
            normalized_number,
        )
        normalized = f"{match.group('prefix')}{normalized_number}{separator}{title}"
        report.record_change(
            "fanwai_title_separator_normalized",
            member,
            int(separator != match.group("separator")),
            text,
            normalized,
        )
        text = normalized

    if force:
        prefix = GROUPED_FANWAI_PREFIX_RE.match(text)
        without_prefix = text[prefix.end() :] if prefix is not None else text
        if without_prefix:
            report.record_change(
                "fanwai_group_prefix_removed",
                member,
                int(without_prefix != text),
                text,
                without_prefix,
            )
            text = without_prefix
    return text


def _normalize_nav_epub_namespace_prefix(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    match = EPUB_OPS_NAMESPACE_DECL_RE.search(text)
    if match is None or match.group("prefix") == "epub":
        return text

    prefix = match.group("prefix")
    normalized = EPUB_OPS_NAMESPACE_DECL_RE.sub(
        'xmlns:epub="http://www.idpf.org/2007/ops"',
        text,
        count=1,
    )
    qualified_name_re = re.compile(
        rf"(?P<boundary>[<\s/]){re.escape(prefix)}:(?=[A-Za-z_])"
    )
    normalized, qualified_names = qualified_name_re.subn(
        r"\g<boundary>epub:",
        normalized,
    )
    report.record_change(
        "nav_epub_namespace_prefix_normalized",
        member,
        1 + qualified_names,
        text,
        normalized,
    )
    return normalized


def _normalize_title_periods(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    output: list[str] = []
    count = 0
    for index, char in enumerate(text):
        if char != ".":
            output.append(char)
            continue
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        preserve = (
            (previous.isdigit() and following.isdigit())
            or previous == "."
            or following == "."
            or bool(FILENAME_EXTENSION_RE.match(text, index + 1))
        )
        if preserve:
            output.append(char)
        else:
            output.append("·")
            count += 1
    normalized = "".join(output)
    report.record_change(
        "chapter_title_period_to_middle_dot",
        member,
        count,
        text,
        normalized,
    )
    return normalized


def _normalize_decorative_end_marker(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    match = DECORATIVE_END_MARKER_RE.fullmatch(text)
    if not match:
        return text
    label = match.group("label").strip()
    term = next((value for value in ENDING_TERMS if label.endswith(value)), None)
    if not term:
        return text
    prefix = label[: -len(term)].rstrip("· ")
    if prefix:
        prefix = re.sub(
            r"^((?:第?[零〇一二三四五六七八九十百千万两]+卷|"
            r"卷[零〇一二三四五六七八九十百千万两]+))(?=[^·：:\s])",
            r"\1·",
            prefix,
        )
        separator = "" if prefix.endswith(("·", "：", ":", "？", "！")) else "·"
        label = f"{prefix}{separator}{term}"
    else:
        label = term
    normalized = f"——{label}——"
    report.record_change(
        "decorative_end_marker_normalized",
        member,
        int(normalized != text),
        text,
        normalized,
    )
    return normalized


def _normalize_code_commas(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    thousands = _thousands_comma_positions(text)
    output: list[str] = []
    index = 0
    count = 0
    while index < len(text):
        char = text[index]
        if (
            char == ","
            and index not in thousands
            and index > 0
            and _is_narrow_alnum(text[index - 1])
        ):
            next_index = index + 1
            while next_index < len(text) and text[next_index] in " \t":
                next_index += 1
            if next_index < len(text) and _is_narrow_alnum(text[next_index]):
                output.append(", ")
                if text[index + 1 : next_index] != " ":
                    count += 1
                index = next_index
                continue
        output.append(char)
        index += 1
    normalized = "".join(output)
    report.record_change(
        "english_or_code_comma_spacing",
        member,
        count,
        text,
        normalized,
    )
    return normalized


def _mixed_boundary_positions(text: str, preserve_ordinals: bool) -> set[int]:
    excluded: set[int] = set()
    if preserve_ordinals:
        for match in ORDINAL_RE.finditer(text):
            excluded.update(range(match.start(), match.end() - 1))
    return {
        index
        for index, (left, right) in enumerate(zip(text, text[1:]))
        if index not in excluded
        and (
            (_is_han(left) and _is_narrow_alnum(right))
            or (_is_narrow_alnum(left) and _is_han(right))
        )
    }


def _insert_mixed_width_spaces(
    text: str,
    *,
    preserve_ordinals: bool,
    member: str,
    report: NormalizationReport,
) -> str:
    positions = _mixed_boundary_positions(text, preserve_ordinals)
    if not positions:
        return text
    output: list[str] = []
    for index, char in enumerate(text):
        output.append(char)
        if index in positions:
            output.append(" ")
    normalized = "".join(output)
    report.record_change(
        "mixed_width_space_inserted",
        member,
        len(positions),
        text,
        normalized,
    )
    return normalized


def _normalize_known_split_han_words(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    before = text
    changes = 0
    for word in KNOWN_SPLIT_HAN_WORDS:
        pattern = re.compile(
            re.escape(word[0]) + r"[\t \u00a0\u3000]+" + re.escape(word[1])
        )
        text, count = pattern.subn(word, text)
        changes += count
    for left, right in KNOWN_SPLIT_HAN_FRAGMENTS:
        pattern = re.compile(re.escape(left) + r"[\t \u00a0\u3000]+" + re.escape(right))
        text, count = pattern.subn(left + right, text)
        changes += count
    report.record_change(
        "known_split_han_word_joined",
        member,
        changes,
        before,
        text,
    )
    return text


def _normalize_han_internal_spacing(
    text: str,
    *,
    preserve_ordinals: bool,
    remove_han_spaces: bool,
    member: str,
    report: NormalizationReport,
) -> str:
    known_split = any(
        re.search(
            re.escape(word[0]) + r"[\t \u00a0\u3000]+" + re.escape(word[1]),
            text,
        )
        for word in KNOWN_SPLIT_HAN_WORDS
    ) or any(
        re.search(
            re.escape(left) + r"[\t \u00a0\u3000]+" + re.escape(right),
            text,
        )
        for left, right in KNOWN_SPLIT_HAN_FRAGMENTS
    )
    if preserve_ordinals:
        ordinal_separator_re = re.compile(
            rf"({ORDINAL_RE.pattern})([\t \u00a0\u3000]*)(?=[^\s，。！？；：、·])"
        )
        separator_changes = 0
        before = text

        def normalize_separator(match: re.Match[str]) -> str:
            nonlocal separator_changes
            normalized = match.group(1) + " "
            if match.group(0) != normalized:
                separator_changes += 1
            return normalized

        text = ordinal_separator_re.sub(normalize_separator, text)
        report.record_change(
            "chapter_title_separator_normalized",
            member,
            separator_changes,
            before,
            text,
        )
        protected_starts = {
            match.end()
            for match in ORDINAL_RE.finditer(text)
            if match.end() < len(text) and text[match.end()] in "\t \u00a0\u3000"
        }
    else:
        protected_starts = set()

    if (
        not remove_han_spaces
        or ORDINAL_RE.match(text.strip())
        or METADATA_SPACING_RE.search(text)
        or re.search(r"(?:搜索栏|搜索词|关键字)", text)
        or text.strip().startswith(("——摘自", "—摘自", "摘自"))
        or (
            len(text.strip()) < 30
            and not re.search(r"[，。！？；：]", text)
            and not known_split
        )
    ):
        return text
    text = _normalize_known_split_han_words(
        text,
        member=member,
        report=report,
    )
    matches = list(HAN_INTERNAL_SPACE_RE.finditer(text))
    if len(matches) != 1:
        return text

    removed_count = 0
    before = text

    def remove_space(match: re.Match[str]) -> str:
        nonlocal removed_count
        if match.start() in protected_starts:
            return " "
        removed_count += 1
        return ""

    text = HAN_INTERNAL_SPACE_RE.sub(remove_space, text)
    report.record_change(
        "space_between_han_removed",
        member,
        removed_count,
        before,
        text,
    )
    return text


def _normalize_han_internal_backslash(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    matches = list(HAN_INTERNAL_BACKSLASH_RE.finditer(text))
    if len(matches) != 1:
        return text
    return _apply_regex(
        text,
        HAN_INTERNAL_BACKSLASH_RE,
        "",
        kind="backslash_between_han_removed",
        member=member,
        report=report,
    )


def _normalize_plain_text(
    text: str,
    *,
    preserve_ordinals: bool,
    remove_han_spaces: bool,
    title_punctuation: bool,
    end_marker: bool,
    force_fanwai_title: bool,
    member: str,
    report: NormalizationReport,
) -> str:
    text = _normalize_chinese_numeral_zero(
        text,
        member=member,
        report=report,
    )
    if title_punctuation:
        text = _normalize_fanwai_title(
            text,
            force=force_fanwai_title,
            member=member,
            report=report,
        )
        text = _normalize_title_periods(
            text,
            member=member,
            report=report,
        )
    if end_marker:
        text = _normalize_decorative_end_marker(
            text,
            member=member,
            report=report,
        )
    # Punctuation is the first general prose-normalization stage. Quote
    # direction and spacing rules intentionally consume its normalized output.
    text = _normalize_ascii_punctuation(text, member=member, report=report)
    text = _apply_regex(
        text,
        REPEATED_CHINESE_COMMA_RE,
        "，",
        kind="repeated_chinese_comma_collapsed",
        member=member,
        report=report,
    )
    for pattern, replacement in (
        (re.compile(r"，。"), "。"),
        (re.compile(r"。！"), "！"),
        (re.compile(r"。？"), "？"),
    ):
        text = _apply_regex(
            text,
            pattern,
            replacement,
            kind="broken_terminal_punctuation_fixed",
            member=member,
            report=report,
        )
    text = _apply_regex(
        text,
        STRAIGHT_DOUBLE_QUOTE_RE,
        r"“\1”",
        kind="straight_double_quotes_curled",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        STRAIGHT_SINGLE_QUOTE_RE,
        r"‘\1’",
        kind="straight_single_quotes_curled",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        REVERSED_SINGLE_QUOTE_PAIR_RE,
        r"‘\g<body>’",
        kind="single_quote_direction_fixed",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        DIALOGUE_ATTRIBUTION_RE,
        r"\g<close_punct>”\g<attribution>\g<resume_punct>“",
        kind="dialogue_quote_direction_fixed",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        DUPLICATE_OPEN_AFTER_COLON_RE,
        "“",
        kind="duplicate_quote_removed",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        DUPLICATE_OPEN_AT_START_RE,
        "“",
        kind="duplicate_quote_removed",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        DIALOGUE_BOUNDARY_DOUBLE_RE,
        "”“",
        kind="quote_boundary_direction_fixed",
        member=member,
        report=report,
    )
    text = _normalize_contextual_quote_directions(
        text,
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        DUPLICATE_CLOSE_AT_END_RE,
        "”",
        kind="duplicate_quote_removed",
        member=member,
        report=report,
    )
    text = _apply_regex(
        text,
        RESOURCE_GROUP_AD_RE,
        "",
        kind="resource_group_ad_removed",
        member=member,
        report=report,
    )
    text = _normalize_han_internal_backslash(
        text,
        member=member,
        report=report,
    )
    text = _normalize_han_internal_spacing(
        text,
        preserve_ordinals=preserve_ordinals,
        remove_han_spaces=remove_han_spaces,
        member=member,
        report=report,
    )
    spacing_patterns = [
        SPACE_BEFORE_CJK_PUNCT_RE,
        SPACE_BEFORE_CLOSING_QUOTE_RE,
        SPACE_AFTER_CJK_PUNCT_RE,
        SPACE_AFTER_CJK_OPEN_RE,
        CJK_SPACE_BEFORE_OPEN_RE,
    ]
    if preserve_ordinals:
        spacing_patterns.remove(CJK_SPACE_BEFORE_OPEN_RE)
    for pattern in spacing_patterns:
        text = _apply_regex(
            text,
            pattern,
            "",
            kind="cjk_punctuation_space_removed",
            member=member,
            report=report,
        )
    text = _normalize_code_commas(text, member=member, report=report)
    return _insert_mixed_width_spaces(
        text,
        preserve_ordinals=preserve_ordinals,
        member=member,
        report=report,
    )


def _normalize_fragment(
    fragment: str,
    *,
    preserve_ordinals: bool,
    remove_han_spaces: bool,
    title_punctuation: bool,
    end_marker: bool,
    force_fanwai_title: bool,
    member: str,
    report: NormalizationReport,
) -> str:
    tokens = TOKEN_RE.split(fragment)
    for index, token in enumerate(tokens):
        if not token or token.startswith("<") or token.startswith("&"):
            continue
        tokens[index] = _normalize_plain_text(
            token,
            preserve_ordinals=preserve_ordinals,
            remove_han_spaces=remove_han_spaces,
            title_punctuation=title_punctuation,
            end_marker=end_marker,
            force_fanwai_title=force_fanwai_title,
            member=member,
            report=report,
        )

    records: list[tuple[int, int, int, str] | None] = []
    for token_index, token in enumerate(tokens):
        if not token:
            continue
        if token.startswith("<"):
            tag = token[1:].lstrip("/").split(None, 1)[0].rstrip(">/").lower()
            if tag in {"br", "img", "hr"} and not token.startswith("</"):
                records.append(None)
            continue
        if token.startswith("&"):
            decoded = html.unescape(token)
            if len(decoded) == 1:
                records.append((token_index, 0, len(token), decoded))
            else:
                records.append(None)
            continue
        records.extend(
            (token_index, offset, offset + 1, char) for offset, char in enumerate(token)
        )
    visible = "".join(record[3] if record else "\n" for record in records)
    positions = _mixed_boundary_positions(visible, preserve_ordinals)
    insertions: dict[int, set[int]] = defaultdict(set)
    for position in positions:
        left = records[position]
        right = records[position + 1]
        if left is None or right is None or left[0] == right[0]:
            continue
        insertions[left[0]].add(left[2])
    cross_tag_count = sum(len(offsets) for offsets in insertions.values())
    if cross_tag_count:
        before = "".join(tokens)
        for token_index, offsets in insertions.items():
            token = tokens[token_index]
            for offset in sorted(offsets, reverse=True):
                token = token[:offset] + " " + token[offset:]
            tokens[token_index] = token
        report.record_change(
            "mixed_width_space_inserted",
            member,
            cross_tag_count,
            before,
            "".join(tokens),
        )
    return "".join(tokens)


def _visible_fragment_text(fragment: str) -> str:
    try:
        root = ET.fromstring(f"<root>{fragment}</root>")
    except ET.ParseError:
        return re.sub(r"<[^>]+>", "", fragment)
    return "".join(root.itertext())


def _author_note_heading_offset(fragment: str) -> int | None:
    """Find an author-note heading in a plain-text part of an XHTML fragment."""
    cursor = 0
    for token in TOKEN_RE.finditer(fragment):
        plain = fragment[cursor : token.start()]
        match = AUTHOR_NOTE_HEADING_RE.search(plain)
        if match is not None:
            return cursor + match.start()
        cursor = token.end()
    match = AUTHOR_NOTE_HEADING_RE.search(fragment[cursor:])
    if match is None:
        return None
    return cursor + match.start()


def _fragment_is_well_formed(fragment: str) -> bool:
    prefixes = {
        prefix
        for prefix in re.findall(r"</?([A-Za-z_][\w.-]*):", fragment)
        if prefix != "xml"
    }
    declarations = "".join(
        f' xmlns:{prefix}="urn:epub-normalizer:{prefix}"' for prefix in sorted(prefixes)
    )
    try:
        ET.fromstring(f"<root{declarations}>{fragment}</root>")
    except ET.ParseError:
        return False
    return True


def _center_paragraph_opening(opening: str) -> str:
    style_match = re.search(r"""(\sstyle\s*=\s*)(["'])(.*?)\2""", opening)
    additions = []
    existing = style_match.group(3) if style_match else ""
    if "text-align" not in existing:
        additions.append("text-align: center")
    if "text-indent" not in existing:
        additions.append("text-indent: 0")
    if not additions:
        return opening
    if style_match:
        separator = "" if not existing or existing.rstrip().endswith(";") else ";"
        style = existing + separator + " " + "; ".join(additions) + ";"
        return opening[: style_match.start(3)] + style + opening[style_match.end(3) :]
    style = "; ".join(additions) + ";"
    return opening[:-1] + f' style="{style}">'


def _normalize_structural_paragraphs(
    text: str,
    *,
    member: str,
    report: NormalizationReport,
) -> str:
    while True:
        paragraphs = list(PARAGRAPH_RE.finditer(text))
        replacement: tuple[int, int, str, str, str] | None = None
        for paragraph in paragraphs:
            body = paragraph.group("body")
            marker_offset = _author_note_heading_offset(body)
            if marker_offset is None:
                continue
            before_body = body[:marker_offset].rstrip()
            author_note_body = body[marker_offset:].lstrip()
            if not _visible_fragment_text(before_body).strip():
                continue
            if not (
                _fragment_is_well_formed(before_body)
                and _fragment_is_well_formed(author_note_body)
            ):
                continue
            before = text[paragraph.start() : paragraph.end()]
            after = (
                paragraph.group("open")
                + before_body
                + paragraph.group("close")
                + "\n"
                + paragraph.group("open")
                + author_note_body
                + paragraph.group("close")
            )
            replacement = (
                paragraph.start(),
                paragraph.end(),
                after,
                "author_note_paragraph_break_inserted",
                before,
            )
            break
        if replacement is not None:
            start, end, after, kind, before = replacement
            report.record_change(kind, member, 1, before, after)
            text = text[:start] + after + text[end:]
            continue
        for first, second in zip(paragraphs, paragraphs[1:]):
            if text[first.end() : second.start()].strip():
                continue
            first_text = _visible_fragment_text(first.group("body")).strip()
            second_text = _visible_fragment_text(second.group("body")).strip()
            if second_text in {"”", "’"}:
                before = text[first.start() : second.end()]
                after = (
                    first.group("open")
                    + first.group("body").rstrip()
                    + second_text
                    + first.group("close")
                )
                replacement = (
                    first.start(),
                    second.end(),
                    after,
                    "isolated_closing_quote_merged",
                    before,
                )
                break
            if (
                first_text.endswith(("，", ","))
                and second_text
                and AUTHOR_NOTE_PARAGRAPH_RE.match(second_text) is None
                and ZH_TRANSLATION_CLASS_RE.search(first.group("open")) is None
                and ZH_TRANSLATION_CLASS_RE.search(second.group("open")) is None
            ):
                before = text[first.start() : second.end()]
                after = (
                    first.group("open")
                    + first.group("body").rstrip()
                    + second.group("body").lstrip()
                    + first.group("close")
                )
                replacement = (
                    first.start(),
                    second.end(),
                    after,
                    "comma_paragraph_break_merged",
                    before,
                )
                break
        if replacement is None:
            return text
        start, end, after, kind, before = replacement
        report.record_change(kind, member, 1, before, after)
        text = text[:start] + after + text[end:]


def _grouped_fanwai_title_labels(members: dict[str, bytes]) -> set[str]:
    labels: set[str] = set()

    def record_label(value: str) -> None:
        match = CHAPTER_NUMBER_PREFIX_RE.match(value)
        if match is not None:
            if value[match.end() :].strip():
                labels.add(value)
        elif value:
            labels.add(value)

    for member, data in members.items():
        basename = Path(member).name
        if basename == "nav.xhtml":
            root = ET.fromstring(data)
            for item in root.iter():
                if _local_name(item.tag) != "li":
                    continue
                direct_labels = [
                    child
                    for child in list(item)
                    if _local_name(child.tag) in {"a", "span"}
                ]
                if not direct_labels:
                    continue
                parent_label = "".join(direct_labels[0].itertext()).strip()
                if parent_label != "番外":
                    continue
                for descendant in item.iter():
                    if (
                        descendant is direct_labels[0]
                        or _local_name(descendant.tag) != "a"
                    ):
                        continue
                    record_label("".join(descendant.itertext()).strip())
        elif basename == "toc.ncx":
            root = ET.fromstring(data)
            for point in root.iter():
                if _local_name(point.tag) != "navPoint":
                    continue
                direct_label = next(
                    (
                        child
                        for child in list(point)
                        if _local_name(child.tag) == "navLabel"
                    ),
                    None,
                )
                if direct_label is None:
                    continue
                parent_label = "".join(direct_label.itertext()).strip()
                if parent_label != "番外":
                    continue
                for descendant in point.iter():
                    if descendant is point or _local_name(descendant.tag) != "navPoint":
                        continue
                    child_label = next(
                        (
                            child
                            for child in list(descendant)
                            if _local_name(child.tag) == "navLabel"
                        ),
                        None,
                    )
                    if child_label is not None:
                        record_label("".join(child_label.itertext()).strip())
    return labels


def normalize_member(
    member: str,
    data: bytes,
    report: NormalizationReport,
    *,
    grouped_fanwai_titles: set[str],
) -> bytes:
    tags = _surface_tags(member)
    if not tags:
        return data
    text = data.decode("utf-8")
    if Path(member).name == "nav.xhtml":
        text = _normalize_nav_epub_namespace_prefix(
            text,
            member=member,
            report=report,
        )
    if member.endswith(".xhtml"):
        text = _normalize_structural_paragraphs(
            text,
            member=member,
            report=report,
        )

    def replace(match: re.Match[str]) -> str:
        opening, fragment, closing = match.groups()
        opening_tag = re.match(
            r"<(?:[A-Za-z_][\w.-]*:)?([A-Za-z0-9]+)",
            opening,
        )
        tag = opening_tag.group(1).lower() if opening_tag else ""
        preserve_ordinals = tag in XHTML_TITLE_TAGS | NAV_TAGS | NCX_TAGS | OPF_TAGS
        visible = _visible_fragment_text(fragment).strip()
        force_fanwai_title = visible in grouped_fanwai_titles
        title_punctuation = bool(
            preserve_ordinals
            and (
                force_fanwai_title
                or "番外" in visible
                or CHAPTER_LIKE_TITLE_RE.match(visible)
            )
        )
        marker_match = (
            DECORATIVE_END_MARKER_RE.fullmatch(visible) if tag == "p" else None
        )
        end_marker = bool(
            marker_match
            and any(
                marker_match.group("label").strip().endswith(term)
                for term in ENDING_TERMS
            )
        )
        if end_marker:
            centered_opening = _center_paragraph_opening(opening)
            report.record_change(
                "decorative_end_marker_centered",
                member,
                int(centered_opening != opening),
                opening,
                centered_opening,
            )
            opening = centered_opening
        remove_han_spaces = tag == "p" and Path(member).name != "intro.xhtml"
        return (
            opening
            + _normalize_fragment(
                fragment,
                preserve_ordinals=preserve_ordinals,
                remove_han_spaces=remove_han_spaces,
                title_punctuation=title_punctuation,
                end_marker=end_marker,
                force_fanwai_title=force_fanwai_title,
                member=member,
                report=report,
            )
            + closing
        )

    if Path(member).name == "nav.xhtml":
        pattern = _tag_pattern(NAV_TAGS | XHTML_TITLE_TAGS)
    else:
        pattern = _tag_pattern(tags)
    normalized = pattern.sub(replace, text)
    ET.fromstring(normalized.encode("utf-8"))
    report.xml_members_checked += 1
    return normalized.encode("utf-8")


def _is_emoji_codepoint(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or unicodedata.category(char) == "So"
    )


def _is_intentional_emoticon_character(text: str, index: int) -> bool:
    char = text[index]
    codepoint = ord(char)
    if char in {"\u200c", "\u200d"}:
        neighbors = text[max(0, index - 2) : min(len(text), index + 3)]
        return any(_is_emoji_codepoint(value) for value in neighbors)
    if not 0x3100 <= codepoint <= 0x312F:
        return False
    start = max(0, index - 16)
    end = min(len(text), index + 17)
    window = text[start:end]
    local_index = index - start
    left_paren = max(
        window.rfind("(", 0, local_index + 1),
        window.rfind("（", 0, local_index + 1),
    )
    right_candidates = [
        position
        for position in (
            window.find(")", local_index),
            window.find("）", local_index),
        )
        if position >= 0
    ]
    if left_paren >= 0 and right_candidates:
        segment = window[left_paren : min(right_candidates) + 1]
        if re.search(r"[_^oOTＴ﹏~～￣▽ω/\\]", segment) or segment.count(char) >= 2:
            return True
    for token_match in re.finditer(r"[^\s，。！？；：]+", window):
        if not token_match.start() <= local_index < token_match.end():
            continue
        token = token_match.group(0)
        return token.count(char) >= 2 and bool(re.search(r"[_~～/\\]", token))
    return False


def _bad_character_reason(char: str) -> str | None:
    codepoint = ord(char)
    category = unicodedata.category(char)
    if char in "\n\r\t":
        return None
    if char == "\ufffd":
        return "Unicode replacement character"
    if 0xE000 <= codepoint <= 0xF8FF or category == "Co":
        return "private-use character"
    if 0x3100 <= codepoint <= 0x312F:
        return "Bopomofo character"
    if char in {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}:
        return "zero-width or byte-order character"
    if category == "Cc":
        return "control character"
    if category == "Cn":
        return "unassigned Unicode character"
    return None


def _scan_member_issues(
    member: str,
    data: bytes,
    report: NormalizationReport,
) -> None:
    if not member.endswith((".xhtml", ".ncx", ".opf")):
        return
    root = ET.fromstring(data)
    report.xml_members_checked += 1
    # Intro pages contain metadata, tag lists, source links, and intentionally
    # loose formatting rather than continuous prose. Deterministic cleanup has
    # already run, but prose-quality review findings from intro.xhtml are noise.
    if Path(member).name == "intro.xhtml":
        return
    paragraphs: list[str] = []
    author_note_flags: list[bool] = []
    review_paragraphs: list[str] = []
    if member.endswith(".xhtml"):
        paragraph_elements = [
            element for element in root.iter() if _local_name(element.tag) == "p"
        ]
        paragraphs = ["".join(element.itertext()) for element in paragraph_elements]
        in_author_note = False
        trailing_start = int(len(paragraphs) * 0.7)
        for index, paragraph in enumerate(paragraphs):
            if index >= trailing_start and (
                AUTHOR_NOTE_MARKER_RE.search(paragraph)
                or AUTHOR_NOTE_SIGNAL_RE.search(paragraph)
            ):
                in_author_note = True
            author_note_flags.append(in_author_note)
        review_paragraphs = [
            paragraph
            for paragraph, is_author_note in zip(
                paragraphs,
                author_note_flags,
                strict=True,
            )
            if not is_author_note
        ]
        heading_texts = [
            "".join(element.itertext())
            for element in root.iter()
            if _local_name(element.tag) in XHTML_TITLE_TAGS
        ]
        visible = "\n".join(heading_texts + review_paragraphs)
    else:
        visible = "\n".join(text for text in root.itertext() if text and text.strip())
    bad: dict[tuple[str, str], tuple[int, str]] = {}
    for index, char in enumerate(visible):
        reason = _bad_character_reason(char)
        if not reason or _is_intentional_emoticon_character(visible, index):
            continue
        key = (char, reason)
        count, sample = bad.get(key, (0, ""))
        start = max(0, index - 35)
        end = min(len(visible), index + 36)
        bad[key] = (count + 1, visible[start:end])
    for marker in KNOWN_MOJIBAKE:
        count = visible.count(marker)
        if count and not any(key[0] == marker for key in bad):
            bad[(marker, "known mojibake marker")] = (count, marker)
    for (char, reason), (count, sample) in sorted(bad.items()):
        code = " ".join(f"U+{ord(value):04X}" for value in char)
        report.issues.append(
            NormalizationIssue(
                kind="suspicious_character",
                member=member,
                message=f"{reason}: {code}; occurrences={count}",
                excerpt=_excerpt(sample),
                count=count,
                recommended_action=(
                    "Use Browser Act to search the source context and confirm "
                    "the intended character before editing."
                ),
            )
        )

    if member.endswith(".xhtml"):
        prose = "\n".join(review_paragraphs)
        structural_markers = [
            paragraph.strip()
            for paragraph in paragraphs
            if STRUCTURAL_MARKER_RE.fullmatch(paragraph)
        ]
        if structural_markers:
            report.issues.append(
                NormalizationIssue(
                    kind="possible_structural_marker",
                    member=member,
                    message=(
                        "standalone volume, prologue, epilogue, or extras "
                        f"markers={len(structural_markers)}"
                    ),
                    excerpt=_excerpt(structural_markers[0]),
                    count=len(structural_markers),
                    recommended_action=(
                        "Let Codex decide whether this belongs in TOC hierarchy "
                        "and whether the body marker should be removed."
                    ),
                )
            )
        orphan_attribution_quotes = [
            paragraph
            for paragraph in review_paragraphs
            if ORPHAN_ATTRIBUTION_OPEN_RE.search(paragraph)
        ]
        if orphan_attribution_quotes:
            report.issues.append(
                NormalizationIssue(
                    kind="orphan_dialogue_quote_residue",
                    member=member,
                    message=(
                        "dialogue attribution ends with a new opening quote "
                        f"but no following text; matches={len(orphan_attribution_quotes)}"
                    ),
                    excerpt=_excerpt(orphan_attribution_quotes[0], width=240),
                    count=len(orphan_attribution_quotes),
                    recommended_action=(
                        "Let Codex inspect the next paragraph and remove only a "
                        "confirmed leftover attribution colon and opening quote."
                    ),
                )
            )
        for label, pattern in SUSPICIOUS_AD_PATTERNS:
            findings = [
                (paragraph, match)
                for paragraph, is_author_note in zip(
                    paragraphs,
                    author_note_flags,
                )
                if not is_author_note
                for match in pattern.finditer(paragraph)
            ]
            if not findings:
                continue
            first_paragraph, first = findings[0]
            start = max(0, first.start() - 45)
            end = min(len(first_paragraph), first.end() + 45)
            report.issues.append(
                NormalizationIssue(
                    kind="suspicious_ad",
                    member=member,
                    message=f"{label}; matches={len(findings)}",
                    excerpt=_excerpt(first_paragraph[start:end]),
                    count=len(findings),
                    recommended_action=(
                        "Review in narrative context; do not delete "
                        "automatically. Let Codex judge whether it is an ad."
                    ),
                )
            )
        duplicate_quotes = list(AMBIGUOUS_DUPLICATE_QUOTE_RE.finditer(prose))
        if duplicate_quotes:
            first = duplicate_quotes[0]
            start = max(0, first.start() - 45)
            end = min(len(prose), first.end() + 45)
            report.issues.append(
                NormalizationIssue(
                    kind="ambiguous_duplicate_quote",
                    member=member,
                    message=f"unresolved adjacent quote runs={len(duplicate_quotes)}",
                    excerpt=_excerpt(prose[start:end]),
                    count=len(duplicate_quotes),
                    recommended_action=(
                        "Review the source context before changing quote direction "
                        "or removing a quote."
                    ),
                )
            )
        ambiguous_backslashes = list(HAN_INTERNAL_BACKSLASH_RE.finditer(prose))
        if ambiguous_backslashes:
            first = ambiguous_backslashes[0]
            start = max(0, first.start() - 45)
            end = min(len(prose), first.end() + 45)
            report.issues.append(
                NormalizationIssue(
                    kind="ambiguous_han_backslash",
                    member=member,
                    message=(
                        "multiple or unresolved backslashes between Han "
                        f"characters={len(ambiguous_backslashes)}"
                    ),
                    excerpt=_excerpt(prose[start:end]),
                    count=len(ambiguous_backslashes),
                    recommended_action=(
                        "Review whether the backslashes are censorship artifacts "
                        "or intentional separators before editing."
                    ),
                )
            )
        ambiguous_han_spaces = list(HAN_INTERNAL_SPACE_RE.finditer(prose))
        if ambiguous_han_spaces:
            first = ambiguous_han_spaces[0]
            start = max(0, first.start() - 45)
            end = min(len(prose), first.end() + 45)
            report.issues.append(
                NormalizationIssue(
                    kind="ambiguous_han_spacing",
                    member=member,
                    message=(
                        "unresolved spaces between Han characters="
                        f"{len(ambiguous_han_spaces)}"
                    ),
                    excerpt=_excerpt(prose[start:end]),
                    count=len(ambiguous_han_spaces),
                    recommended_action=(
                        "Let Codex distinguish deliberate phrase separation "
                        "from a split Chinese word before editing."
                    ),
                )
            )
        suspicious_ascii_punctuation = list(
            re.finditer(
                rf"(?<=[{HAN_CLASS}])\?(?=[{HAN_CLASS}A-Za-z])|(?i:amp;)",
                prose,
            )
        )
        if suspicious_ascii_punctuation:
            first = suspicious_ascii_punctuation[0]
            start = max(0, first.start() - 45)
            end = min(len(prose), first.end() + 45)
            report.issues.append(
                NormalizationIssue(
                    kind="suspicious_ascii_punctuation",
                    member=member,
                    message=(
                        "ASCII punctuation may represent corruption or a "
                        f"literal token; matches={len(suspicious_ascii_punctuation)}"
                    ),
                    excerpt=_excerpt(prose[start:end]),
                    count=len(suspicious_ascii_punctuation),
                    recommended_action=(
                        "Let Codex inspect or search the source before choosing "
                        "Chinese punctuation or a different missing character."
                    ),
                )
            )
        if _is_chinese_context(prose):
            straight_double = prose.count('"')
            straight_single = len(
                re.findall(
                    rf"(?<![A-Za-z])'(?:[^'\n]*[{HAN_CLASS}][^'\n]*)?'(?![A-Za-z])",
                    prose,
                )
            )
            if straight_double:
                report.issues.append(
                    NormalizationIssue(
                        kind="unresolved_straight_quote",
                        member=member,
                        message=f"remaining straight double quotes={straight_double}",
                        excerpt=_excerpt(prose),
                        count=straight_double,
                    )
                )
            if straight_single:
                report.issues.append(
                    NormalizationIssue(
                        kind="unresolved_straight_quote",
                        member=member,
                        message=f"remaining straight single quote pairs={straight_single}",
                        excerpt=_excerpt(prose),
                        count=straight_single,
                    )
                )
        for opening, closing in QUOTE_PAIRS:
            for (
                paragraph_index,
                opening_count,
                closing_count,
            ) in _quote_mismatch_indices(review_paragraphs, opening, closing):
                report.issues.append(
                    NormalizationIssue(
                        kind="quote_mismatch",
                        member=member,
                        message=(
                            f"{opening}{closing} paragraph {paragraph_index + 1} "
                            f"counts differ: opening={opening_count}, "
                            f"closing={closing_count}"
                        ),
                        excerpt=_quote_review_context(
                            review_paragraphs,
                            paragraph_index,
                        ),
                        count=max(1, abs(opening_count - closing_count)),
                        recommended_action=(
                            "Codex should inspect this paragraph with its neighbors; "
                            "keep deliberate multi-paragraph quotations and repair "
                            "only a locally justified quote direction or omission."
                        ),
                    )
                )
