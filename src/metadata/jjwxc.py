from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


TIME_AREAS = ("近代现代", "古色古香", "架空历史", "幻想未来")
GENRES = (
    "爱情",
    "武侠",
    "奇幻",
    "仙侠",
    "游戏",
    "传奇",
    "科幻",
    "童话",
    "惊悚",
    "悬疑",
    "剧情",
    "轻小说",
    "古典衍生",
    "东方衍生",
    "西方衍生",
    "其他衍生",
)

ARTICLE_TYPE_RE = re.compile(
    r"文章类型：</span>\s*<span[^>]*itemprop=[\"']genre[\"'][^>]*>\s*(.*?)\s*</span>",
    re.S,
)
TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class JjwxcTypeMetadata:
    article_type: str
    time_area: str
    genre: str
    source: str


def novel_id_from_url(value: str) -> str:
    parsed = urlparse(value)
    query = parse_qs(parsed.query)
    novel_id = query.get("novelid", [""])[0]
    if novel_id:
        return novel_id
    if value.isdigit():
        return value
    return ""


def parse_article_type(value: str, *, source: str = "jjwxc") -> JjwxcTypeMetadata | None:
    cleaned = TAG_RE.sub("", value or "")
    cleaned = re.sub(r"\s+", "", cleaned)
    if not cleaned:
        return None
    parts = [part for part in cleaned.split("-") if part]
    time_area = next((part for part in parts if part in TIME_AREAS), "")
    genre = next((part for part in parts if part in GENRES), "")
    if not time_area and len(parts) >= 3 and parts[2] in TIME_AREAS:
        time_area = parts[2]
    if not genre and len(parts) >= 4 and parts[3] in GENRES:
        genre = parts[3]
    if not genre:
        genre = infer_genre(cleaned)
    if not time_area:
        time_area = infer_time_area(cleaned)
    return JjwxcTypeMetadata(cleaned, time_area, genre, source)


def parse_jjwxc_book_page(html: str) -> JjwxcTypeMetadata | None:
    match = ARTICLE_TYPE_RE.search(html)
    if not match:
        return None
    return parse_article_type(match.group(1), source="jjwxc")


def fetch_jjwxc_type(value: str, *, timeout: int = 20) -> JjwxcTypeMetadata | None:
    novel_id = novel_id_from_url(value)
    if not novel_id:
        return None
    req = Request(
        f"https://www.jjwxc.net/onebook.php?novelid={novel_id}",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    data = urlopen(req, timeout=timeout).read()
    html = data.decode("gb18030", "replace")
    return parse_jjwxc_book_page(html)


def infer_time_area(text: str) -> str:
    if re.search(r"星际|未来|末世|虫族|机甲|AI|机器人|宇宙|银河", text, re.I):
        return "幻想未来"
    if re.search(r"架空|皇|帝|王爷|朝堂|宫廷|将军|侯|国师|锦衣卫|战国|天宝|江山", text):
        return "架空历史"
    if re.search(r"古代|江湖|武林|修仙|仙尊|神魔|妖|剑|侠|山海", text):
        return "古色古香"
    return "近代现代"


def infer_genre(text: str) -> str:
    if re.search(r"无限|副本|逃生|恐怖|鬼|死亡|梦魇|惊悚", text):
        return "惊悚"
    if re.search(r"悬疑|刑侦|破案|案件|犯罪|律师|审判|侦探", text):
        return "悬疑"
    if re.search(r"电竞|网游|游戏|直播|玩家|系统", text):
        return "游戏"
    if re.search(r"星际|未来|末世|虫族|机甲|机器人|AI|科幻", text, re.I):
        return "科幻"
    if re.search(r"修仙|仙侠|仙尊|灵根|飞升", text):
        return "仙侠"
    if re.search(r"武侠|江湖|武林|剑客|侠", text):
        return "武侠"
    if re.search(r"奇幻|魔|妖|异能|灵魂|神怪", text):
        return "奇幻"
    if "童话" in text:
        return "童话"
    if "传奇" in text:
        return "传奇"
    if "轻小说" in text:
        return "轻小说"
    if "衍生" in text:
        return "其他衍生"
    if "剧情" in text:
        return "剧情"
    return "爱情"


def classify_from_text(title: str, author: str = "", intro: str = "", sample: str = "") -> JjwxcTypeMetadata:
    text = "\n".join(part for part in [title, author, intro, sample[:4000]] if part)
    time_area = infer_time_area(text)
    genre = infer_genre(text)
    return JjwxcTypeMetadata("", time_area, genre, "heuristic")
