from __future__ import annotations

import html
import re
from dataclasses import dataclass
from urllib.parse import quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup

BING_SEARCH_URL = "https://www.bing.com/search"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_URL = "https://lite.duckduckgo.com/lite/"
GOOGLE_SEARCH_URL = "https://www.google.com/search"
SEARCH_TIMEOUT = 20
DEFAULT_ENGINES = ("duckduckgo", "google")
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
)


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    description: str
    engine: str


def site_search(
    query: str,
    *,
    site: str,
    path_prefix: str = "",
    limit: int = 10,
    engines: tuple[str, ...] = DEFAULT_ENGINES,
) -> list[WebSearchResult]:
    """Search for pages on one site, returning only canonical URLs under path_prefix.

    Public result-page scraping is best-effort and may be rate-limited.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    results: list[WebSearchResult] = []
    seen: set[str] = set()

    for engine in engines:
        items = _search_engine(
            session,
            engine,
            query,
            site=site,
            path_prefix=path_prefix,
            limit=limit,
        )
        for item in items:
            url = canonical_site_url(item.url, site=site, path_prefix=path_prefix)
            if not url or url in seen:
                continue
            seen.add(url)
            results.append(
                WebSearchResult(
                    title=item.title,
                    url=url,
                    description=item.description,
                    engine=item.engine,
                )
            )
            if len(results) >= limit:
                return results
    return results


def web_search(
    query: str,
    *,
    limit: int = 10,
    engines: tuple[str, ...] = ("duckduckgo", "google"),
) -> list[WebSearchResult]:
    """Run a broad web search without site filtering."""
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    results: list[WebSearchResult] = []
    seen: set[str] = set()

    for engine in engines:
        items = _web_search_engine(session, engine, query, limit=limit)
        for item in items:
            if item.url in seen:
                continue
            seen.add(item.url)
            results.append(item)
            if len(results) >= limit:
                return results
    return results


def canonical_site_url(url: str, *, site: str, path_prefix: str = "") -> str | None:
    url = html.unescape(unquote(url))
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    expected = site.lower()
    if host.startswith("www."):
        host = host[4:]
    if expected.startswith("www."):
        expected = expected[4:]
    if host != expected:
        return None
    if path_prefix and not parsed.path.startswith(path_prefix):
        return None
    path = re.sub(r"/{2,}", "/", parsed.path)
    return f"{parsed.scheme or 'https'}://www.{expected}{path}"


def _search_engine(
    session: requests.Session,
    engine: str,
    query: str,
    *,
    site: str,
    path_prefix: str,
    limit: int,
) -> list[WebSearchResult]:
    if engine == "duckduckgo":
        return _duckduckgo_search(
            session,
            query,
            site=site,
            path_prefix=path_prefix,
            limit=limit,
        )
    if engine == "bing":
        return _bing_search(
            session,
            query,
            site=site,
            path_prefix=path_prefix,
            limit=limit,
        )
    if engine == "google":
        return _google_search(
            session,
            query,
            site=site,
            path_prefix=path_prefix,
            limit=limit,
        )
    raise ValueError(f"unknown search engine: {engine}")


def _web_search_engine(
    session: requests.Session,
    engine: str,
    query: str,
    *,
    limit: int,
) -> list[WebSearchResult]:
    if engine == "duckduckgo":
        return _duckduckgo_web_search(session, query, limit=limit)
    if engine == "google":
        return _google_web_search(session, query, limit=limit)
    raise ValueError(f"unknown search engine: {engine}")


def _site_query(query: str, *, site: str, path_prefix: str = "") -> str:
    scoped_site = site.rstrip("/")
    if path_prefix:
        scoped_site += "/" + path_prefix.strip("/") + "/"
    return f"site:{scoped_site} {query}"


def _search_query_variants(query: str, *, site: str, path_prefix: str = "") -> tuple[str, ...]:
    site_hint = site.replace("www.", "")
    variants = [f"{query} {site_hint}"]
    if path_prefix:
        variants.append(f"{query} {site_hint}{path_prefix}")
    variants.append(_site_query(query, site=site, path_prefix=path_prefix))

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        deduped.append(variant)
    return tuple(deduped)


def _json_response(response: requests.Response) -> dict:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _duckduckgo_web_search(
    session: requests.Session,
    query: str,
    *,
    limit: int,
) -> list[WebSearchResult]:
    for url in (DUCKDUCKGO_HTML_URL, DUCKDUCKGO_LITE_URL):
        try:
            response = session.get(
                url,
                params={"q": query},
                timeout=SEARCH_TIMEOUT,
            )
            response.raise_for_status()
        except Exception:
            continue
        items = _duckduckgo_items(response.text)
        if items:
            return items[:limit]
    return []


def _duckduckgo_search(
    session: requests.Session,
    query: str,
    *,
    site: str,
    path_prefix: str,
    limit: int,
) -> list[WebSearchResult]:
    items: list[WebSearchResult] = []
    seen: set[str] = set()
    for search_query in _search_query_variants(query, site=site, path_prefix=path_prefix):
        for item in _duckduckgo_web_search(session, search_query, limit=limit):
            if item.url in seen:
                continue
            seen.add(item.url)
            items.append(item)
            if len(items) >= limit:
                return items
    return items


def _duckduckgo_items(html_text: str) -> list[WebSearchResult]:
    soup = BeautifulSoup(html_text, "lxml")
    items: list[WebSearchResult] = []
    for a in soup.find_all("a", href=True):
        href = _duckduckgo_result_url(a["href"])
        if not href:
            continue
        title = a.get_text(" ", strip=True)
        if not title:
            continue
        result = a.find_parent(class_=re.compile(r"result"))
        desc_node = result.select_one(".result__snippet") if result else None
        items.append(
            WebSearchResult(
                title=title,
                url=href,
                description=desc_node.get_text(" ", strip=True) if desc_node else "",
                engine="duckduckgo",
            )
        )
    return items


def _duckduckgo_result_url(href: str) -> str | None:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        target = _first_query_value(parsed.query, "uddg")
        return target or None
    if href.startswith("http"):
        return href
    return None


def _first_query_value(query: str, key: str) -> str:
    for part in query.split("&"):
        if not part.startswith(key + "="):
            continue
        return unquote(part[len(key) + 1 :])
    return ""


def _bing_search(
    session: requests.Session,
    query: str,
    *,
    site: str,
    path_prefix: str,
    limit: int,
) -> list[WebSearchResult]:
    site_query = _site_query(query, site=site, path_prefix=path_prefix)
    rss_url = f"{BING_SEARCH_URL}?format=rss&q={quote_plus(site_query)}"
    html_url = f"{BING_SEARCH_URL}?q={quote_plus(site_query)}"

    items = _bing_rss_items(session, rss_url)
    if not items:
        items = _bing_html_items(session, html_url)
    return items[:limit]


def _bing_rss_items(session: requests.Session, url: str) -> list[WebSearchResult]:
    try:
        response = session.get(url, timeout=SEARCH_TIMEOUT)
        response.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(response.text, "xml")
    items: list[WebSearchResult] = []
    for item in soup.find_all("item"):
        title = item.title.get_text(" ", strip=True) if item.title else ""
        link = item.link.get_text(" ", strip=True) if item.link else ""
        description = (
            item.description.get_text(" ", strip=True)
            if item.description
            else ""
        )
        items.append(WebSearchResult(title, link, description, "bing"))
    return items


def _bing_html_items(session: requests.Session, url: str) -> list[WebSearchResult]:
    try:
        response = session.get(url, timeout=SEARCH_TIMEOUT)
        response.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    items: list[WebSearchResult] = []
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a[href]")
        if not a:
            continue
        desc = li.select_one(".b_caption p")
        items.append(
            WebSearchResult(
                title=a.get_text(" ", strip=True),
                url=a["href"],
                description=desc.get_text(" ", strip=True) if desc else "",
                engine="bing",
            )
        )
    return items


def _google_search(
    session: requests.Session,
    query: str,
    *,
    site: str,
    path_prefix: str,
    limit: int,
) -> list[WebSearchResult]:
    site_query = _site_query(query, site=site, path_prefix=path_prefix)
    url = f"{GOOGLE_SEARCH_URL}?q={quote_plus(site_query)}&hl=zh-CN&num={limit}"
    return _google_items(session, url, limit=limit)


def _google_web_search(
    session: requests.Session,
    query: str,
    *,
    limit: int,
) -> list[WebSearchResult]:
    url = f"{GOOGLE_SEARCH_URL}?q={quote_plus(query)}&hl=zh-CN&num={limit}"
    return _google_items(session, url, limit=limit)


def _google_items(
    session: requests.Session,
    url: str,
    *,
    limit: int,
) -> list[WebSearchResult]:
    try:
        response = session.get(url, timeout=SEARCH_TIMEOUT)
        response.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(response.text, "lxml")
    items: list[WebSearchResult] = []
    for block in soup.select("div.g, div.MjjYud"):
        a = block.select_one("a[href]")
        if not a:
            continue
        href = _google_result_url(a["href"])
        if not href:
            continue
        title_node = block.select_one("h3")
        desc_node = block.select_one(".VwiC3b, .IsZvec")
        items.append(
            WebSearchResult(
                title=title_node.get_text(" ", strip=True) if title_node else a.get_text(" ", strip=True),
                url=href,
                description=desc_node.get_text(" ", strip=True) if desc_node else "",
                engine="google",
            )
        )

    if items:
        return items[:limit]

    for a in soup.find_all("a", href=True):
        href = _google_result_url(a["href"])
        if not href:
            continue
        items.append(
            WebSearchResult(
                title=a.get_text(" ", strip=True),
                url=href,
                description="",
                engine="google",
            )
        )
    return items[:limit]


def _google_result_url(href: str) -> str | None:
    if href.startswith("/url?"):
        query = urlparse(href).query
        for part in query.split("&"):
            if part.startswith("q="):
                return unquote(part[2:])
        return None
    if href.startswith("http"):
        return href
    return None
