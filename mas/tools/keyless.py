"""Search backends that need no credential.

The paid search APIs are better, and none of them is required. Google News
answers any query as RSS and Wikipedia answers any query as JSON, both without
a key, which means a clone of this repository with an empty .env still does
real research rather than printing a message about missing configuration.

That is not only a convenience. It is also the honest baseline: when a run
produces a thin report it should be possible to tell whether the agents did
badly or simply had nothing but Google News to work with, and the only way to
know is for the keyless path to be a first class backend that reports itself.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from ..core import settings
from ..core.util import parse_datetime_iso, strip_html, truncate
from ..core.web.base import HttpClient, RawItem, SourceError

GOOGLE_NEWS = "https://news.google.com/rss/search"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def google_news(query: str, *, limit: int = 20, client: HttpClient | None = None) -> list[RawItem]:
    """Search Google News. No key, no quota, worldwide publisher coverage."""
    import feedparser

    client = client or HttpClient()
    url = f"{GOOGLE_NEWS}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    # Google News does not publish a robots rule for this feed and serves it to
    # any client, but it is still a fetch through the shared client so the per
    # host delay and the timeout apply exactly as they do everywhere else.
    response = client.get(url, accept="application/rss+xml", allow_robots_override=True)
    if not response.ok:
        raise SourceError(f"Google News returned HTTP {response.status}.")

    parsed = feedparser.parse(response.text)
    items: list[RawItem] = []
    for entry in parsed.entries[: max(1, limit)]:
        link = str(getattr(entry, "link", "") or "")
        if not link:
            continue
        # Google News wraps every link in its own redirector and puts the
        # publisher in a sub element, so the useful source name is there rather
        # than in the host of the URL.
        source = ""
        raw_source = getattr(entry, "source", None)
        if raw_source is not None:
            source = str(getattr(raw_source, "title", "") or "")
        items.append(
            RawItem(
                url=link,
                title=str(getattr(entry, "title", "") or ""),
                summary=strip_html(str(getattr(entry, "summary", "") or "")),
                published_at=parse_datetime_iso(getattr(entry, "published", "")),
                meta={"backend": "googlenews", "query": query, "publisher": source},
            )
        )
    return items


def wikipedia(query: str, *, limit: int = 5, client: HttpClient | None = None) -> list[RawItem]:
    """Search Wikipedia. Useful for the background half of almost any brief."""
    client = client or HttpClient()
    payload = client.get_json(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max(1, min(limit, 20)),
            "format": "json",
            "srprop": "snippet|timestamp",
        },
        accept="application/json",
    )
    results = ((payload or {}).get("query") or {}).get("search") or []
    items: list[RawItem] = []
    for row in results:
        title = str(row.get("title") or "")
        if not title:
            continue
        items.append(
            RawItem(
                url=f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}",
                title=title,
                summary=strip_html(str(row.get("snippet") or "")),
                published_at=parse_datetime_iso(row.get("timestamp")),
                meta={"backend": "wikipedia", "query": query},
            )
        )
    return items


def wikipedia_extract(title: str, *, client: HttpClient | None = None) -> str:
    """The plain text lead of one Wikipedia article."""
    client = client or HttpClient()
    payload = client.get_json(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "titles": title,
            "format": "json",
            "redirects": "1",
        },
        accept="application/json",
    )
    pages = ((payload or {}).get("query") or {}).get("pages") or {}
    for page in pages.values():
        text = str(page.get("extract") or "").strip()
        if text:
            return text
    return ""


def available() -> list[str]:
    return ["googlenews", "wikipedia"] if settings.ENABLE_KEYLESS_SEARCH else []


def describe(items: list[RawItem]) -> list[dict[str, Any]]:
    return [
        {
            "url": item.url,
            "title": item.title,
            "snippet": truncate(item.summary, 400),
            "published_at": item.published_at,
            "backend": item.meta.get("backend", ""),
        }
        for item in items
    ]
