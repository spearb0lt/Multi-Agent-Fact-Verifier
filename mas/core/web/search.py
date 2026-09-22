"""Web search as a source, for the stories no configured feed carried.

Feeds and sitemaps only cover publishers somebody thought to add. A standing
query against a search backend is the safety net: it finds the outlet nobody
subscribed to, on the day it matters.

One adapter covers five backends rather than five adapters, because a user does
not care which of them answers. They care that their query runs. So the adapter
dispatches to whichever backend has a key, in a fixed preference order, and
says plainly when none does. SearXNG is last but needs no key at all, which
makes a self hosted instance the only fully free option here.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from .. import settings
from ..llm import keyring
from ..util import parse_datetime_iso, strip_html
from .base import (
    FetchOutcome,
    RawItem,
    SourceAdapter,
    SourceError,
    dedupe_items,
    register,
    within_window,
)

# Preference order. SerpAPI first because it returns Google's own news
# ranking, then the two that index news natively, then the generic ones.
BACKENDS = ("serpapi", "brave", "tavily", "exa", "searxng")

_BACKEND_LABELS = {
    "serpapi": "SerpAPI",
    "brave": "Brave Search",
    "tavily": "Tavily",
    "exa": "Exa",
    "searxng": "SearXNG",
}


def _key_for(slug: str, settings_attr: str) -> str:
    """A visitor's own key first, then the server's, as everywhere else."""
    return (keyring.key_for(slug) or getattr(settings, settings_attr, "") or "").strip()


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> Any:
    """A JSON POST, which the shared GET only client cannot express.

    Tavily and Exa accept no GET form at all. Nothing else is given up by
    calling requests here: a search API is not subject to robots.txt, and one
    request per run needs no host pacing.
    """
    import requests  # noqa: PLC0415

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", **headers},
            timeout=float(settings.FETCH_TIMEOUT),
        )
    except Exception as exc:  # noqa: BLE001
        raise SourceError(f"Could not reach {urlsplit(url).netloc}: {exc}") from exc

    if response.status_code == 429:
        raise SourceError(
            f"{urlsplit(url).netloc} rate limited this search.",
            hint="Free search tiers are measured in requests a month, not a day.",
        )
    if response.status_code in (401, 403):
        raise SourceError(
            f"{urlsplit(url).netloc} rejected the API key.",
            retryable=False,
            hint=response.text[:200],
        )
    if response.status_code >= 400:
        raise SourceError(
            f"{urlsplit(url).netloc} returned HTTP {response.status_code}.",
            retryable=response.status_code >= 500,
            hint=response.text[:200],
        )
    try:
        return response.json()
    except ValueError as exc:
        raise SourceError(f"{urlsplit(url).netloc} did not return JSON.") from exc


@register
class SearchAdapter(SourceAdapter):
    kind = "search"
    label = "Web search"
    urlless = True
    url_hint = "Leave blank and set `query` in the config"
    key_names = (
        "SERPAPI_KEY",
        "BRAVE_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
        "SEARXNG_BASE_URL",
    )

    def available_backends(self) -> list[str]:
        out: list[str] = []
        if _key_for("serpapi", "SERPAPI_KEY"):
            out.append("serpapi")
        if _key_for("brave", "BRAVE_API_KEY"):
            out.append("brave")
        if _key_for("tavily", "TAVILY_API_KEY"):
            out.append("tavily")
        if _key_for("exa", "EXA_API_KEY"):
            out.append("exa")
        if (settings.SEARXNG_BASE_URL or "").strip():
            out.append("searxng")
        return out

    def fetch(
        self,
        url: str,
        *,
        config: dict[str, Any] | None = None,
        etag: str = "",
        last_modified: str = "",
        limit: int = 0,
        since: str = "",
    ) -> FetchOutcome:
        config = config or {}
        query = str(config.get("query") or config.get("q") or "").strip()
        if not query and url:
            query = url.strip()
        if not query:
            raise SourceError(
                "A search source needs a query.",
                retryable=False,
                hint="Set `query` in the source config, for example: RBI repo rate",
            )

        cap = limit or int(config.get("limit") or 0) or 20
        cap = max(1, min(cap, 50))

        available = self.available_backends()
        if not available:
            return FetchOutcome(
                items=[],
                status="skipped",
                error=(
                    "No web search backend is configured. Set one of "
                    "SERPAPI_KEY, BRAVE_API_KEY, TAVILY_API_KEY, EXA_API_KEY, "
                    "or SEARXNG_BASE_URL for a self hosted instance."
                ),
            )

        wanted = str(config.get("backend") or "").strip().lower()
        if wanted and wanted not in BACKENDS:
            raise SourceError(
                f"Unknown search backend '{wanted}'.",
                retryable=False,
                hint=f"Choose one of: {', '.join(BACKENDS)}.",
            )
        if wanted and wanted not in available:
            return FetchOutcome(
                items=[],
                status="skipped",
                error=(
                    f"{_BACKEND_LABELS[wanted]} is selected but has no key "
                    f"configured. Configured backends: {', '.join(available)}."
                ),
            )

        backend = wanted or next(name for name in BACKENDS if name in available)
        runner = {
            "serpapi": self._serpapi,
            "brave": self._brave,
            "tavily": self._tavily,
            "exa": self._exa,
            "searxng": self._searxng,
        }[backend]

        items = runner(query, config, cap, since)
        for item in items:
            item.meta.setdefault("query", query)
            item.meta.setdefault("backend", backend)
        items = dedupe_items(within_window(items, since))[:cap]

        return FetchOutcome(
            items=items,
            status="ok",
            notes={"backend": backend, "query": query, "available": available},
        )

    # ------------------------------------------------------------- backends

    def _serpapi(
        self, query: str, config: dict[str, Any], limit: int, since: str
    ) -> list[RawItem]:
        payload = self.client.get_json(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_news",
                "q": query,
                "hl": str(config.get("language") or "en"),
                "gl": str(config.get("country") or "in").lower(),
                "num": limit,
                "api_key": _key_for("serpapi", "SERPAPI_KEY"),
            },
        )
        rows = list(
            (payload or {}).get("news_results") or (payload or {}).get("organic_results") or []
        )
        # Google News groups a story with the rest of its coverage, and the
        # group's own entry carries no link, so the nested stories are what
        # actually gets read.
        flattened: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            nested = row.get("stories")
            if isinstance(nested, list) and nested:
                flattened.extend(entry for entry in nested if isinstance(entry, dict))
            else:
                flattened.append(row)

        items: list[RawItem] = []
        for row in flattened:
            link = str(row.get("link") or "")
            if not link:
                continue
            source = row.get("source") or {}
            publisher = source.get("name") if isinstance(source, dict) else str(source or "")
            thumbnail = row.get("thumbnail")
            items.append(
                RawItem(
                    url=link,
                    title=str(row.get("title") or ""),
                    summary=str(row.get("snippet") or ""),
                    published_at=parse_datetime_iso(row.get("date") or ""),
                    image_url=str(thumbnail or "") if isinstance(thumbnail, str) else "",
                    meta={"publisher": str(publisher or ""), "position": row.get("position")},
                )
            )
        return items

    def _brave(
        self, query: str, config: dict[str, Any], limit: int, since: str
    ) -> list[RawItem]:
        payload = self.client.get_json(
            "https://api.search.brave.com/res/v1/news/search",
            params={
                "q": query,
                "count": min(50, limit),
                "country": str(config.get("country") or "IN").upper(),
                "search_lang": str(config.get("language") or "en"),
                "freshness": str(config.get("freshness") or "pd"),
            },
            headers={
                "X-Subscription-Token": _key_for("brave", "BRAVE_API_KEY"),
                "Accept": "application/json",
            },
        )
        items: list[RawItem] = []
        for row in (payload or {}).get("results") or []:
            link = str(row.get("url") or "")
            if not link:
                continue
            thumbnail = row.get("thumbnail") or {}
            meta_url = row.get("meta_url") or {}
            items.append(
                RawItem(
                    url=link,
                    title=strip_html(str(row.get("title") or "")),
                    summary=strip_html(str(row.get("description") or "")),
                    published_at=parse_datetime_iso(row.get("page_age") or row.get("age") or ""),
                    image_url=str(thumbnail.get("src") or "") if isinstance(thumbnail, dict) else "",
                    meta={
                        "publisher": str(meta_url.get("hostname") or "")
                        if isinstance(meta_url, dict)
                        else "",
                        "age": str(row.get("age") or ""),
                    },
                )
            )
        return items

    def _tavily(
        self, query: str, config: dict[str, Any], limit: int, since: str
    ) -> list[RawItem]:
        key = _key_for("tavily", "TAVILY_API_KEY")
        payload = _post_json(
            "https://api.tavily.com/search",
            {
                "query": query,
                "topic": str(config.get("topic") or "news"),
                "max_results": min(20, limit),
                "days": int(config.get("days") or 2),
                "include_answer": False,
                "search_depth": str(config.get("depth") or "basic"),
                # Older deployments of the API read the key from the body while
                # the current one reads the header, so both are sent.
                "api_key": key,
            },
            {"Authorization": f"Bearer {key}"},
        )
        items: list[RawItem] = []
        for row in (payload or {}).get("results") or []:
            link = str(row.get("url") or "")
            if not link:
                continue
            items.append(
                RawItem(
                    url=link,
                    title=str(row.get("title") or ""),
                    summary=strip_html(str(row.get("content") or ""))[:2000],
                    published_at=parse_datetime_iso(row.get("published_date") or ""),
                    meta={"relevance": row.get("score")},
                )
            )
        return items

    def _exa(
        self, query: str, config: dict[str, Any], limit: int, since: str
    ) -> list[RawItem]:
        body: dict[str, Any] = {
            "query": query,
            "numResults": min(25, limit),
            "category": str(config.get("category") or "news"),
            "contents": {"text": {"maxCharacters": 4000}},
        }
        start = parse_datetime_iso(since)
        if start:
            body["startPublishedDate"] = start
        payload = _post_json(
            "https://api.exa.ai/search",
            body,
            {"x-api-key": _key_for("exa", "EXA_API_KEY")},
        )
        items: list[RawItem] = []
        for row in (payload or {}).get("results") or []:
            link = str(row.get("url") or "")
            if not link:
                continue
            items.append(
                RawItem(
                    url=link,
                    title=str(row.get("title") or ""),
                    content=strip_html(str(row.get("text") or "")),
                    author=str(row.get("author") or ""),
                    published_at=parse_datetime_iso(row.get("publishedDate") or ""),
                    image_url=str(row.get("image") or ""),
                    meta={"relevance": row.get("score")},
                )
            )
        return items

    def _searxng(
        self, query: str, config: dict[str, Any], limit: int, since: str
    ) -> list[RawItem]:
        base = (settings.SEARXNG_BASE_URL or "").strip().rstrip("/")
        try:
            payload = self.client.get_json(
                f"{base}/search",
                params={
                    "q": query,
                    "format": "json",
                    "categories": str(config.get("categories") or "news"),
                    "language": str(config.get("language") or "en"),
                    "time_range": str(config.get("time_range") or "day"),
                },
            )
        except SourceError as exc:
            raise SourceError(
                f"The SearXNG instance at {base} could not be searched: {exc.message}",
                hint=(
                    "Most instances disable the JSON format by default. Add "
                    "'json' to search.formats in settings.yml on your instance."
                ),
            ) from exc

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {}
        items: list[RawItem] = []
        for row in (payload or {}).get("results") or []:
            link = str(row.get("url") or "")
            if not link:
                continue
            items.append(
                RawItem(
                    url=link,
                    title=str(row.get("title") or ""),
                    summary=strip_html(str(row.get("content") or "")),
                    published_at=parse_datetime_iso(row.get("publishedDate") or ""),
                    image_url=str(row.get("img_src") or ""),
                    meta={"engine": str(row.get("engine") or "")},
                )
            )
        return items
