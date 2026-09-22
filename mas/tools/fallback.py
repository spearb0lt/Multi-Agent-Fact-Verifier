"""Reading a page that refused to be read directly.

Roughly a quarter of the fetches in a real run come back 403, and they are not
a random quarter. The sites that block a plain client hardest are the ones most
worth citing: PubMed Central, the American Heart Association, MDPI. A run that
gives up on those is left building its report from whatever blog happened to be
indexed next, which is exactly how a thin report happens while every stage
reports success.

Two fallbacks, tried in order, both of which are ordinary public services
reading a public page:

* r.jina.ai, a text extraction proxy that returns the readable article as
  markdown. Fast, and handles pages that need JavaScript.
* The Wayback Machine, which serves a snapshot the publisher already allowed to
  be archived. Slower and possibly out of date, so it is second, and the date
  of the snapshot is recorded with the evidence.

What this deliberately does not do is pretend to be a browser. There is no
rotating user agent and no attempt to defeat a bot check. A 403 from a site
that does not want automated readers is respected; these are two services that
already hold or can lawfully render the page, and the run says which one it
used so a reader can judge the provenance.

robots.txt is still honoured on the original URL. A page whose robots file
disallows crawling is not fetched through a side door.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

from ..core import settings
from ..core.util import truncate
from ..core.web.base import HttpClient, SourceError
from ..core.web.extract import extract_article

JINA_PREFIX = "https://r.jina.ai/"
WAYBACK_API = "https://archive.org/wayback/available"

# Statuses that mean "not to you" rather than "not here". A 404 is a dead link
# and no proxy will conjure it back, so it is not worth a second request.
BLOCKED_STATUSES = {401, 403, 406, 429, 451, 500, 503}

# A fallback that returns almost nothing has not really worked, and accepting
# it would store a cookie banner as though it were an article.
MIN_USEFUL_CHARS = 400


def _via_jina(url: str, client: HttpClient) -> tuple[str, str, dict[str, Any]] | None:
    """Text extraction proxy. Returns (body, title, meta) or None."""
    try:
        response = client.get(
            f"{JINA_PREFIX}{url}",
            accept="text/plain",
            # The proxy is not the origin, so the origin's robots rule was
            # already checked before we got here.
            allow_robots_override=True,
        )
    except SourceError:
        return None
    if not response.ok:
        return None

    text = (response.text or "").strip()
    if len(text) < MIN_USEFUL_CHARS:
        return None

    # The proxy prefixes a small header block. The title is worth keeping and
    # the rest is noise once the body has been separated from it.
    title = ""
    body = text
    if text.startswith("Title:"):
        head, _, rest = text.partition("\n")
        title = head[len("Title:"):].strip()
        body = rest.strip()
        for marker in ("Markdown Content:", "Content:"):
            if marker in body:
                body = body.split(marker, 1)[1].strip()
                break
    return body, title, {"via": "jina"}


def _via_wayback(url: str, client: HttpClient) -> tuple[str, str, dict[str, Any]] | None:
    """The newest archived snapshot, when one exists."""
    try:
        payload = client.get_json(
            WAYBACK_API, params={"url": url}, accept="application/json"
        )
    except SourceError:
        return None

    snapshot = ((payload or {}).get("archived_snapshots") or {}).get("closest") or {}
    if not snapshot.get("available") or not snapshot.get("url"):
        return None

    try:
        response = client.get(snapshot["url"], accept="text/html", allow_robots_override=True)
    except SourceError:
        return None
    if not response.ok:
        return None

    article = extract_article(response.text, url)
    body = str(article.get("content") or "").strip()
    if len(body) < MIN_USEFUL_CHARS:
        return None

    stamp = str(snapshot.get("timestamp") or "")
    readable = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}" if len(stamp) >= 8 else ""
    return (
        body,
        str(article.get("title") or ""),
        {"via": "wayback", "snapshot_date": readable, "snapshot_url": snapshot["url"]},
    )


def read_blocked_page(
    url: str, client: HttpClient, *, status: int = 0
) -> tuple[str, str, dict[str, Any]] | None:
    """Try the fallbacks in order. Returns (body, title, meta), or None.

    Called only after a direct fetch has already failed, so the cost of trying
    is paid exactly on the pages that would otherwise have been lost.
    """
    if not settings.env_bool("ENABLE_FETCH_FALLBACK", True):
        return None
    if status and status not in BLOCKED_STATUSES:
        return None

    for attempt in (_via_jina, _via_wayback):
        try:
            result = attempt(url, client)
        except Exception:
            result = None
        if result:
            return result
    return None


def describe(meta: dict[str, Any]) -> str:
    """How the evidence should say it was obtained."""
    via = meta.get("via", "")
    if via == "jina":
        return "read through a text extraction proxy after the site refused a direct request"
    if via == "wayback":
        date = meta.get("snapshot_date")
        return f"read from the Wayback Machine snapshot{f' of {date}' if date else ''}"
    return ""


def quoted(url: str) -> str:
    return quote(url, safe=":/?&=#%")


def summarise(meta: dict[str, Any]) -> str:
    return truncate(describe(meta), 120)
