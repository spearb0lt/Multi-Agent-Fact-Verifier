"""The contract every news source implements, and the HTTP client they share.

A source adapter turns some remote thing, an RSS feed, a sitemap, a search API,
a subreddit, into a list of `RawItem`. It does not deduplicate, classify,
summarise or store: those are pipeline stages that work the same way whatever
the item came from. Keeping adapters this thin is what makes adding a source a
small job.

The shared client is where politeness lives. Every outbound request goes
through it, so robots.txt, the per host delay, conditional requests and the
size cap are enforced once rather than remembered by each adapter.
"""
from __future__ import annotations

import re
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib import robotparser
from urllib.parse import urljoin, urlsplit

from .. import settings
from ..util import host_of, normalise_url, parse_datetime_iso, strip_html, truncate


@dataclass
class RawItem:
    """One story as a source reported it, before any processing."""

    url: str
    title: str = ""
    summary: str = ""
    content: str = ""
    author: str = ""
    published_at: str = ""  # ISO 8601 UTC, empty when the source gave none
    image_url: str = ""
    language: str = ""
    # Anything source specific worth keeping: score and comment count from
    # Reddit or Hacker News, the section from a sitemap, the query that found
    # it from a search API.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.url = (self.url or "").strip()
        # An aggregator that appends " - Publisher" to every headline leaves a
        # dangling separator when the headline itself was empty, which reaches
        # the digest as a story called "- AP News".
        self.title = strip_html(self.title or "").strip().strip("-|• ").strip()
        self.summary = truncate(strip_html(self.summary or "").strip(), 2000)
        self.author = strip_html(self.author or "").strip()[:200]

    @property
    def domain(self) -> str:
        return host_of(self.url)

    def is_usable(self) -> bool:
        """Enough to be worth storing: a real link and something to call it."""
        if not self.url.startswith(("http://", "https://")):
            return False
        if self.summary or self.content:
            return True
        # With no body at all, the title is the entire story, so a fragment
        # like a bare publisher name is not enough to be worth a digest slot.
        return len(self.title) >= 12


@dataclass
class FetchOutcome:
    """What one source run produced, including why it produced nothing."""

    items: list[RawItem] = field(default_factory=list)
    status: str = "ok"  # ok, unchanged, error, skipped
    error: str = ""
    etag: str = ""
    last_modified: str = ""
    # Carried back to the sources row so the next run can be conditional.
    notes: dict[str, Any] = field(default_factory=dict)


class SourceError(RuntimeError):
    """A source could not be read. Carries a reason worth showing a user."""

    def __init__(self, message: str, *, retryable: bool = True, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.hint = hint


# --------------------------------------------------------------- HTTP client


class RobotsCache:
    """robots.txt per host, fetched once and kept for the process's lifetime.

    A host that does not serve robots.txt, or serves an error, is treated as
    allowing the fetch. That is the conventional reading, and the alternative
    would silently disable a source because of an unrelated outage.
    """

    def __init__(self, *, timeout: float = 8.0) -> None:
        self._parsers: dict[str, robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()
        self._timeout = timeout

    def allows(self, url: str, user_agent: str) -> bool:
        if not settings.RESPECT_ROBOTS:
            return True
        try:
            parts = urlsplit(url)
        except ValueError:
            return False
        if not parts.netloc:
            return False
        root = f"{parts.scheme}://{parts.netloc}"

        with self._lock:
            cached = self._parsers.get(root, "missing")  # type: ignore[arg-type]
        if cached == "missing":  # type: ignore[comparison-overlap]
            parser = self._load(root)
            with self._lock:
                self._parsers[root] = parser
        else:
            parser = cached  # type: ignore[assignment]

        if parser is None:
            return True
        try:
            return parser.can_fetch(user_agent, url)
        except Exception:  # noqa: BLE001 - a malformed file should not block
            return True

    def crawl_delay(self, url: str, user_agent: str) -> float:
        try:
            root = f"{urlsplit(url).scheme}://{urlsplit(url).netloc}"
        except ValueError:
            return 0.0
        with self._lock:
            parser = self._parsers.get(root)
        if parser is None:
            return 0.0
        try:
            delay = parser.crawl_delay(user_agent)
            return float(delay) if delay else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _load(self, root: str) -> robotparser.RobotFileParser | None:
        try:
            import requests

            response = requests.get(
                urljoin(root, "/robots.txt"),
                timeout=self._timeout,
                headers={"User-Agent": settings.USER_AGENT},
            )
            if response.status_code >= 400:
                return None
            parser = robotparser.RobotFileParser()
            parser.parse(response.text.splitlines())
            return parser
        except Exception:  # noqa: BLE001 - absence means no restriction
            return None


_robots = RobotsCache()


@dataclass
class Response:
    url: str
    status: int
    text: str = ""
    content: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    @property
    def unchanged(self) -> bool:
        return self.status == 304


class HttpClient:
    """The one way this app talks to the open web.

    Politeness is enforced here so no adapter can forget it: robots.txt is
    consulted, a per host delay is observed, the response is capped so a
    surprise multi megabyte page cannot exhaust a serverless function's memory,
    and conditional headers are sent when the caller has them.
    """

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout: float | None = None,
        respect_robots: bool | None = None,
    ) -> None:
        self.user_agent = user_agent or settings.USER_AGENT
        self.timeout = timeout if timeout is not None else float(settings.FETCH_TIMEOUT)
        self.respect_robots = (
            settings.RESPECT_ROBOTS if respect_robots is None else respect_robots
        )
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()

    def _wait_turn(self, url: str) -> None:
        """Space out requests to one host without blocking requests to others."""
        host = host_of(url)
        if not host:
            return
        delay = max(settings.PER_HOST_DELAY, _robots.crawl_delay(url, self.user_agent))
        if delay <= 0:
            return
        with self._lock:
            previous = self._last_hit.get(host, 0.0)
            now = time.monotonic()
            wait = previous + delay - now
            self._last_hit[host] = now + max(0.0, wait)
        if wait > 0:
            time.sleep(min(wait, 10.0))

    def get(
        self,
        url: str,
        *,
        etag: str = "",
        last_modified: str = "",
        accept: str = "",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        allow_robots_override: bool = False,
    ) -> Response:
        import requests

        if (
            self.respect_robots
            and not allow_robots_override
            and not _robots.allows(url, self.user_agent)
        ):
            raise SourceError(
                f"robots.txt at {host_of(url)} disallows this path.",
                retryable=False,
                hint=(
                    "Set RESPECT_ROBOTS=0 only if you have permission from the "
                    "site owner. The site's own RSS feed is usually allowed."
                ),
            )

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "en,*;q=0.5",
            "Accept": accept or "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if etag:
            request_headers["If-None-Match"] = etag
        if last_modified:
            request_headers["If-Modified-Since"] = last_modified
        if headers:
            request_headers.update(headers)

        self._wait_turn(url)
        try:
            response = requests.get(
                url,
                headers=request_headers,
                params=params,
                timeout=self.timeout,
                stream=True,
                allow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - normalised for the caller
            raise SourceError(f"Could not reach {host_of(url) or url}: {exc}") from exc

        if response.status_code == 304:
            response.close()
            return Response(url=url, status=304, headers=dict(response.headers))

        # Read with a cap. A single oversized page must not be able to exhaust
        # the memory a serverless function is allowed.
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total > settings.MAX_ARTICLE_BYTES:
                    break
        except Exception as exc:  # noqa: BLE001
            raise SourceError(f"Download from {host_of(url)} failed: {exc}") from exc
        finally:
            response.close()

        body = b"".join(chunks)
        # Two traps here, and both produce mangled text rather than an error.
        #
        # requests' own apparent_encoding sniffs response.content, and reading
        # that raises once the stream above has been consumed, so the guess is
        # made from the bytes already in hand instead.
        #
        # And requests follows RFC 2616 by defaulting any text/* response with
        # no charset to ISO-8859-1. That default is almost always wrong today
        # and it is not None, so trusting response.encoding silently corrupts
        # every non ASCII character on a UTF-8 page whose server omitted the
        # charset. The header is therefore only believed when it genuinely
        # carries one, and the document's own declaration wins otherwise.
        declared = "charset=" in (response.headers.get("content-type") or "").lower()
        encoding = response.encoding if declared and response.encoding else _sniff_encoding(body)
        try:
            text = body.decode(encoding, errors="replace")
        except (LookupError, UnicodeDecodeError):
            text = body.decode("utf-8", errors="replace")

        return Response(
            url=str(response.url),
            status=response.status_code,
            text=text,
            content=body,
            headers={k.lower(): v for k, v in response.headers.items()},
        )

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Fetch a JSON API. API endpoints are not subject to robots.txt."""
        response = self.get(
            url,
            params=params,
            headers=headers,
            accept="application/json",
            allow_robots_override=True,
        )
        if not response.ok:
            raise SourceError(
                f"{host_of(url)} returned HTTP {response.status}.",
                retryable=response.status in (429, 500, 502, 503, 504),
                hint=_status_hint(response.status, response.text),
            )
        import json

        try:
            return json.loads(response.text)
        except ValueError as exc:
            raise SourceError(
                f"{host_of(url)} did not return JSON.", hint=response.text[:200]
            ) from exc


_CHARSET_RE = re.compile(rb"""(?:charset|encoding)\s*=\s*["']?([A-Za-z0-9_.:+-]+)""", re.I)


def _sniff_encoding(body: bytes) -> str:
    """The charset a document declares about itself, defaulting to UTF-8.

    Both an HTML meta tag and an XML declaration name their encoding in the
    first few hundred bytes, which is all that is needed to decode a page the
    server described only as text.
    """
    match = _CHARSET_RE.search(body[:4096])
    if not match:
        return "utf-8"
    name = match.group(1).decode("ascii", errors="ignore").strip().strip("\"'")
    return name or "utf-8"


def _status_hint(status: int, body: str) -> str:
    if status == 401 or status == 403:
        return "The API key was rejected, or this endpoint needs a paid plan."
    if status == 429:
        return "The free tier's rate limit is used up. It resets on the provider's schedule."
    if status == 404:
        return "The endpoint or resource does not exist. Check the URL or the query."
    if status >= 500:
        return "The provider is having trouble. This usually resolves on its own."
    return body[:200]


# ------------------------------------------------------------- adapter base


class SourceAdapter:
    """One kind of source. Instances are cheap and created per run."""

    kind: str = ""
    label: str = ""
    # What the `url` column means for this kind, shown in the UI's add-source form.
    url_hint: str = ""
    # True when the adapter needs no `url` at all, because the source is
    # defined entirely by its config (a search query, a subreddit name).
    urlless: bool = False
    key_names: tuple[str, ...] = ()
    # Capability this adapter needs, checked before it runs.
    requires_capability: str = ""

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    @classmethod
    def validate_url(cls, url: str) -> str:
        """Why this value is not usable as this kind's `url`, or "" if it is.

        Checked when a source is created, because the alternative is storing
        the typo and reporting it once a day as a fetch failure, which reads
        like the site is down rather than like the address is wrong.
        """
        candidate = (url or "").strip()
        if not candidate:
            return "" if cls.urlless else "A URL is required."
        parsed = urlsplit(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "That is not a web address. It needs to start with http:// or https://."
        return ""

    def is_available(self) -> tuple[bool, str]:
        """Whether this adapter can run here, and why not when it cannot."""
        if self.requires_capability:
            from ..runtime import current

            rt = current()
            if not rt.can(self.requires_capability):
                return False, rt.reason(self.requires_capability)
        return True, ""

    def fetch(
        self,
        url: str,
        *,
        config: dict[str, Any] | None = None,
        etag: str = "",
        last_modified: str = "",
        limit: int = 0,
        since: str = "",
    ) -> FetchOutcome:  # pragma: no cover - interface
        raise NotImplementedError


_ADAPTERS: dict[str, type[SourceAdapter]] = {}


def register(adapter_cls: type[SourceAdapter]) -> type[SourceAdapter]:
    """Class decorator that makes an adapter reachable by its `kind`."""
    if not adapter_cls.kind:
        raise ValueError(f"{adapter_cls.__name__} has no kind.")
    _ADAPTERS[adapter_cls.kind] = adapter_cls
    return adapter_cls


def adapter_for(kind: str) -> type[SourceAdapter] | None:
    return _ADAPTERS.get((kind or "").strip().lower())


def all_adapters() -> dict[str, type[SourceAdapter]]:
    return dict(_ADAPTERS)


def describe_adapters() -> list[dict[str, Any]]:
    """What the UI needs to render the add-source form."""
    out: list[dict[str, Any]] = []
    for kind, cls in sorted(_ADAPTERS.items()):
        instance = cls()
        available, reason = instance.is_available()
        out.append(
            {
                "kind": kind,
                "label": cls.label or kind,
                "url_hint": cls.url_hint,
                "urlless": cls.urlless,
                "key_names": list(cls.key_names),
                "available": available,
                "reason": reason,
            }
        )
    return out


def dedupe_items(items: Iterable[RawItem]) -> list[RawItem]:
    """Drop items a single source listed more than once.

    A feed that paginates, or a search that runs several queries, commonly
    returns the same URL twice. Removing those here keeps the count a source
    reports honest and saves the pipeline the work.
    """
    seen: set[str] = set()
    out: list[RawItem] = []
    for item in items:
        if not item.is_usable():
            continue
        key = normalise_url(item.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def within_window(items: Sequence[RawItem], since_iso: str) -> list[RawItem]:
    """Keep items published after `since_iso`.

    An item with no date is kept. Feeds often omit the date, and dropping those
    would silently lose whole sources; the pipeline instead treats the fetch
    time as the publication time later on.
    """
    if not since_iso:
        return list(items)
    out: list[RawItem] = []
    for item in items:
        stamp = parse_datetime_iso(item.published_at) if item.published_at else ""
        if not stamp or stamp >= since_iso:
            out.append(item)
    return out
