"""The tools that reach the open web: search, fetch, and reading back.

These are the capabilities that make the Researcher and the Fact Checker able
to do something rather than recall something, so they are also where most of a
run's wall clock and all of its exposure to the outside world lives.

Three decisions worth stating.

Search fans out across every configured backend rather than picking one. The
backends disagree substantially about what is relevant, their rankings are not
comparable, and merging them and re-ranking on the query is strictly better
than trusting whichever one happened to be configured first.

Fetching writes to the evidence table, not to the agent. What comes back to the
model is a reference and a summary; the full text stays in the database. A page
body is thousands of tokens, the agent needs to know what is in it rather than
recite it, and anything that later needs the text (the Fact Checker, the
citation renderer) reads the row.

Nothing here trusts robots.txt to be someone else's problem. Fetching goes
through the shared client, which consults robots, paces per host and caps the
response size, so no tool can forget to be polite.
"""
from __future__ import annotations

from typing import Any

from ..core import settings
from ..core.util import host_of, simhash, truncate
from ..core.web.base import HttpClient, RawItem, SourceError, dedupe_items
from ..core.web.extract import extract_article
from ..core.web.search import SearchAdapter
from ..kernel.tool import Tool, ToolError, tool
from . import keyless
from .rank import rank_items

# One shared client, so the per host delay is observed across every agent
# rather than per agent. Three researchers each with their own client would
# hit the same domain three times in a row with no pause at all.
_client = HttpClient()


def _search_backends() -> list[str]:
    adapter = SearchAdapter(_client)
    return list(adapter.available_backends()) + keyless.available()


@tool
class WebSearch(Tool):
    name = "web_search"
    description = (
        "Search the web for pages about a query. Returns ranked results with a "
        "title, URL and snippet, but not the page text. Use fetch_page to read one."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for. Write it as a search query, not a sentence.",
            },
            "limit": {
                "type": "integer",
                "description": "How many results to return, 1 to 20.",
                "default": 8,
                "minimum": 1,
                "maximum": 20,
            },
            "recency_days": {
                "type": "integer",
                "description": "Only return pages published within this many days. 0 means any age.",
                "default": 0,
                "minimum": 0,
                "maximum": 3650,
            },
        },
        "required": ["query"],
    }

    def is_available(self, ctx: Any) -> tuple[bool, str]:
        if _search_backends():
            return True, ""
        return False, (
            "No search backend is configured. Set TAVILY_API_KEY, EXA_API_KEY, "
            "SERPAPI_KEY or BRAVE_API_KEY, or enable keyless search."
        )

    def call(self, ctx: Any, *, query: str, limit: int = 8, recency_days: int = 0) -> Any:
        query = query.strip()
        if not query:
            raise ToolError("A search needs a non empty query.")

        since = ""
        if recency_days:
            from ..core.util import hours_ago_iso

            since = hours_ago_iso(recency_days * 24)

        items: list[RawItem] = []
        used: list[str] = []
        errors: list[str] = []

        adapter = SearchAdapter(_client)
        for backend in adapter.available_backends():
            try:
                outcome = adapter.fetch(
                    "", config={"query": query, "backend": backend}, limit=limit * 2, since=since
                )
                if outcome.items:
                    items.extend(outcome.items)
                    used.append(backend)
            except SourceError as exc:
                errors.append(f"{backend}: {exc}")
            except Exception as exc:
                errors.append(f"{backend}: {type(exc).__name__}")

        if settings.ENABLE_KEYLESS_SEARCH:
            for name, fn in (("googlenews", keyless.google_news), ("wikipedia", keyless.wikipedia)):
                try:
                    found = fn(query, limit=limit if name == "googlenews" else 3, client=_client)
                    if found:
                        items.extend(found)
                        used.append(name)
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}")

        if not items:
            detail = "; ".join(errors[:3])
            raise ToolError(
                f"No results for '{query}'.",
                hint=(f"Backends reported: {detail}. " if detail else "")
                + "Try different wording, fewer words, or a broader query.",
            )

        merged = dedupe_items(items)
        ranked = rank_items(merged, query=query, limit=limit)

        ctx.board.append(
            "searches", {"query": query, "backends": used, "results": len(ranked)}
        )
        return {
            "query": query,
            "backends_used": sorted(set(used)),
            "count": len(ranked),
            "results": [
                {
                    "n": i + 1,
                    "title": item.title,
                    "url": item.url,
                    "domain": item.domain,
                    "published_at": item.published_at,
                    "snippet": truncate(item.summary, 320),
                }
                for i, item in enumerate(ranked)
            ],
        }


@tool
class FetchPage(Tool):
    name = "fetch_page"
    description = (
        "Read one web page and store it as numbered evidence. Returns the source "
        "reference (like S3) and a summary of what the page says. Cite that "
        "reference later. Fetching the same URL twice is free and returns the "
        "same reference."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The full http or https URL to read."},
            "why": {
                "type": "string",
                "description": "What you expect to learn from this page. One short line.",
                "default": "",
            },
        },
        "required": ["url"],
    }

    def call(self, ctx: Any, *, url: str, why: str = "") -> Any:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise ToolError(f"'{url}' is not an http or https URL.")

        # Paying for the same page twice is the most common waste in a run with
        # several researchers, and the cheapest to prevent.
        for row in ctx.evidence():
            if row["url"] == url or row["url_key"] == _key(url):
                return {
                    "ref": row["ref"],
                    "title": row["title"],
                    "url": row["url"],
                    "already_had_it": True,
                    "summary": truncate(row["body"] or row["snippet"], 1500),
                }

        try:
            response = _client.get(url, accept="text/html")
        except SourceError as exc:
            raise ToolError(f"Could not fetch {host_of(url)}: {exc}") from exc

        if not response.ok:
            raise ToolError(
                f"{host_of(url)} returned HTTP {response.status}.",
                hint="Try a different source for the same fact.",
            )

        article = extract_article(response.text, url)
        # The extractor names the body "content"; "text" is what this tool
        # reports back to the agent, which is a different thing and truncated.
        body = str(article.get("content") or "").strip()
        title = str(article.get("title") or "").strip()

        if not body:
            raise ToolError(
                f"No readable article text could be extracted from {host_of(url)}.",
                hint="The page may be a paywall, a video or a listing. Try another source.",
            )

        # A lesson worth carrying into the next run, not just this one.
        if len(body) < 400:
            ctx.remember(
                kind="source_quality",
                key=f"thin:{host_of(url)}",
                content=f"{host_of(url)} returns very little extractable text.",
            )

        ref, is_new = ctx.add_evidence(
            {
                "url": url,
                "domain": host_of(url),
                "title": title,
                "author": str(article.get("author") or ""),
                "snippet": truncate(body, 500),
                "body": body,
                "published_at": str(article.get("published_at") or "") or None,
                "backend": "fetch",
                "found_by": getattr(ctx, "_current_agent", ""),
                "query": why,
                "simhash": str(simhash(body)),
            }
        )

        return {
            "ref": ref,
            "title": title,
            "url": url,
            "domain": host_of(url),
            "published_at": article.get("published_at") or "",
            "words": len(body.split()),
            "extraction": article.get("method", ""),
            "already_had_it": not is_new,
            # Enough for the agent to judge relevance and quote accurately,
            # without putting the whole page in its context every turn.
            "text": truncate(body, 6000),
        }


@tool
class ListEvidence(Tool):
    name = "list_evidence"
    description = (
        "List the sources gathered so far in this run, with their references. "
        "Use it to check what is already known before searching again."
    )
    parameters = {"type": "object", "properties": {}}
    metered = False

    def call(self, ctx: Any) -> Any:
        rows = ctx.evidence()
        return {
            "count": len(rows),
            "sources": [
                {
                    "ref": row["ref"],
                    "title": row["title"],
                    "domain": row["domain"],
                    "published_at": row["published_at"] or "",
                    "snippet": truncate(row["snippet"], 200),
                }
                for row in rows
            ],
        }


@tool
class ReadEvidence(Tool):
    name = "read_evidence"
    description = (
        "Read the stored text of sources already gathered, by reference "
        "(for example S1, S4). Use this to verify a claim against what a source "
        "actually said rather than fetching the page again."
    )
    parameters = {
        "type": "object",
        "properties": {
            "refs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Source references to read, like [\"S1\", \"S3\"].",
            },
            "chars": {
                "type": "integer",
                "description": "How much of each source to return.",
                "default": 4000,
                "minimum": 200,
                "maximum": 12000,
            },
        },
        "required": ["refs"],
    }
    metered = False

    def call(self, ctx: Any, *, refs: list[str], chars: int = 4000) -> Any:
        wanted = tuple(str(r).strip().upper() for r in refs if str(r).strip())
        if not wanted:
            raise ToolError("Name at least one source reference, like S1.")
        rows = ctx.evidence(refs=wanted)
        if not rows:
            known = [row["ref"] for row in ctx.evidence()]
            raise ToolError(
                f"No source matches {', '.join(wanted)}.",
                hint=f"References in this run: {', '.join(known) or 'none yet'}.",
            )
        return {
            "sources": [
                {
                    "ref": row["ref"],
                    "title": row["title"],
                    "url": row["url"],
                    "domain": row["domain"],
                    "published_at": row["published_at"] or "",
                    "text": truncate(row["body"] or row["snippet"], chars),
                }
                for row in rows
            ]
        }


def _key(url: str) -> str:
    from ..core.util import url_key

    return url_key(url)


def search_status() -> dict[str, Any]:
    """What the UI shows about research capability, without running a search."""
    adapter = SearchAdapter(_client)
    keyed = adapter.available_backends()
    free = keyless.available()
    return {
        "backends": keyed + free,
        "keyed": keyed,
        "keyless": free,
        "usable": bool(keyed or free),
    }
