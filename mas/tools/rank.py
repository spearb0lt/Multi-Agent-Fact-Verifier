"""Merging several search backends into one honest ordering.

Every backend returns results in its own order, and those orders are not
comparable with each other. Tavily's third result and Exa's third result are
both "third" and mean nothing in common, so concatenating the lists produces a
ranking that is really just an accident of which backend was configured first.
Re-scoring all of them against the query is the only defensible way to merge.

Three signals, in the order they matter:

* relevance to the query, measured in the embedding space, which is the whole
  reason the bundled ONNX model is carried in this project,
* freshness, as a nudge rather than a driver, because a week old page that
  answers the question beats an hour old page that mentions it,
* domain diversity, applied last, because eight results from one outlet is one
  source wearing eight hats and corroboration counted across them is a lie.

The embedding path degrades to a lexical one rather than failing. A deployment
with no embedding backend at all still gets a defensible ordering.
"""
from __future__ import annotations

from ..core.embeddings import build_text
from ..core.embeddings import registry as embeddings
from ..core.util import host_of, jaccard, parse_datetime, tokens, utcnow
from ..core.web.base import RawItem

# Hours past which a page contributes no freshness bonus at all. Two weeks: a
# research brief regularly wants background that is years old, so this only
# ever breaks ties among recent things.
FRESHNESS_HORIZON_HOURS = 336.0
FRESHNESS_WEIGHT = 0.15
# How far a result is pushed down for each earlier result from the same domain.
DIVERSITY_PENALTY = 0.12


def _text_of(item: RawItem) -> str:
    # The longer of summary and content, not the first non empty one. A search
    # backend's summary is a short snippet, and preferring it would discard a
    # full body that a previous fetch had already paid for.
    return build_text(item.title, max((item.content or ""), (item.summary or ""), key=len))


def _display_domain(item: RawItem) -> str:
    # Google News hands back a redirector, so every result would otherwise read
    # as news.google.com. The adapter records the real outlet in meta.
    return (item.meta or {}).get("publisher") or item.domain or host_of(item.url)


def _freshness(item: RawItem) -> float:
    published = parse_datetime(item.published_at) if item.published_at else None
    if published is None:
        # Unknown age scores as middling rather than as new or as ancient.
        return 0.5
    hours = max(0.0, (utcnow() - published).total_seconds() / 3600.0)
    return max(0.0, 1.0 - min(hours, FRESHNESS_HORIZON_HOURS) / FRESHNESS_HORIZON_HOURS)


def _lexical_relevance(query: str, items: list[RawItem]) -> list[float]:
    wanted = set(tokens(query))
    if not wanted:
        return [0.0] * len(items)
    return [jaccard(wanted, set(tokens(_text_of(item)))) for item in items]


def _semantic_relevance(query: str, items: list[RawItem]) -> tuple[list[float], object]:
    embedder = embeddings.resolve()
    vectors = embedder.embed([query, *[_text_of(i) for i in items]], operation="rank").vectors
    query_vector, doc_vectors = vectors[0], vectors[1:]
    return list(doc_vectors @ query_vector), doc_vectors


def rank_items(
    items: list[RawItem],
    *,
    query: str,
    limit: int = 10,
    diversify: bool = True,
) -> list[RawItem]:
    """Order merged results by relevance, then freshness, then outlet spread."""
    if not items:
        return []
    if len(items) == 1:
        return items[:limit]

    try:
        relevance, doc_vectors = _semantic_relevance(query, items)
        duplicate_cut = embeddings.thresholds(embeddings.resolve())[1]
    except Exception:
        relevance = _lexical_relevance(query, items)
        doc_vectors = None
        duplicate_cut = 1.1  # Unreachable, so nothing is dropped as a near duplicate.

    scored = [
        (float(relevance[i]) + FRESHNESS_WEIGHT * _freshness(item), i)
        for i, item in enumerate(items)
    ]
    scored.sort(reverse=True)

    import numpy as np

    kept: list[RawItem] = []
    kept_vectors: list = []
    seen_domains: dict[str, int] = {}
    pending: list[tuple[float, int]] = []

    for score, index in scored:
        item = items[index]
        if doc_vectors is not None and kept_vectors:
            vector = doc_vectors[index]
            if float(np.max(np.asarray(kept_vectors) @ vector)) >= duplicate_cut:
                # The same story reprinted. Corroboration counted across two
                # copies of one wire report is not corroboration.
                continue
        domain = _display_domain(item)
        seen = seen_domains.get(domain, 0)
        if diversify and seen:
            pending.append((score - DIVERSITY_PENALTY * seen, index))
            continue
        seen_domains[domain] = seen + 1
        if doc_vectors is not None:
            kept_vectors.append(doc_vectors[index])
        kept.append(item)
        if len(kept) >= limit:
            return kept

    # Repeated outlets fill whatever room is left, best scoring first, so a
    # query that genuinely only one outlet covers still returns results.
    pending.sort(reverse=True)
    for _, index in pending:
        if len(kept) >= limit:
            break
        kept.append(items[index])
    return kept
