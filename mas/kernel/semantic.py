"""Vector work over what a run has gathered.

Two jobs that both need the same thing, and both were previously left to the
model to do by reading.

Retrieval: once a page has been fetched, an agent wanting one fact from it
should not have to read twelve thousand words back into its context, and
certainly should not fetch the page again. Storing a vector per source makes
"which of the things we already have is about this" a local computation.

Adjudication: deciding whether two claims contradict each other starts with
knowing which pairs are even about the same thing. Comparing all pairs with a
model is quadratic and expensive; comparing all pairs with cosine is quadratic
and free, so the vectors find the candidates and the model only rules on the
handful that are plausibly in conflict.

Everything here degrades rather than fails. With no embedding backend the
similarity falls back to token overlap, which is worse and still useful, and
the callers do not branch on which one ran.
"""
from __future__ import annotations

import re
from typing import Any

from ..core.embeddings import build_text, from_blob, to_blob
from ..core.embeddings import registry as embeddings
from ..core.util import jaccard, tokens

# How much of a page is embedded. The local model truncates anyway, and the
# opening of an article carries its subject far better than the middle does.
EMBED_CHARS = 4000

# Paragraphs shorter than this are navigation, bylines and cookie notices.
MIN_PARAGRAPH_CHARS = 120


def _vector_for(texts: list[str]) -> Any:
    embedder = embeddings.resolve()
    return embedder.embed(texts, operation="semantic").vectors


def rank_evidence(
    rows: list[dict[str, Any]], query: str, *, limit: int = 4
) -> list[tuple[dict[str, Any], float]]:
    """Order stored sources by how well each matches the query."""
    if not rows:
        return []
    try:
        import numpy as np

        query_vector = _vector_for([query])[0]
        scored: list[tuple[dict[str, Any], float]] = []
        for row in rows:
            vector = from_blob(row.get("embedding"))
            if not vector:
                continue
            scored.append((row, float(np.dot(np.asarray(vector), np.asarray(query_vector)))))
        if scored:
            scored.sort(key=lambda pair: pair[1], reverse=True)
            return scored[:limit]
    except Exception:
        pass

    # No vectors stored, or no backend. Token overlap still orders them.
    wanted = set(tokens(query))
    fallback = [
        (row, jaccard(wanted, set(tokens(f"{row.get('title', '')} {row.get('snippet', '')}"))))
        for row in rows
    ]
    fallback.sort(key=lambda pair: pair[1], reverse=True)
    return fallback[:limit]


# Markdown link syntax, which the extraction proxy leaves in. A navigation bar
# rendered as markdown is long enough to pass a length filter while containing
# almost no prose, and it was being returned as the best passage on a page.
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Boilerplate that is written in full sentences and therefore passes every
# other test here: privacy notices, legal disclaimers, search help, consent
# banners. A proxy that returns a whole page rather than its article hands back
# several hundred words of this, and storing it as a source means an agent can
# cite a cookie notice.
_CHROME = re.compile(
    r"skip to (main )?content|cookie|privacy (policy|notice)|subscribe|newsletter|"
    r"all rights reserved|follow us|share this|advertisement|"
    r"personal information|opt[- ]out|do not sell|your california privacy|"
    r"not an endorsement|provided for convenience|try a new search|"
    r"reviewed and approved by|terms of (use|service)|sign up for|"
    r"search field|begin navigating|javascript",
    re.IGNORECASE,
)
# Prose, once the link targets are gone. A paragraph below this is a menu.
MIN_PARAGRAPH_WORDS = 20
# A real sentence, for the purpose of telling prose from a list of topics.
MIN_SENTENCE_WORDS = 10

_LIST_MARKER = re.compile(r"(?m)^[\s>]*[*\-+•]\s+|^[\s>]*\d+[.)]\s+")
# A run of text that actually ends in a full stop, question mark or exclamation
# mark. Splitting on the terminator would not do: a block with no punctuation
# at all comes back from a split as one enormous "sentence" and passes any word
# count, which is exactly what a navigation menu is.
_SENTENCE = re.compile(r"[^.!?\n][^.!?]{0,600}[.!?]")


def _prose_of(paragraph: str) -> str:
    """The paragraph with markdown link targets and list markers removed."""
    return _LIST_MARKER.sub("", _MD_LINK.sub(r"\1", paragraph)).strip()


def _has_a_sentence(prose: str) -> bool:
    """Whether this reads as prose rather than as a list of topics.

    Site navigation survives every filter based on size: strip the links out of
    "Health Topics, Aortic Aneurysm, Arrhythmia, Cardiac Arrest" and there are
    still plenty of characters and words. What it does not have, anywhere, is a
    terminated sentence. Every navigation block on a real page measured here
    had exactly zero, and every article paragraph had at least one, which makes
    this a far sharper test than any threshold on length.
    """
    return any(
        len(match.group(0).split()) >= MIN_SENTENCE_WORDS
        for match in _SENTENCE.finditer(prose)
    )


def split_paragraphs(body: str) -> list[str]:
    """The paragraphs of a page that are actually prose.

    Length alone is not enough to tell an article paragraph from a navigation
    block: a menu rendered as markdown links is hundreds of characters of URL
    and a dozen words. Measuring the text left after the link targets are
    stripped separates the two.
    """
    parts = [p.strip() for p in re.split(r"\n\s*\n", body or "") if p.strip()]
    kept = []
    for part in parts:
        prose = _prose_of(part)
        if len(prose) < MIN_PARAGRAPH_CHARS or len(prose.split()) < MIN_PARAGRAPH_WORDS:
            continue
        if _CHROME.search(prose):
            continue
        if not _has_a_sentence(prose):
            continue
        kept.append(prose)
    return kept


# A page that yields less prose than this did not really yield an article.
MIN_ARTICLE_CHARS = 700


def article_text(body: str) -> str:
    """The prose of a page, with the navigation and the legal notices removed.

    Used on text that came from a proxy, which returns the whole page rather
    than its article. A direct fetch has already been through an extractor that
    does this job, so it is left alone.
    """
    return "\n\n".join(split_paragraphs(body))


def embed_document(title: str, body: str) -> bytes | None:
    """A stored vector for one source, or None when no backend is available.

    Embedded from the page's prose rather than its opening characters. Many
    pages begin with a navigation block, and a vector built from that describes
    the site's menu rather than the article, which makes the page unfindable by
    its own subject.
    """
    try:
        embedder = embeddings.resolve()
        prose = "\n\n".join(split_paragraphs(body)) or (body or "")
        text = build_text(title, "", prose[:EMBED_CHARS])
        vectors = embedder.embed([text], operation="evidence").vectors
        return to_blob(vectors[0])
    except Exception:
        return None


def best_passages(
    body: str, query: str, *, limit: int = 3, chars: int = 900
) -> list[str]:
    """The paragraphs of one source that actually bear on the query.

    Embedded at query time rather than stored, because a passage index would be
    many times the size of the document index and the local model is fast
    enough that the saving is not worth the storage.
    """
    paragraphs = split_paragraphs(body)
    if not paragraphs:
        return [(body or "")[:chars]] if body else []
    if len(paragraphs) <= limit:
        return [p[:chars] for p in paragraphs]

    try:
        import numpy as np

        vectors = _vector_for([query, *paragraphs])
        query_vector, paragraph_vectors = np.asarray(vectors[0]), np.asarray(vectors[1:])
        scores = paragraph_vectors @ query_vector
        order = list(np.argsort(scores)[::-1][:limit])
        # Returned in document order, so a reader is not handed the middle of
        # an argument before its opening.
        return [paragraphs[i][:chars] for i in sorted(int(i) for i in order)]
    except Exception:
        wanted = set(tokens(query))
        scored = [(jaccard(wanted, set(tokens(p))), i) for i, p in enumerate(paragraphs)]
        scored.sort(reverse=True)
        keep = sorted(index for _, index in scored[:limit])
        return [paragraphs[i][:chars] for i in keep]


def candidate_pairs(
    claims: list[dict[str, Any]], *, threshold: float = 0.62, cap: int = 8
) -> list[tuple[dict[str, Any], dict[str, Any], float]]:
    """Claim pairs similar enough to be worth asking whether they conflict.

    Similarity alone cannot tell agreement from contradiction: "the rate rose"
    and "the rate fell" sit close together in every embedding space, which is
    exactly why this returns candidates rather than verdicts. What it buys is
    that a model is asked about a handful of pairs instead of all of them.

    Pairs that already share a source are skipped. Two claims drawn from the
    same page contradicting each other is a reading error rather than a
    disagreement between sources, and it is not what the report should be
    reporting.
    """
    if len(claims) < 2:
        return []

    texts = [str(c.get("text", "")) for c in claims]
    try:
        import numpy as np

        vectors = np.asarray(_vector_for(texts))
        similarity = vectors @ vectors.T
        score_of = lambda i, j: float(similarity[i][j])  # noqa: E731
    except Exception:
        token_sets = [set(tokens(t)) for t in texts]
        score_of = lambda i, j: jaccard(token_sets[i], token_sets[j])  # noqa: E731

    found: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            score = score_of(i, j)
            if score < threshold:
                continue
            shared = set(claims[i].get("sources", [])) & set(claims[j].get("sources", []))
            if shared:
                continue
            found.append((claims[i], claims[j], round(score, 4)))

    found.sort(key=lambda triple: triple[2], reverse=True)
    return found[:cap]
