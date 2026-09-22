"""Getting the article out of a page of HTML.

A feed hands over a headline and a paragraph. Clustering, deduplication and
summarising all work markedly better on the body text, so a fetched page passes
through here before the pipeline sees it.

trafilatura does this job well and is tried first, but it is optional at
runtime: a slim deployment may not carry it, and on some pages it returns
nothing at all. The dependency free extractor below is therefore not a nicety,
it is the path a real share of pages take.

Metadata is read from JSON-LD and the OpenGraph tags whichever extractor
produced the body, because trafilatura misses the lead image and the canonical
URL on many sites, and a missing publication date silently moves a story out of
the lookback window.

Nothing here raises. An extraction failure must cost one article's body text,
never the whole run.
"""
from __future__ import annotations

import json
import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin

from ..util import parse_datetime_iso, strip_html

# Elements whose text is never part of the article. Removed before scoring so a
# navigation bar packed with link text cannot win against a real body.
_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "form",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
    "figcaption",
)

# Void elements never take a closing tag, so the parser must not push them on
# the open element stack or every later close would land on the wrong node.
_VOID_TAGS = frozenset(
    {
        "area", "base", "basefont", "br", "col", "embed", "frame", "hr", "img",
        "input", "isindex", "link", "meta", "param", "source", "track", "wbr",
    }
)

# Tags that end a line of prose, used when flattening a subtree to text.
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "main", "br", "li", "ul", "ol", "tr",
        "table", "blockquote", "pre", "h1", "h2", "h3", "h4", "h5", "h6", "dd",
        "dt", "figure", "hr",
    }
)

# Containers worth considering as the article body.
_CANDIDATE_TAGS = frozenset({"article", "main", "div", "section", "td"})

_POSITIVE_HINT = re.compile(
    r"(^|[\s_-])(article|articlebody|story|storybody|post|postbody|entry|content|"
    r"maincontent|main-content|body|text|prose|blog|news)([\s_-]|$)",
    re.I,
)
_NEGATIVE_HINT = re.compile(
    r"(comment|disqus|sidebar|footer|header|menu|nav|breadcrumb|share|social|"
    r"related|recommend|promo|newsletter|subscribe|advert|advertisement|ad-|"
    r"-ad|banner|widget|popup|modal|cookie|paywall|masthead|tag-list|meta)",
    re.I,
)

_ATTR_RE = re.compile(
    r"""([\w:.\-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+))""",
    re.I,
)
_META_RE = re.compile(r"<meta\s[^>]*?/?>", re.I | re.S)
_LINK_RE = re.compile(r"<link\s[^>]*?/?>", re.I | re.S)
_JSONLD_RE = re.compile(
    r"<script[^>]+type\s*=\s*[\"']?application/ld\+json[\"']?[^>]*>(.*?)</script>",
    re.I | re.S,
)
_HTML_TAG_RE = re.compile(r"<html\s[^>]*>", re.I)
_TITLE_TAG_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
_TIME_RE = re.compile(r"<time\s[^>]*>", re.I)
_ANCHOR_RE = re.compile(r"<a\s([^>]*)>(.*?)</a>", re.I | re.S)

# Article types JSON-LD uses. A site that marks its story as a plain WebPage is
# common enough that the looser types are accepted after the precise ones.
_PREFERRED_LD_TYPES = ("newsarticle", "reportagenewsarticle", "article", "blogposting")
_FALLBACK_LD_TYPES = ("webpage", "creativework", "liveblogposting", "videoobject")

_EMPTY_RESULT: dict[str, Any] = {
    "content": "",
    "title": "",
    "author": "",
    "published_at": "",
    "image_url": "",
    "language": "",
    "canonical_url": "",
    "method": "failed",
}


# ------------------------------------------------------------------ public API


def extract_article(html: str, url: str) -> dict:
    """Pull the readable article and its metadata out of one page.

    Returns the same keys whatever happened, so a caller never has to branch on
    which extractor ran. `method` says which one did, which is worth keeping
    because a page extracted by the fallback is the first suspect when a
    summary reads badly.
    """
    result = dict(_EMPTY_RESULT)
    if not html or not html.strip():
        return result

    try:
        meta = _read_metadata(html, url)
    except Exception:  # noqa: BLE001 - metadata is a bonus, never a blocker
        meta = {}

    content = ""
    method = "failed"
    try:
        content, extra = _trafilatura(html, url)
        if content:
            method = "trafilatura"
            # trafilatura's own metadata only fills gaps: the page's structured
            # data is the more reliable source where both exist.
            for key, value in extra.items():
                if value and not meta.get(key):
                    meta[key] = value
    except Exception:  # noqa: BLE001
        content = ""

    if not content:
        try:
            content = _fallback_content(html)
            if content:
                method = "heuristic"
        except Exception:  # noqa: BLE001
            content = ""

    result.update(
        {
            "content": content,
            "title": meta.get("title", ""),
            "author": meta.get("author", ""),
            "published_at": meta.get("published_at", ""),
            "image_url": meta.get("image_url", ""),
            "language": meta.get("language", ""),
            "canonical_url": meta.get("canonical_url", ""),
            "method": method if content else ("metadata" if meta.get("title") else "failed"),
        }
    )
    return result


def iter_anchors(html: str, base_url: str = "") -> list[tuple[str, str]]:
    """Every link on the page as (absolute href, visible text).

    Shared by the HTML crawler and the browser adapter so that a rendered page
    and a plain fetch are treated identically.
    """
    if not html:
        return []
    out: list[tuple[str, str]] = []
    body = _strip_elements(html, ("script", "style", "noscript", "template", "svg"))
    for match in _ANCHOR_RE.finditer(body):
        attrs = _parse_attrs(match.group(1))
        href = (attrs.get("href") or "").strip()
        if not href:
            continue
        text = strip_html(match.group(2) or "").strip()
        if base_url:
            try:
                href = urljoin(base_url, href)
            except ValueError:
                continue
        out.append((href, text))
    return out


# ------------------------------------------------------------------ extractors


def _trafilatura(html: str, url: str) -> tuple[str, dict[str, str]]:
    """Body text and metadata from trafilatura, or empty when it cannot help.

    The import is local and guarded because trafilatura is optional: the module
    must still import where it was left out of the bundle.
    """
    try:
        import trafilatura  # noqa: PLC0415 - optional at runtime by design
    except Exception:  # noqa: BLE001
        return "", {}

    text = ""
    try:
        text = (
            trafilatura.extract(
                html,
                url=url or None,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
                no_fallback=False,
            )
            or ""
        ).strip()
    except Exception:  # noqa: BLE001 - a parser crash is not a fetch failure
        text = ""
    if not text:
        return "", {}

    extra: dict[str, str] = {}
    try:
        meta = trafilatura.extract_metadata(html, default_url=url or None)
    except Exception:  # noqa: BLE001
        meta = None
    if meta is not None:
        as_dict = meta.as_dict() if hasattr(meta, "as_dict") else {}
        extra = {
            "title": strip_html(str(as_dict.get("title") or "")),
            "author": strip_html(str(as_dict.get("author") or "")),
            "published_at": parse_datetime_iso(as_dict.get("date") or ""),
            "image_url": str(as_dict.get("image") or ""),
            "language": str(as_dict.get("language") or ""),
            "canonical_url": str(as_dict.get("url") or ""),
        }
    return text, {k: v for k, v in extra.items() if v}


def _fallback_content(html: str) -> str:
    """Score the page's containers and return the text of the best one."""
    cleaned = _strip_elements(html, _DROP_TAGS)
    root = _build_dom(cleaned)
    if root is None:
        return ""

    best: _Node | None = None
    best_score = 0.0
    for node in _walk(root):
        if node.tag not in _CANDIDATE_TAGS:
            continue
        score = _score_node(node)
        if score > best_score:
            best_score = score
            best = node

    if best is None or best_score <= 0:
        # Nothing looked like a body, so the whole document is the last resort.
        text = _node_text(root)
        return text if len(text) >= 200 else ""
    return _node_text(best)


# --------------------------------------------------------------- tiny HTML DOM


class _Node:
    __slots__ = ("tag", "attrs", "children", "parent", "chunks")

    def __init__(self, tag: str, attrs: dict[str, str], parent: _Node | None) -> None:
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children: list[_Node] = []
        self.chunks: list[str] = []


class _DomBuilder(HTMLParser):
    """A forgiving tree builder.

    Real pages close tags they never opened and leave tags open forever, so a
    mismatched close walks up to the nearest matching ancestor and anything
    unmatched is ignored rather than corrupting the tree.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("[document]", {}, None)
        self.current = self.root

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        mapped = {k.lower(): (v or "") for k, v in attrs}
        node = _Node(tag.lower(), mapped, self.current)
        self.current.children.append(node)
        if tag.lower() not in _VOID_TAGS:
            self.current = node

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        mapped = {k.lower(): (v or "") for k, v in attrs}
        self.current.children.append(_Node(tag.lower(), mapped, self.current))

    def handle_endtag(self, tag: str) -> None:
        name = tag.lower()
        if name in _VOID_TAGS:
            return
        node: _Node | None = self.current
        while node is not None and node is not self.root:
            if node.tag == name:
                self.current = node.parent or self.root
                return
            node = node.parent

    def handle_data(self, data: str) -> None:
        if data and data.strip():
            self.current.chunks.append(data)


def _build_dom(html: str) -> _Node | None:
    builder = _DomBuilder()
    try:
        builder.feed(html)
        builder.close()
    except Exception:  # noqa: BLE001 - a half parsed tree is still usable
        pass
    return builder.root if builder.root.children else None


def _walk(node: _Node) -> list[_Node]:
    out: list[_Node] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(current.children)
    return out


def _node_text(node: _Node) -> str:
    """Flatten a subtree, keeping paragraph breaks so the text stays readable."""
    parts: list[str] = []
    _collect_text(node, parts)
    text = "".join(parts)
    # The non breaking space a decoded HTML entity leaves behind is not
    # covered by a plain space class, so it is folded here too.
    text = re.sub("[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line is not None).strip()


def _collect_text(node: _Node, parts: list[str]) -> None:
    if node.tag in _BLOCK_TAGS:
        parts.append("\n")
    for chunk in node.chunks:
        parts.append(chunk)
    for child in node.children:
        _collect_text(child, parts)
        if child.chunks or child.children:
            parts.append(" ")
    if node.tag in _BLOCK_TAGS:
        parts.append("\n")


def _score_node(node: _Node) -> float:
    """How much this container looks like the article body.

    Length alone picks the page wrapper every time, so link density and the
    class and id names carry most of the decision: a body is long, mostly
    unlinked prose sitting in an element somebody named after an article.
    """
    text = _node_text(node)
    length = len(text)
    if length < 180:
        return 0.0

    paragraphs = [child for child in _walk(node) if child.tag == "p"]
    paragraph_text = sum(len(_node_text(p)) for p in paragraphs)
    link_text = sum(len(_node_text(a)) for a in _walk(node) if a.tag == "a")
    link_density = min(1.0, link_text / float(length))

    score = paragraph_text + length * 0.25
    score += len([p for p in paragraphs if len(_node_text(p)) > 80]) * 40
    score += text.count(",") * 4
    # A wall of links is a navigation block however long it is.
    score *= max(0.0, 1.0 - link_density * 1.6)

    signature = " ".join(
        (node.attrs.get("class", ""), node.attrs.get("id", ""), node.attrs.get("itemprop", ""))
    )
    if _POSITIVE_HINT.search(signature):
        score *= 1.6
    if _NEGATIVE_HINT.search(signature):
        score *= 0.25
    if node.tag == "article":
        score *= 1.5
    elif node.tag == "main":
        score *= 1.25
    if node.attrs.get("itemprop", "").lower() in {"articlebody", "text"}:
        score *= 1.5
    return score


# ------------------------------------------------------------------- metadata


def _read_metadata(html: str, url: str) -> dict[str, str]:
    """Title, byline, date, image, language and canonical URL for one page.

    Order matters: JSON-LD is the publisher's own structured record and wins,
    OpenGraph is next because it is what the publisher shows to social
    networks, and the plain document tags are the last resort.
    """
    metas = _collect_meta(html)
    links = _collect_link_rels(html)
    ld = _collect_json_ld(html)

    title = (
        _first(ld.get("headline"), ld.get("name"))
        or metas.get("og:title")
        or metas.get("twitter:title")
        or metas.get("dcterms.title")
        or _document_title(html)
    )
    author = (
        ld.get("author")
        or metas.get("author")
        or metas.get("article:author")
        or metas.get("byl")
        or metas.get("dc.creator")
        or metas.get("dcterms.creator")
        or metas.get("parsely-author")
        or metas.get("twitter:creator")
        or ""
    )
    published = (
        ld.get("datePublished")
        or metas.get("article:published_time")
        or metas.get("og:article:published_time")
        or metas.get("article:modified_time")
        or metas.get("date")
        or metas.get("pubdate")
        or metas.get("publish-date")
        or metas.get("dc.date")
        or metas.get("dcterms.date")
        or metas.get("parsely-pub-date")
        or metas.get("sailthru.date")
        or _time_element_date(html)
        or ""
    )
    image = (
        metas.get("og:image")
        or metas.get("og:image:url")
        or metas.get("twitter:image")
        or metas.get("twitter:image:src")
        or ld.get("image")
        or metas.get("parsely-image-url")
        or ""
    )
    language = (
        _html_lang(html)
        or metas.get("og:locale")
        or metas.get("content-language")
        or metas.get("dc.language")
        or ld.get("inLanguage")
        or ""
    )
    canonical = links.get("canonical") or metas.get("og:url") or ld.get("url") or ""

    return {
        "title": strip_html(unescape(title or "")).strip(),
        "author": strip_html(unescape(author or "")).strip()[:200],
        "published_at": parse_datetime_iso(published) if published else "",
        "image_url": _absolutise(image, url),
        "language": _language_code(language),
        "canonical_url": _absolutise(canonical, url),
    }


def _collect_meta(html: str) -> dict[str, str]:
    """Every meta tag keyed by whichever naming attribute it used."""
    out: dict[str, str] = {}
    for match in _META_RE.finditer(html):
        attrs = _parse_attrs(match.group(0))
        content = attrs.get("content", "").strip()
        if not content:
            continue
        for key_attr in ("property", "name", "itemprop", "http-equiv"):
            key = attrs.get(key_attr, "").strip().lower()
            if key and key not in out:
                out[key] = content
    return out


def _collect_link_rels(html: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _LINK_RE.finditer(html):
        attrs = _parse_attrs(match.group(0))
        rel = attrs.get("rel", "").strip().lower()
        href = attrs.get("href", "").strip()
        if rel and href and rel not in out:
            out[rel] = href
    return out


def _collect_json_ld(html: str) -> dict[str, str]:
    """The NewsArticle block, flattened to plain strings.

    Publishers nest these arbitrarily: a list of blocks, an @graph, an author
    that is a string or an object or a list of objects. Every shape seen in the
    wild is reduced here rather than at each call site.
    """
    candidates: list[dict[str, Any]] = []
    for match in _JSONLD_RE.finditer(html):
        raw = match.group(1).strip()
        if not raw:
            continue
        # Some CMSs emit HTML comment wrappers or a trailing comma inside the
        # script block, which json refuses outright.
        raw = re.sub(r"^<!--|-->$", "", raw).strip()
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        candidates.extend(_flatten_ld(parsed))

    chosen: dict[str, Any] | None = None
    for types in (_PREFERRED_LD_TYPES, _FALLBACK_LD_TYPES):
        for block in candidates:
            if _ld_type_matches(block, types):
                chosen = block
                break
        if chosen is not None:
            break
    if chosen is None:
        return {}

    out: dict[str, str] = {}
    for key in ("headline", "name", "datePublished", "dateCreated", "url", "inLanguage"):
        value = _ld_scalar(chosen.get(key))
        if value:
            out.setdefault(key, value)
    if "datePublished" not in out and chosen.get("dateCreated"):
        out["datePublished"] = _ld_scalar(chosen.get("dateCreated"))
    author = _ld_person(chosen.get("author")) or _ld_person(chosen.get("creator"))
    if author:
        out["author"] = author
    image = _ld_image(chosen.get("image")) or _ld_image(chosen.get("thumbnailUrl"))
    if image:
        out["image"] = image
    return out


def _flatten_ld(value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(value, list):
        for entry in value:
            out.extend(_flatten_ld(entry))
    elif isinstance(value, dict):
        out.append(value)
        graph = value.get("@graph")
        if graph:
            out.extend(_flatten_ld(graph))
    return out


def _ld_type_matches(block: dict[str, Any], types: tuple[str, ...]) -> bool:
    raw = block.get("@type") or block.get("type") or ""
    names = raw if isinstance(raw, list) else [raw]
    return any(str(name).strip().lower() in types for name in names)


def _ld_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list) and value:
        return _ld_scalar(value[0])
    if isinstance(value, dict):
        for key in ("name", "@value", "url", "@id"):
            if value.get(key):
                return _ld_scalar(value[key])
    return ""


def _ld_person(value: Any) -> str:
    if isinstance(value, list):
        names = [_ld_person(entry) for entry in value]
        return ", ".join(name for name in names if name)[:200]
    if isinstance(value, dict):
        return _ld_scalar(value.get("name") or value.get("@id") or "")
    if isinstance(value, str):
        return value.strip()
    return ""


def _ld_image(value: Any) -> str:
    if isinstance(value, list) and value:
        return _ld_image(value[0])
    if isinstance(value, dict):
        return _ld_scalar(value.get("url") or value.get("contentUrl") or "")
    if isinstance(value, str):
        return value.strip()
    return ""


def _document_title(html: str) -> str:
    match = _TITLE_TAG_RE.search(html)
    if match:
        title = strip_html(unescape(match.group(1))).strip()
        if title:
            return title
    match = _H1_RE.search(html)
    return strip_html(unescape(match.group(1))).strip() if match else ""


def _time_element_date(html: str) -> str:
    """A <time datetime=...> is common on sites that ship no meta date at all."""
    for match in _TIME_RE.finditer(html):
        attrs = _parse_attrs(match.group(0))
        value = attrs.get("datetime", "").strip()
        if value and parse_datetime_iso(value):
            return value
    return ""


def _html_lang(html: str) -> str:
    match = _HTML_TAG_RE.search(html)
    if not match:
        return ""
    return _parse_attrs(match.group(0)).get("lang", "").strip()


def _language_code(value: str) -> str:
    """Just the language, since 'en-GB' and 'en_US' are the same for our use."""
    code = (value or "").strip().replace("_", "-")
    if not code:
        return ""
    return code.split("-")[0].lower()[:8]


# -------------------------------------------------------------------- helpers


def _strip_elements(html: str, tags: tuple[str, ...]) -> str:
    out = html
    for tag in tags:
        out = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}\s*>", " ", out, flags=re.I | re.S
        )
        # An unclosed one would otherwise swallow the rest of the document.
        out = re.sub(rf"<{tag}\b[^>]*/>", " ", out, flags=re.I)
    return out


def _parse_attrs(tag_html: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in _ATTR_RE.finditer(tag_html):
        name = match.group(1).lower()
        value = match.group(2) or match.group(3) or match.group(4) or ""
        out.setdefault(name, unescape(value))
    return out


def _absolutise(value: str, base_url: str) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return ""
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    if candidate.startswith(("http://", "https://")):
        return candidate
    if not base_url:
        return ""
    try:
        return urljoin(base_url, candidate)
    except ValueError:
        return ""


def _first(*values: Any) -> str:
    for value in values:
        text = (str(value) if value is not None else "").strip()
        if text:
            return text
    return ""
