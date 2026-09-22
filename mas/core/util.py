"""Small shared helpers: time, URLs, text fingerprints.

The URL and fingerprint functions here do the first and cheapest layer of
deduplication. Getting them right removes most duplicates before a single
embedding is computed, which matters because embeddings are the part that costs
money or cold start time.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# ------------------------------------------------------------------- time


def utcnow() -> datetime:
    return datetime.now(UTC)


def now_iso() -> str:
    return to_iso(utcnow())


def to_iso(value: datetime | None) -> str:
    if value is None:
        return ""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def hours_ago_iso(hours: float) -> str:
    return to_iso(utcnow() - timedelta(hours=hours))


_ISO_CLEAN_RE = re.compile(r"(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")


def parse_datetime(value: Any) -> datetime | None:
    """Best effort parse of the many date formats feeds emit.

    Feeds are inconsistent in a way no single parser handles, so the common
    shapes are tried in order of likelihood and anything unrecognised returns
    None rather than a wrong date. A wrong date is worse than a missing one
    here, because the lookback window silently drops or admits the story.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, (tuple, list)) and len(value) >= 6:
        # feedparser's struct_time style tuple.
        try:
            return datetime(*[int(p) for p in value[:6]], tzinfo=UTC)
        except (TypeError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(text)
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError, IndexError):
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %d, %Y",
        "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def parse_datetime_iso(value: Any) -> str:
    return to_iso(parse_datetime(value))


# -------------------------------------------------------------------- URLs

# Parameters that identify a campaign or a referrer rather than a document.
# Two URLs differing only in these point at the same story, so they are dropped
# before the key is taken.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "utm_id", "utm_name", "utm_reader", "utm_brand", "utm_social",
        "utm_social-type", "utm_swu", "utm_place", "utm_pubreferrer", "utm_viz_id",
        "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "twclid", "igshid",
        "mc_cid", "mc_eid", "_hsenc", "_hsmi", "hsa_acc", "hsa_cam", "hsa_grp",
        "ref", "referrer", "referer", "source", "src", "cmpid", "CMP", "cmp",
        "ncid", "sr_share", "smid", "smtyp", "partner", "spm", "scmp_source",
        "at_medium", "at_campaign", "at_custom1", "at_custom2", "at_custom3",
        "at_custom4", "ito", "icid", "ipid", "cid", "s_kwcid", "yclid",
        "vero_conv", "vero_id", "wickedid", "oly_enc_id", "oly_anon_id",
        "__twitter_impression", "guccounter", "guce_referrer", "guce_referrer_sig",
        "amp", "outputType", "sh", "share", "sharetype", "taid", "feature",
    }
)

# Path suffixes that mark an alternate rendering of the same document.
_AMP_SUFFIXES = ("/amp", "/amp/", ".amp", "/amp.html", "?amp", "/story.amp")


def host_of(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host.split(":")[0]


def normalise_url(url: str) -> str:
    """A canonical form of a URL for comparison, not for fetching.

    Strips tracking parameters, AMP markers, default ports, the fragment and a
    trailing slash, lowercases the host, and orders the remaining query so that
    parameter order cannot make one story look like two.
    """
    if not url:
        return ""
    raw = url.strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw

    scheme = (parts.scheme or "https").lower()
    if scheme not in {"http", "https"}:
        return raw
    # http and https of the same page are the same page.
    scheme = "https"

    host = parts.netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[1]
    if host.endswith(":80") or host.endswith(":443"):
        host = host.rsplit(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m.") and len(host) > 2:
        host = host[2:]

    path = parts.path or "/"
    for suffix in _AMP_SUFFIXES:
        if path.endswith(suffix) and len(path) > len(suffix):
            path = path[: -len(suffix)] or "/"
            break
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    query = urlencode(sorted(kept))

    return urlunsplit((scheme, host, path, query, ""))


def url_key(url: str) -> str:
    """A stable short key for the canonical URL, used as the unique index."""
    return hashlib.sha1(normalise_url(url).encode("utf-8")).hexdigest()


# -------------------------------------------------------------------- text

_WORD_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_TAG_RE = re.compile(r"<[^>]+>")

# Words too common to carry signal in a headline comparison. Deliberately
# short: an aggressive list starts removing words that distinguish stories.
_STOPWORDS = frozenset(
    ["a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "of", "in", "on", "at", "to", "for", "with", "from", "by", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "he", "she", "they", "them", "we", "you", "i", "not", "no", "nor", "so", "such", "up", "out", "over", "under", "again", "further", "once", "here", "there", "all", "any", "both", "each", "few", "more", "most", "other", "some", "only", "own", "same", "too", "very", "can", "will", "just", "do", "does", "did", "doing", "have", "has", "had", "having", "would", "could", "should", "may", "might", "said", "says", "say", "new", "news", "report", "reports", "update", "updates", "after", "before", "amid"]
)


def strip_html(text: str) -> str:
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = (
        cleaned.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&rsquo;", "'")
        .replace("&ldquo;", '"')
        .replace("&rdquo;", '"')
    )
    return _WS_RE.sub(" ", cleaned).strip()


def normalise_text(text: str) -> str:
    """Lowercased, accent folded, whitespace collapsed."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return _WS_RE.sub(" ", folded.lower()).strip()


def tokens(text: str, *, drop_stopwords: bool = True) -> list[str]:
    words = _WORD_RE.findall(normalise_text(text))
    if drop_stopwords:
        return [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return words


def content_hash(text: str) -> str:
    """Hash of the normalised text, which catches byte identical republishing."""
    return hashlib.sha256(normalise_text(strip_html(text)).encode("utf-8")).hexdigest()


def simhash(text: str, *, bits: int = 64) -> int:
    """A 64 bit locality sensitive fingerprint of the text.

    Two documents that share most of their wording produce hashes a few bits
    apart, so a cheap integer comparison finds syndicated wire copy and lightly
    reworded reprints without any model involved. Shingles of three tokens are
    used rather than single words so that word order carries weight.
    """
    words = tokens(text)
    if not words:
        return 0
    if len(words) < 3:
        shingles = words
    else:
        shingles = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]

    weights = [0] * bits
    mask = (1 << bits) - 1
    for shingle in shingles:
        digest = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=bits // 8).digest(),
            "big",
        ) & mask
        for position in range(bits):
            if digest >> position & 1:
                weights[position] += 1
            else:
                weights[position] -= 1

    result = 0
    for position in range(bits):
        if weights[position] > 0:
            result |= 1 << position
    # Stored in a signed 64 bit column, so the top bit is folded into the sign
    # rather than overflowing on Postgres.
    if bits == 64 and result >= 1 << 63:
        result -= 1 << 64
    return result


def hamming(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 64
    return ((a ^ b) & 0xFFFFFFFFFFFFFFFF).bit_count()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_length: int = 72) -> str:
    slug = _SLUG_RE.sub("-", normalise_text(text)).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit("-", 1)[0]
    return slug or "item"


def truncate(text: str, limit: int, *, suffix: str = "...") -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - len(suffix))]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + suffix


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


# -------------------------------------------------------------------- JSON


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def loads(value: Any, default: Any = None) -> Any:
    """Read a JSON column that may already be decoded.

    Postgres hands back a JSONB column as a Python object while SQLite hands
    back the text, so both shapes reach the callers and both are accepted here.
    """
    if value is None or value == "":
        return default if default is not None else {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default if default is not None else {}


def tcp_reachable(url: str, timeout: float = 0.25) -> bool:
    """Whether something is listening on a URL's host and port.

    Used before making an HTTP request to a service that is usually not
    running, because the two cost wildly different amounts when it is absent.
    An HTTP GET to a closed localhost port took 4 seconds here: `localhost`
    resolves to both ::1 and 127.0.0.1 and Windows drops rather than refuses,
    so each family burns the full timeout. The same check over a raw socket at
    a quarter second costs 0.5 seconds, and when the port IS open it answers in
    about 11 milliseconds either way.

    That mattered: probing four local model servers made a cold /api/config
    take over twenty seconds.
    """
    import socket

    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False

    for family, socket_type, proto, _, address in infos:
        sock = socket.socket(family, socket_type, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(address)
            return True
        except OSError:
            continue
        finally:
            sock.close()
    return False


def map_concurrently(function: Callable[[Any], Any], items: Sequence[Any],
                     *, max_workers: int = 12) -> list:
    """Run an independent call per item in parallel, preserving context.

    Thread pool workers do NOT inherit the caller's context variables, and this
    project keeps each request's own API keys in one. Mapping provider status
    across a plain pool therefore reported every bring your own key provider as
    unavailable: the key was bound on the request thread and invisible in the
    worker. A separate copy of the calling context is taken per task here, in
    the calling thread, because one Context cannot be entered by two threads at
    once.
    """
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    items = list(items)
    if not items:
        return []
    if len(items) == 1:
        return [function(items[0])]

    contexts = [copy_context() for _ in items]
    with ThreadPoolExecutor(max_workers=min(len(items), max_workers)) as pool:
        return list(
            pool.map(lambda pair: pair[0].run(function, pair[1]), zip(contexts, items, strict=True))
        )
