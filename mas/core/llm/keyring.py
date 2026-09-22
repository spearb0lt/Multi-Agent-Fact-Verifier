"""Per request provider credentials, so a visitor can bring their own key.

The server may hold no keys at all. When a browser sends a key in a request
header, that key is bound to a context variable for the lifetime of that one
request and is used in place of the server's own environment value. Nothing is
written to disk, added to a log line, held between requests, or echoed back in
a response.

A context variable is the right scope because the provider adapters are process
wide singletons: the adapter object is shared, the credential is not.
"""
from __future__ import annotations

import hashlib
import re
from contextvars import ContextVar

# Header names a browser uses. A per provider header keeps the wire format
# obvious in a network inspector and avoids inventing an encoding.
#   X-LLM-Key-gemini:     <the visitor's Google AI Studio key>
#   X-LLM-Base-openai:    https://my-gateway.example/v1
#   X-LLM-Account-cloudflare: <account id>
KEY_PREFIX = "x-llm-key-"
BASE_URL_PREFIX = "x-llm-base-"
ACCOUNT_PREFIX = "x-llm-account-"

# Non LLM credentials ride the same mechanism under reserved slugs, so a
# visitor can supply their own NewsAPI or Reddit key too.
DATA_SLUGS = frozenset(
    {
        "newsapi", "gnews", "newsdata", "currents", "mediastack", "worldnews",
        "thenewsapi", "marketaux", "finnhub", "alphavantage", "guardian", "nyt",
        "youtube", "reddit", "reddit_secret", "serpapi", "brave", "tavily",
        "exa", "jina", "voyage", "cohere",
    }
)

# A credential longer than this, or carrying anything outside printable ASCII,
# is a mistake or an injection attempt rather than a real key. Rejecting it
# here means it never reaches an upstream Authorization header.
MAX_KEY_LENGTH = 512
_SAFE_KEY = re.compile(r"^[\x21-\x7e]+$")
_SAFE_URL = re.compile(r"^https?://[\x21-\x7e]+$")
# An account id is interpolated into a URL path, so it is held to letters and
# digits only. That makes it impossible for a crafted value to escape its path
# segment or point the request at a different host.
_SAFE_ACCOUNT = re.compile(r"^[0-9a-zA-Z]{8,64}$")

_EMPTY: dict[str, str] = {}

_keys: ContextVar[dict[str, str]] = ContextVar("llm_client_keys", default=_EMPTY)
_base_urls: ContextVar[dict[str, str]] = ContextVar("llm_client_base_urls", default=_EMPTY)
_accounts: ContextVar[dict[str, str]] = ContextVar("llm_client_accounts", default=_EMPTY)


def clean_key(value: str | None) -> str:
    candidate = (value or "").strip()
    if not candidate or len(candidate) > MAX_KEY_LENGTH:
        return ""
    return candidate if _SAFE_KEY.match(candidate) else ""


def clean_base_url(value: str | None) -> str:
    """Only an http(s) gateway URL is accepted as an override.

    Plain http is allowed because a local model server (Ollama, LM Studio)
    listens on http, and those are only reachable from a server tier
    deployment in the first place.
    """
    candidate = (value or "").strip().rstrip("/")
    if not candidate or len(candidate) > MAX_KEY_LENGTH:
        return ""
    return candidate if _SAFE_URL.match(candidate) else ""


def clean_account(value: str | None) -> str:
    candidate = (value or "").strip()
    return candidate if _SAFE_ACCOUNT.match(candidate) else ""


def bind(
    keys: dict[str, str] | None,
    base_urls: dict[str, str] | None = None,
    accounts: dict[str, str] | None = None,
) -> None:
    """Attach a request's credentials to the current context."""
    _keys.set({k: v for k, v in (keys or {}).items() if v} or _EMPTY)
    _base_urls.set({k: v for k, v in (base_urls or {}).items() if v} or _EMPTY)
    _accounts.set({k: v for k, v in (accounts or {}).items() if v} or _EMPTY)


def reset() -> None:
    _keys.set(_EMPTY)
    _base_urls.set(_EMPTY)
    _accounts.set(_EMPTY)


def key_for(provider_id: str) -> str:
    return _keys.get().get(provider_id, "")


def base_url_for(provider_id: str) -> str:
    return _base_urls.get().get(provider_id, "")


def account_for(provider_id: str) -> str:
    return _accounts.get().get(provider_id, "")


def supplied() -> frozenset[str]:
    """Slugs the current request brought a key for, safe to report back."""
    return frozenset(_keys.get())


def fingerprint(value: str | None) -> str:
    """A short, non reversible tag for a credential.

    Used only as a cache key, so a model list built for one visitor's key is
    never handed to another visitor.
    """
    if not value:
        return "none"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
