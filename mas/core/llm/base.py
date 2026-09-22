"""Provider neutral contracts for the LLM layer."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from . import keyring

# One global style rule appended to every system prompt, so a digest reads the
# same regardless of which provider wrote it.
STYLE_RULES = (
    "Writing rules that override any other style preference:\n"
    "1. Never use em dashes or en dashes. Use a comma, a colon, a full stop, "
    "or the word 'to' for ranges.\n"
    "2. Use plain ASCII punctuation throughout.\n"
    "3. Do not open with filler such as 'Certainly' or 'Here is'.\n"
    "4. Never invent a fact, a number, a quote or a name that is not in the "
    "source text. If the sources do not say, write that they do not say."
)


class LLMError(RuntimeError):
    """Raised when a provider call cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        status: int | None = None,
        retryable: bool = False,
        hint: str = "",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status = status
        self.retryable = retryable
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "retryable": self.retryable,
            "hint": self.hint,
        }


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    note: str = ""
    # Roughly how cheap and fast this model is, used to pick a default for the
    # high volume classification stage where quality matters least.
    cheap: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "note": self.note, "cheap": self.cheap}


@dataclass
class ProviderStatus:
    id: str
    label: str
    available: bool
    reason: str = ""
    models: Sequence[ModelSpec] = field(default_factory=tuple)
    default_model: str = ""
    docs_url: str = ""
    key_names: Sequence[str] = field(default_factory=tuple)
    # "client" when this request supplied the key, "server" when it comes from
    # the deployment's environment, "" when there is no key at all.
    key_source: str = ""
    # True when the provider needs a second value alongside the key, which
    # Cloudflare does because its account id sits in the URL path.
    needs_account: bool = False
    has_account: bool = False
    # True for a model server the operator runs themselves, which needs no key
    # but is only reachable from a server tier deployment.
    local: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "available": self.available,
            "reason": self.reason,
            "models": [m.as_dict() for m in self.models],
            "default_model": self.default_model,
            "docs_url": self.docs_url,
            "key_names": list(self.key_names),
            "key_source": self.key_source,
            "needs_account": self.needs_account,
            "has_account": self.has_account,
            "local": self.local,
        }


@dataclass
class Usage:
    """What one call consumed, for the cost meter."""

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class NativeToolCall:
    """A tool call the provider returned through its own function calling API."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Completion:
    text: str
    usage: Usage
    # Populated only when the caller passed tool schemas and the model chose to
    # call one. Some models, notably the gpt-oss family, will emit a native
    # tool call whether or not tools were declared, so a caller that describes
    # tools in the prompt alone gets a hard 400 from the provider. Declaring
    # them properly is the fix, and this is where the answer comes back.
    tool_calls: list[NativeToolCall] = field(default_factory=list)
    finish_reason: str = ""


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*|\s*```$")

# Models ignore style instructions often enough that the rule is also enforced
# on every response. Built from code points so that no source file in this
# project contains one of these characters itself.
# En dash, em dash, figure dash, horizontal bar, plus the non breaking and
# plain Unicode hyphens. The last two are not em dashes and so were missed
# at first, but they are still non ASCII and a Windows console cannot encode
# them, which turns a perfectly good report into a UnicodeEncodeError.
_DASH = "".join(chr(code) for code in (0x2013, 0x2014, 0x2012, 0x2015, 0x2010, 0x2011))
_RANGE_DASH_RE = re.compile(rf"(?<=\d)\s*[{_DASH}]\s*(?=\d)")
_LEADING_DASH_RE = re.compile(rf"(?m)^([ \t]*)[{_DASH}][ \t]+")
_SPACED_DASH_RE = re.compile(rf"[ \t]+[{_DASH}][ \t]+")
_ANY_DASH_RE = re.compile(rf"[{_DASH}]")
_DOUBLE_PUNCT_RE = re.compile(r"([,;:])[ \t]*,[ \t]+")


# Several models write a citation marker with CJK lenticular brackets, so a
# cited report comes back reading 1 rather than [1] and stops matching the
# numbered list underneath it. Same class of problem as the dashes: the model
# ignores the instruction, and the characters are outside cp1252 as well, so
# printing the report on a Windows console raises UnicodeEncodeError.
_BRACKETS = {
    0x3010: "[",
    0x3011: "]",
    0xFF3B: "[",
    0xFF3D: "]",
    0xFF08: "(",
    0xFF09: ")",
}


def sanitise_output(text: str) -> str:
    """Replace typographic dashes and brackets with plain ASCII punctuation."""
    if not text:
        return text
    cleaned = _RANGE_DASH_RE.sub(" to ", text.translate(_BRACKETS))
    cleaned = _LEADING_DASH_RE.sub(lambda m: m.group(1) + "- ", cleaned)
    cleaned = _SPACED_DASH_RE.sub(", ", cleaned)
    cleaned = _ANY_DASH_RE.sub("-", cleaned)
    cleaned = _DOUBLE_PUNCT_RE.sub(lambda m: m.group(1) + " ", cleaned)
    return cleaned


def coerce_json(raw: str) -> Any:
    """Parse model output that is meant to be JSON but may carry noise.

    Handles bare JSON, fenced JSON, and JSON embedded in prose.
    """
    if raw is None:
        raise ValueError("Model returned no content.")
    text = raw.strip()
    if not text:
        raise ValueError("Model returned an empty response.")

    text = _FENCE_RE.sub("", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"Model did not return valid JSON. First 300 characters: {text[:300]!r}"
    )


class BaseProvider:
    """Interface every provider adapter implements.

    An adapter is a process wide singleton, so the credential cannot live on
    the instance. `api_key` prefers the key the current request brought and
    falls back to the deployment's environment value, which lets one object
    serve a bring your own key visitor and a server key deployment without
    either knowing about the other.
    """

    id: str = ""
    label: str = ""
    docs_url: str = ""
    key_names: tuple[str, ...] = ()
    models: tuple[ModelSpec, ...] = ()

    env_key: str | None = None
    needs_account: bool = False
    # Whether `generate` accepts a `tools` argument and can return tool calls.
    # The agent loop uses the native path where this is true and falls back to
    # a JSON protocol in the prompt everywhere else.
    supports_tools: bool = False
    # A model server the operator runs. Needs no key, but is unreachable from
    # a serverless deployment.
    local: bool = False

    @property
    def account_id(self) -> str:
        return ""

    @property
    def api_key(self) -> str | None:
        return keyring.key_for(self.id) or self.env_key

    @property
    def key_source(self) -> str:
        if keyring.key_for(self.id):
            return "client"
        return "server" if self.env_key else ""

    def has_server_key(self) -> bool:
        return bool(self.env_key)

    def is_available(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def unavailable_reason(self) -> str:
        return (
            f"Paste a {self.label} API key in Settings, or set "
            f"{' or '.join(self.key_names)} on the server."
        )

    def default_model(self) -> str:
        return self.models[0].id if self.models else ""

    def cheap_model(self) -> str:
        """The model to use for high volume, low judgement work.

        Classification runs once per article and summarisation runs once per
        event, so the two stages have very different cost profiles. Picking a
        small model for the former is the single biggest lever on what a run
        costs.
        """
        for spec in self.models:
            if spec.cheap:
                return spec.id
        return self.default_model()

    def status(self) -> ProviderStatus:
        available = self.is_available()
        return ProviderStatus(
            id=self.id,
            label=self.label,
            available=available,
            reason="" if available else self.unavailable_reason(),
            models=self.models if available else (),
            default_model=self.default_model() if available else "",
            docs_url=self.docs_url,
            key_names=self.key_names,
            key_source=self.key_source,
            needs_account=self.needs_account,
            has_account=bool(self.account_id),
            local=self.local,
        )

    def verify(self) -> list[str]:
        """Check the credential against the provider and list callable models."""
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError
