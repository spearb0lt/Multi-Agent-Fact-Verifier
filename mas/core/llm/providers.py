"""Concrete provider adapters.

Most providers expose an OpenAI compatible chat completions endpoint, so one
adapter with a different base URL covers the majority of them, including the
model servers an operator runs locally. Gemini and Anthropic each have their
own SDK and get their own adapter.
"""
from __future__ import annotations

import json
import re
import time
from contextlib import suppress
from typing import Any

from .. import settings
from ..runtime import current as runtime
from ..util import tcp_reachable
from . import keyring
from .base import (
    STYLE_RULES,
    BaseProvider,
    Completion,
    LLMError,
    ModelSpec,
    NativeToolCall,
    Usage,
    sanitise_output,
)

# Model catalogues move fast, so every adapter asks its provider what it can
# actually serve and merges that with a curated list. Discovery is cached for
# the lifetime of the process, and a failure simply leaves the curated list.
_DISCOVERY_TTL = 900.0
_discovered: dict[str, tuple[float, list[str]]] = {}
_CACHE_CAP = 64

# Ids that are not chat models, or that would only clutter the picker.
_SKIP_MODEL = re.compile(
    r"whisper|tts|speech|audio|orpheus|guard|embed|rerank|moderation|image|dall|"
    r"sora|veo|imagen|video|transcribe|realtime|search-|computer-use|aqa|"
    r"-tuning|bge-|stable-diffusion|flux|upscal|melotts|resnet|detr|m2m100",
    re.IGNORECASE,
)

_FAMILY_ORDER = (
    "llama", "qwen", "deepseek", "gpt", "gemini", "claude", "mistral", "mixtral",
    "gemma", "glm", "kimi", "command", "minimax", "nemotron", "phi", "granite",
)

_GOOD_FAMILY = re.compile(
    r"llama|qwen|deepseek|mistral|mixtral|gemma|phi-|command|nemotron|olmo|"
    r"glm|kimi|minimax|hermes|granite|smol|gpt-oss|compound",
    re.IGNORECASE,
)

_TRANSIENT_MARKERS = (
    "429", "resource_exhausted", "rate limit", "rate_limit", "503",
    "unavailable", "overloaded", "timeout", "timed out", "502", "504",
    "connection reset", "temporarily",
)

# Models that spend output tokens on hidden reasoning before answering.
_REASONING_MODEL = re.compile(
    r"gpt-oss|qwen3|deepseek-r1|magistral|o1-|o3-|o4-|thinking", re.IGNORECASE
)

_RETRY_AFTER_RE = re.compile(r"try again in\s*(\d+(?:\.\d+)?)\s*(ms|s|m)\b", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"retry[^0-9]{0,20}(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _family_rank(model_id: str) -> tuple[int, int, str]:
    lowered = model_id.lower()
    rank = len(_FAMILY_ORDER)
    for index, family in enumerate(_FAMILY_ORDER):
        if family in lowered:
            rank = index
            break
    tuned = 0 if ("instruct" in lowered or "-it" in lowered or "chat" in lowered) else 1
    return (rank, tuned, lowered)


def _is_transient(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def _retry_delay(message: str, attempt: int) -> float:
    """How long to wait before retrying, honouring the provider's own advice."""
    match = _RETRY_AFTER_RE.search(message)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        seconds = value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
        return min(seconds + 1.0, 45.0)
    match = _RETRY_DELAY_RE.search(message)
    if match:
        return min(float(match.group(1)) + 1.0, 45.0)
    return min(2.5 * (attempt + 1), 20.0)


def _hint_for(message: str, provider_label: str) -> str:
    lowered = message.lower()
    if "429" in lowered or "quota" in lowered or "rate limit" in lowered:
        return (
            f"{provider_label} rejected the call for rate or quota reasons. Quotas "
            "are counted per model, so another model in the picker often still "
            "works. Otherwise wait a moment or switch provider."
        )
    if "402" in lowered or "credit" in lowered or "billing" in lowered:
        return (
            f"The {provider_label} account has no credit left for inference. Top it "
            "up, or switch provider in the model picker."
        )
    if "not available on" in lowered or "model agreement" in lowered:
        return (
            f"Your {provider_label} account cannot call that model. It may need a "
            "paid plan or an agreement you have not accepted. Choose another model."
        )
    if "413" in lowered or "context" in lowered and "length" in lowered:
        return (
            "That model's context window is too small for this request. Choose a "
            "model with a larger context, or shorten the input."
        )
    if any(token in lowered for token in ("401", "403", "invalid api key", "unauthor")):
        return (
            f"The {provider_label} API key was rejected. Check that it is the whole "
            "key, that it is still active, and that it belongs to this provider."
        )
    if any(token in lowered for token in ("404", "not found", "does not exist", "decommission")):
        return f"{provider_label} does not recognise that model id. Choose another."
    if "connect" in lowered or "refused" in lowered:
        return (
            f"{provider_label} could not be reached. If this is a model server you "
            "run yourself, check that it is running and that the base URL is right."
        )
    return ""


def _cache_key(provider_id: str, api_key: str | None) -> str:
    return f"{provider_id}:{keyring.fingerprint(api_key)}"


def _cache_get(provider_id: str, api_key: str | None = None) -> list[str] | None:
    entry = _discovered.get(_cache_key(provider_id, api_key))
    if entry and (time.monotonic() - entry[0]) < _DISCOVERY_TTL:
        return entry[1]
    return None


def _cache_put(provider_id: str, ids: list[str], api_key: str | None = None) -> None:
    if len(_discovered) >= _CACHE_CAP:
        for stale in sorted(_discovered, key=lambda k: _discovered[k][0])[: _CACHE_CAP // 2]:
            _discovered.pop(stale, None)
    _discovered[_cache_key(provider_id, api_key)] = (time.monotonic(), ids)


def _content_text(content: Any) -> str:
    """Normalise a chat completion's content into text.

    The OpenAI schema types this as a string, but not every gateway obeys.
    Some return an already parsed object, and the multipart form arrives as a
    list of blocks, so both are flattened rather than allowed to crash.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def _compose_system(system: str | None) -> str:
    return f"{system.strip()}\n\n{STYLE_RULES}" if system else STYLE_RULES


def _extra(provider_id: str) -> list[ModelSpec]:
    raw = settings.EXTRA_MODELS.get(provider_id) or ""
    return [
        ModelSpec(id=chunk.strip(), label=chunk.strip(), note="From environment")
        for chunk in raw.split(",")
        if chunk.strip()
    ]


def _promote(
    models: list[ModelSpec], preferred: str | None, live_ids: list[str] | None = None
) -> tuple[ModelSpec, ...]:
    """Move an environment preferred model to the front of the picker.

    An id the provider does not serve is only trusted when discovery told us
    nothing. Providers retire models without warning, and a pinned id that has
    since been decommissioned would otherwise become the default and make every
    call fail with a 404 while the picker sat full of models that do work.
    Groq dropped the entire Llama 3.x family this way.
    """
    if not preferred:
        return tuple(models)
    preferred = preferred.strip()
    for index, spec in enumerate(models):
        if spec.id == preferred:
            return tuple([models[index]] + models[:index] + models[index + 1 :])

    if live_ids:
        # Discovery worked and did not list it, so it is gone. Keep the served
        # models rather than defaulting to one that cannot be called.
        return tuple(models)
    return tuple([ModelSpec(preferred, preferred, note="From environment")] + models)


def _merge_catalogue(
    curated: list[ModelSpec],
    live_ids: list[str] | None,
    *,
    cap: int,
    families_only: bool = False,
) -> list[ModelSpec]:
    """Combine the curated list with what the provider says it can serve."""
    if live_ids is None:
        return curated

    live = set(live_ids)
    merged = [spec for spec in curated if spec.id in live]
    known = {spec.id for spec in merged}

    extras = [
        model_id
        for model_id in live_ids
        if model_id not in known
        and not _SKIP_MODEL.search(model_id)
        and (not families_only or _GOOD_FAMILY.search(model_id))
    ]
    extras.sort(key=_family_rank)
    for model_id in extras:
        if len(merged) >= cap:
            break
        merged.append(
            ModelSpec(model_id, model_id.split("/")[-1], note="Offered by your account")
        )

    return merged or curated


def _parse_native_calls(message_obj: Any) -> list[NativeToolCall]:
    """Read tool calls off an OpenAI shaped message.

    Arguments arrive as a JSON string and are occasionally malformed, so a call
    that cannot be parsed comes back with empty arguments rather than being
    dropped: the agent loop turns that into an observation the model can
    correct, which is far better than silently losing the turn.
    """
    raw = getattr(message_obj, "tool_calls", None) or []
    calls: list[NativeToolCall] = []
    for item in raw:
        function = getattr(item, "function", None)
        name = str(getattr(function, "name", "") or "")
        if not name:
            continue
        arguments = getattr(function, "arguments", "") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            NativeToolCall(id=str(getattr(item, "id", "") or name), name=name, arguments=arguments)
        )
    return calls


# Groq returns this when a model emits a tool call but the request declared no
# tools. The intended call is in `failed_generation`, so it can be recovered
# rather than costing the whole turn.
_FAILED_GENERATION_RE = re.compile(r"'failed_generation':\s*(?P<q>['\"])(?P<body>.*?)(?P=q)\}", re.S)


def _recover_failed_generation(message: str) -> list[NativeToolCall]:
    match = _FAILED_GENERATION_RE.search(message)
    if not match:
        return []
    body = match.group("body").encode().decode("unicode_escape")
    try:
        payload = json.loads(body)
    except ValueError:
        return []
    if isinstance(payload, dict) and payload.get("name"):
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        return [
            NativeToolCall(
                id=str(payload.get("name")),
                name=str(payload["name"]),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        ]
    return []


class OpenAICompatibleProvider(BaseProvider):
    """Shared adapter for any OpenAI compatible chat completions endpoint."""

    supports_tools: bool = True
    env_base_url: str = ""
    max_attempts: int = 3
    model_cap: int = 24
    families_only: bool = False
    curated: list[ModelSpec] = []
    preferred: str | None = None
    # A local server authenticates with nothing, so an empty key is fine.
    key_optional: bool = False

    @property
    def base_url(self) -> str:
        """A visitor may point the adapter at their own compatible gateway."""
        return keyring.base_url_for(self.id) or self.env_base_url

    def is_available(self) -> bool:
        if self.local and not runtime().can("local_llm"):
            return False
        return bool(self.api_key) or self.key_optional

    def unavailable_reason(self) -> str:
        if self.local and not runtime().can("local_llm"):
            return runtime().reason("local_llm")
        return super().unavailable_reason()

    def _discover(self) -> list[str]:
        """Ask the gateway which models this key can actually call."""
        if not (self.api_key or self.key_optional):
            return []
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=self.api_key or "not-needed",
                base_url=self.base_url,
                timeout=8.0,
                max_retries=0,
            )
            return [item.id for item in client.models.list().data if getattr(item, "id", None)]
        except Exception:  # noqa: BLE001 - discovery is best effort
            return []

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        tag = self.api_key or self.base_url
        live = _cache_get(self.id, tag)
        if live is None:
            live = self._discover()
            _cache_put(self.id, live, tag)
        merged = _merge_catalogue(
            list(self.curated), live, cap=self.model_cap, families_only=self.families_only
        )
        return _promote(merged, self.preferred, live)

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "The openai package is not installed on the server.", provider=self.id
            ) from exc
        if not (self.api_key or self.key_optional):
            raise LLMError(
                self.unavailable_reason(), provider=self.id, hint="Missing API key."
            )
        return OpenAI(
            api_key=self.api_key or "not-needed",
            base_url=self.base_url,
            timeout=float(settings.FETCH_TIMEOUT * 4),
            max_retries=0,
        )

    def verify(self) -> list[str]:
        """Prove the credential works, and report what it can call."""
        client = self._client()
        try:
            ids = [item.id for item in client.models.list().data if getattr(item, "id", None)]
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            message = str(exc)
            raise LLMError(
                f"{self.label} did not accept that key.",
                provider=self.id,
                hint=_hint_for(message, self.label),
            ) from exc
        _cache_put(self.id, ids, self.api_key or self.base_url)
        return ids

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        system_text = _compose_system(system)
        # JSON mode and tool calling are mutually exclusive on every gateway
        # that supports both: the model has to answer either with an object or
        # with a call, and asking for both produces a 400 on some and silent
        # nonsense on others.
        if json_mode and not tools:
            system_text += (
                "\n\nReturn only a single valid JSON value. No prose, no markdown fences."
            )
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ]

        client = self._client()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            json_mode = False
        elif json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if _REASONING_MODEL.search(model):
            # Reasoning models spend the output budget thinking. Keep that
            # short so the visible answer still fits, and give it more room.
            kwargs["reasoning_effort"] = "low"
            kwargs["max_tokens"] = max(max_tokens, 8192)

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0] if response.choices else None
                message_obj = choice.message if choice else None
                text = _content_text(
                    getattr(message_obj, "content", None) if message_obj else None
                )
                calls = _parse_native_calls(message_obj) if message_obj else []
                finish = (getattr(choice, "finish_reason", "") or "") if choice else ""

                # A tool call with no prose alongside it is a complete and
                # correct answer, so an empty `content` is only a problem when
                # the model produced nothing at all.
                if text.strip() or calls:
                    raw_usage = getattr(response, "usage", None)
                    return Completion(
                        text=sanitise_output(text.strip()) if text.strip() else "",
                        usage=Usage(
                            provider=self.id,
                            model=model,
                            input_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                            output_tokens=int(
                                getattr(raw_usage, "completion_tokens", 0) or 0
                            ),
                        ),
                        tool_calls=calls,
                        finish_reason=finish,
                    )

                if finish == "length":
                    raise RuntimeError(
                        "The model ran out of output tokens before producing an answer. "
                        "A reasoning model usually spent the budget thinking. Pick a "
                        "non reasoning model, or shorten the input."
                    )
                if finish in ("content_filter", "safety"):
                    raise RuntimeError(
                        "The provider's safety filter blocked this response."
                    )
                raise RuntimeError(
                    f"The model returned an empty response (finish reason: {finish or 'none'})."
                )
            except Exception as exc:  # noqa: BLE001 - normalised below
                message = str(exc)
                last_error = exc
                lowered = message.lower()

                # Some models, the gpt-oss family especially, emit a native
                # tool call whether or not the request declared any tools. The
                # gateway rejects that with a 400 and hands back the call it
                # tried to make, so the turn is recoverable rather than lost.
                if "tool_use_failed" in lowered or "model called a tool" in lowered:
                    recovered = _recover_failed_generation(message)
                    if recovered:
                        raw_usage = None
                        return Completion(
                            text="",
                            usage=Usage(provider=self.id, model=model),
                            tool_calls=recovered,
                            finish_reason="tool_calls",
                        )

                # Gateways vary in which optional parameters they accept. Each
                # of these is dropped once and the call retried, rather than
                # failing over a parameter the request did not need.
                if "tools" in kwargs and (
                    "tool" in lowered and ("not supported" in lowered or "unsupported" in lowered)
                ):
                    # The model cannot take tool schemas. Fall back to the
                    # prompt protocol rather than failing the whole run.
                    kwargs.pop("tools", None)
                    kwargs.pop("tool_choice", None)
                    continue
                if json_mode and ("response_format" in lowered or "json_object" in lowered):
                    kwargs.pop("response_format", None)
                    json_mode = False
                    continue
                if "max_completion_tokens" in lowered and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "temperature" in lowered and "unsupported" in lowered:
                    kwargs.pop("temperature", None)
                    continue
                if "reasoning_effort" in lowered and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort", None)
                    continue
                if _is_transient(message) and attempt < self.max_attempts - 1:
                    time.sleep(_retry_delay(message, attempt))
                    continue
                break

        message = str(last_error) if last_error else "Unknown provider error."
        raise LLMError(
            f"{self.label} call failed: {message[:400]}",
            provider=self.id,
            model=model,
            retryable=_is_transient(message),
            hint=_hint_for(message, self.label),
        )


# --------------------------------------------------------- hosted providers


class GroqProvider(OpenAICompatibleProvider):
    id = "groq"
    label = "Groq"
    docs_url = "https://console.groq.com/keys"
    key_names = ("GROQ_API_KEY",)
    env_base_url = "https://api.groq.com/openai/v1"
    model_cap = 20

    def __init__(self) -> None:
        self.env_key = settings.GROQ_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("groq")
        # Groq retired the whole Llama 3.x family, so nothing from it is listed
        # here. Discovery adds whatever else the account can call.
        self.curated = [
            ModelSpec("openai/gpt-oss-120b", "GPT OSS 120B",
                      note="Strong all rounder, the best default here"),
            ModelSpec("openai/gpt-oss-20b", "GPT OSS 20B",
                      note="Fast and cheap, good for classification", cheap=True),
            ModelSpec("qwen/qwen3.8-27b", "Qwen3.8 27B"),
            ModelSpec("groq/compound", "Compound", note="Tool augmented"),
            ModelSpec("groq/compound-mini", "Compound Mini", cheap=True),
        ] + _extra(self.id)


class OpenAIProvider(OpenAICompatibleProvider):
    id = "openai"
    label = "OpenAI"
    docs_url = "https://platform.openai.com/api-keys"
    key_names = ("OPENAI_API_KEY",)
    model_cap = 22

    def __init__(self) -> None:
        self.env_key = settings.OPENAI_API_KEY
        self.env_base_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
        self.preferred = settings.MODEL_OVERRIDES.get("openai")
        self.curated = [
            ModelSpec("gpt-4o-mini", "GPT-4o mini", note="Fast and inexpensive", cheap=True),
            ModelSpec("gpt-4o", "GPT-4o", note="Highest quality summaries"),
            ModelSpec("gpt-4.1-mini", "GPT-4.1 mini", cheap=True),
            ModelSpec("gpt-4.1", "GPT-4.1", note="Large context"),
        ] + _extra(self.id)

    def unavailable_reason(self) -> str:
        return (
            "Paste an OpenAI API key in Settings, or set OPENAI_API_KEY on the "
            "server. Any OpenAI compatible gateway also works if you give its base URL."
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    id = "openrouter"
    label = "OpenRouter"
    docs_url = "https://openrouter.ai/keys"
    key_names = ("OPENROUTER_API_KEY",)
    env_base_url = "https://openrouter.ai/api/v1"
    model_cap = 30
    families_only = True

    def __init__(self) -> None:
        self.env_key = settings.OPENROUTER_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("openrouter")
        self.curated = [
            ModelSpec("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B Instruct"),
            ModelSpec("google/gemini-2.0-flash-001", "Gemini 2.0 Flash", cheap=True),
            ModelSpec("deepseek/deepseek-chat", "DeepSeek Chat", cheap=True),
        ] + _extra(self.id)


class TogetherProvider(OpenAICompatibleProvider):
    id = "together"
    label = "Together AI"
    docs_url = "https://api.together.ai/settings/api-keys"
    key_names = ("TOGETHER_API_KEY",)
    env_base_url = "https://api.together.xyz/v1"
    model_cap = 26
    families_only = True

    def __init__(self) -> None:
        self.env_key = settings.TOGETHER_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("together")
        self.curated = [
            ModelSpec("meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama 3.3 70B Turbo"),
            ModelSpec("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                      "Llama 3.1 8B Turbo", cheap=True),
        ] + _extra(self.id)


class DeepSeekProvider(OpenAICompatibleProvider):
    id = "deepseek"
    label = "DeepSeek"
    docs_url = "https://platform.deepseek.com/api_keys"
    key_names = ("DEEPSEEK_API_KEY",)
    env_base_url = "https://api.deepseek.com/v1"
    model_cap = 8

    def __init__(self) -> None:
        self.env_key = settings.DEEPSEEK_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("deepseek")
        self.curated = [
            ModelSpec("deepseek-chat", "DeepSeek Chat", note="Fast and cheap", cheap=True),
            ModelSpec("deepseek-reasoner", "DeepSeek Reasoner", note="Deeper reasoning"),
        ] + _extra(self.id)


class MistralProvider(OpenAICompatibleProvider):
    id = "mistral"
    label = "Mistral"
    docs_url = "https://console.mistral.ai/api-keys"
    key_names = ("MISTRAL_API_KEY",)
    env_base_url = "https://api.mistral.ai/v1"
    model_cap = 14

    def __init__(self) -> None:
        self.env_key = settings.MISTRAL_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("mistral")
        self.curated = [
            ModelSpec("mistral-small-latest", "Mistral Small", cheap=True),
            ModelSpec("mistral-large-latest", "Mistral Large"),
            ModelSpec("open-mistral-nemo", "Mistral Nemo", cheap=True),
        ] + _extra(self.id)


class CerebrasProvider(OpenAICompatibleProvider):
    id = "cerebras"
    label = "Cerebras"
    docs_url = "https://cloud.cerebras.ai"
    key_names = ("CEREBRAS_API_KEY",)
    env_base_url = "https://api.cerebras.ai/v1"
    model_cap = 12

    def __init__(self) -> None:
        self.env_key = settings.CEREBRAS_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("cerebras")
        self.curated = [
            ModelSpec("llama-3.3-70b", "Llama 3.3 70B", note="Very fast inference"),
            ModelSpec("llama3.1-8b", "Llama 3.1 8B", cheap=True),
        ] + _extra(self.id)


class SambaNovaProvider(OpenAICompatibleProvider):
    id = "sambanova"
    label = "SambaNova"
    docs_url = "https://cloud.sambanova.ai/apis"
    key_names = ("SAMBANOVA_API_KEY",)
    env_base_url = "https://api.sambanova.ai/v1"
    model_cap = 14

    def __init__(self) -> None:
        self.env_key = settings.SAMBANOVA_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("sambanova")
        self.curated = [
            ModelSpec("Meta-Llama-3.3-70B-Instruct", "Llama 3.3 70B Instruct"),
            ModelSpec("Meta-Llama-3.1-8B-Instruct", "Llama 3.1 8B Instruct", cheap=True),
        ] + _extra(self.id)


class XAIProvider(OpenAICompatibleProvider):
    id = "xai"
    label = "xAI Grok"
    docs_url = "https://console.x.ai"
    key_names = ("XAI_API_KEY",)
    env_base_url = "https://api.x.ai/v1"
    model_cap = 10

    def __init__(self) -> None:
        self.env_key = settings.XAI_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("xai")
        self.curated = [
            ModelSpec("grok-2-latest", "Grok 2"),
            ModelSpec("grok-beta", "Grok Beta"),
        ] + _extra(self.id)


class FireworksProvider(OpenAICompatibleProvider):
    id = "fireworks"
    label = "Fireworks AI"
    docs_url = "https://fireworks.ai/api-keys"
    key_names = ("FIREWORKS_API_KEY",)
    env_base_url = "https://api.fireworks.ai/inference/v1"
    model_cap = 20
    families_only = True

    def __init__(self) -> None:
        self.env_key = settings.FIREWORKS_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("fireworks")
        self.curated = [
            ModelSpec("accounts/fireworks/models/llama-v3p3-70b-instruct",
                      "Llama 3.3 70B Instruct"),
        ] + _extra(self.id)


class PerplexityProvider(OpenAICompatibleProvider):
    id = "perplexity"
    label = "Perplexity"
    docs_url = "https://www.perplexity.ai/settings/api"
    key_names = ("PERPLEXITY_API_KEY",)
    env_base_url = "https://api.perplexity.ai"
    model_cap = 8

    def __init__(self) -> None:
        self.env_key = settings.PERPLEXITY_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("perplexity")
        self.curated = [
            ModelSpec("sonar", "Sonar", note="Answers with live web citations", cheap=True),
            ModelSpec("sonar-pro", "Sonar Pro", note="Deeper web research"),
        ] + _extra(self.id)

    def _discover(self) -> list[str]:
        # Perplexity serves no models list, so the curated set is the catalogue.
        return []


class HuggingFaceProvider(OpenAICompatibleProvider):
    id = "huggingface"
    label = "Hugging Face"
    docs_url = "https://huggingface.co/settings/tokens"
    key_names = ("HUGGINGFACE_API_KEY", "HF_TOKEN")
    model_cap = 30
    families_only = True

    def __init__(self) -> None:
        self.env_key = settings.HUGGINGFACE_API_KEY
        self.env_base_url = (
            settings.HUGGINGFACE_BASE_URL or "https://router.huggingface.co/v1"
        )
        self.preferred = settings.MODEL_OVERRIDES.get("huggingface")
        self.curated = [
            ModelSpec("meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B Instruct"),
            ModelSpec("Qwen/Qwen2.5-72B-Instruct", "Qwen2.5 72B Instruct"),
            ModelSpec("deepseek-ai/DeepSeek-V3-0324", "DeepSeek V3", note="Long context"),
        ] + _extra(self.id)

    def unavailable_reason(self) -> str:
        return (
            "Paste a Hugging Face access token in Settings, or set "
            "HUGGINGFACE_API_KEY on the server. Which models you can call depends "
            "on the account and its inference credits."
        )


_CF_API_ROOT = "https://api.cloudflare.com/client/v4"


class CloudflareProvider(OpenAICompatibleProvider):
    """Cloudflare Workers AI, through its OpenAI compatible endpoint.

    Two things make this one different from the other OpenAI protocol
    providers: the account id is part of the URL rather than a header, so a
    token alone is not enough, and the compatibility layer serves no models
    list, so discovery uses Cloudflare's own catalogue endpoint.
    """

    id = "cloudflare"
    label = "Cloudflare Workers AI"
    docs_url = "https://dash.cloudflare.com/profile/api-tokens"
    key_names = ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID")
    model_cap = 26
    needs_account = True

    def __init__(self) -> None:
        self.env_key = settings.CLOUDFLARE_API_TOKEN
        self.env_account = settings.CLOUDFLARE_ACCOUNT_ID
        self.preferred = settings.MODEL_OVERRIDES.get("cloudflare")
        self.curated = [
            ModelSpec("@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                      "Llama 3.3 70B Instruct FP8 Fast",
                      note="Quick and capable default"),
            ModelSpec("@cf/openai/gpt-oss-120b", "GPT OSS 120B",
                      note="128k context, strongest reasoning here"),
            ModelSpec("@cf/meta/llama-3.1-8b-instruct-fp8", "Llama 3.1 8B Instruct FP8",
                      note="Cheapest per neuron", cheap=True),
            ModelSpec("@cf/meta/llama-3.2-3b-instruct", "Llama 3.2 3B Instruct",
                      note="Smallest, for classification", cheap=True),
        ] + _extra(self.id)

    @property
    def account_id(self) -> str:
        return keyring.account_for(self.id) or self.env_account or ""

    @property
    def base_url(self) -> str:
        override = keyring.base_url_for(self.id)
        if override:
            return override
        account = self.account_id
        return f"{_CF_API_ROOT}/accounts/{account}/ai/v1" if account else ""

    def is_available(self) -> bool:
        return bool(self.api_key and self.account_id)

    def unavailable_reason(self) -> str:
        if self.api_key and not self.account_id:
            return (
                "Cloudflare needs an account id as well as a token, because the "
                "account is part of the request URL. Add it in Settings, or set "
                "CLOUDFLARE_ACCOUNT_ID on the server."
            )
        if self.account_id and not self.api_key:
            return (
                "An account id is set but no token. Paste a Cloudflare API token in "
                "Settings, or set CLOUDFLARE_API_TOKEN on the server."
            )
        return (
            "Paste a Cloudflare API token and your account id in Settings, or set "
            "CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID on the server. Both are "
            "needed: the account id goes in the request URL."
        )

    def _discover(self) -> list[str]:
        token, account = self.api_key, self.account_id
        if not (token and account):
            return []
        try:
            import requests

            response = requests.get(
                f"{_CF_API_ROOT}/accounts/{account}/ai/models/search",
                params={"task": "Text Generation", "per_page": 100,
                        "hide_experimental": "true"},
                headers={"Authorization": f"Bearer {token}"},
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success"):
                return []
            return [
                name
                for item in (payload.get("result") or [])
                if isinstance(item, dict)
                and isinstance(name := (item.get("name") or item.get("id") or ""), str)
                and name.startswith(("@cf/", "@hf/"))
            ]
        except Exception:  # noqa: BLE001 - discovery is best effort
            return []

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        tag = f"{self.api_key}:{self.account_id}"
        live = _cache_get(self.id, tag)
        if live is None:
            live = self._discover()
            _cache_put(self.id, live, tag)
        return _promote(
            _merge_catalogue(list(self.curated), live, cap=self.model_cap),
            self.preferred,
            live,
        )

    def verify(self) -> list[str]:
        if not (self.api_key and self.account_id):
            raise LLMError(self.unavailable_reason(), provider=self.id)
        ids = self._discover()
        if not ids:
            raise LLMError(
                "Cloudflare did not accept that token and account id together.",
                provider=self.id,
                hint=(
                    "Check that the token carries the Workers AI Read permission and "
                    "that the account id is the 32 character value from the dashboard "
                    "overview page for the same account."
                ),
            )
        _cache_put(self.id, ids, f"{self.api_key}:{self.account_id}")
        return ids


# ----------------------------------------------------------- local servers


class LocalOpenAIProvider(OpenAICompatibleProvider):
    """A model server the operator runs, reached over the OpenAI protocol.

    Availability is probed rather than assumed, because the usual failure is
    that the server simply is not running. The probe result is cached briefly
    so listing providers does not pay for a socket connection every time.
    """

    local = True
    key_optional = True
    _probe_ttl = 30.0
    _absent_ttl = 180.0

    def __init__(self) -> None:
        self._probe: tuple[float, bool] = (0.0, False)

    def _reachable(self) -> bool:
        """Whether a model server, specifically, is answering on this port.

        Accepting any response below 500 is not enough. These defaults point at
        ordinary localhost ports, and vLLM's is 8000, which is also this app's
        own default. Something answering there with a 404 would otherwise be
        reported as an available provider with an empty model list. So the
        probe requires a real OpenAI style model listing.
        """
        stamp, ok = self._probe
        # A server that is absent stays absent for longer than one that is
        # present stays healthy, so a negative answer is cached for longer.
        ttl = self._probe_ttl if ok else self._absent_ttl
        if time.monotonic() - stamp < ttl:
            return ok

        ok = False
        # Cheap first: nothing listening means no HTTP request at all.
        if tcp_reachable(self.base_url):
            try:
                import requests

                response = requests.get(f"{self.base_url}/models", timeout=2.0)
                if response.status_code == 200:
                    payload = response.json()
                    ok = isinstance(payload, dict) and isinstance(payload.get("data"), list)
            except Exception:  # noqa: BLE001 - not running is the expected case
                ok = False
        self._probe = (time.monotonic(), ok)
        return ok

    def is_available(self) -> bool:
        if not runtime().can("local_llm"):
            return False
        return self._reachable()

    def unavailable_reason(self) -> str:
        if not runtime().can("local_llm"):
            return runtime().reason("local_llm")
        return (
            f"{self.label} is not answering at {self.base_url}. Start it, or set a "
            f"different base URL with {self.key_names[0] if self.key_names else 'the environment'}."
        )


class OllamaProvider(LocalOpenAIProvider):
    id = "ollama"
    label = "Ollama"
    docs_url = "https://ollama.com/download"
    key_names = ("OLLAMA_BASE_URL",)
    model_cap = 40

    def __init__(self) -> None:
        super().__init__()
        self.env_key = None
        # Ollama's OpenAI compatible surface lives under /v1 of its own port.
        root = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        self.env_base_url = root if root.endswith("/v1") else f"{root}/v1"
        self.preferred = settings.MODEL_OVERRIDES.get("ollama")
        # Whatever the operator has pulled is the real catalogue, so discovery
        # carries this one entirely.
        self.curated = _extra(self.id)


class LMStudioProvider(LocalOpenAIProvider):
    id = "lmstudio"
    label = "LM Studio"
    docs_url = "https://lmstudio.ai"
    key_names = ("LMSTUDIO_BASE_URL",)
    model_cap = 40

    def __init__(self) -> None:
        super().__init__()
        self.env_key = None
        self.env_base_url = settings.LMSTUDIO_BASE_URL or "http://localhost:1234/v1"
        self.preferred = settings.MODEL_OVERRIDES.get("lmstudio")
        self.curated = _extra(self.id)


class LlamaCppProvider(LocalOpenAIProvider):
    id = "llamacpp"
    label = "llama.cpp server"
    docs_url = "https://github.com/ggml-org/llama.cpp"
    key_names = ("LLAMACPP_BASE_URL",)
    model_cap = 12

    def __init__(self) -> None:
        super().__init__()
        self.env_key = None
        self.env_base_url = settings.LLAMACPP_BASE_URL or "http://localhost:8080/v1"
        self.preferred = settings.MODEL_OVERRIDES.get("llamacpp")
        self.curated = _extra(self.id)


class VLLMProvider(LocalOpenAIProvider):
    id = "vllm"
    label = "vLLM"
    docs_url = "https://docs.vllm.ai"
    key_names = ("VLLM_BASE_URL",)
    model_cap = 12

    def __init__(self) -> None:
        super().__init__()
        self.env_key = settings.VLLM_API_KEY
        self.env_base_url = settings.VLLM_BASE_URL or "http://localhost:8000/v1"
        self.preferred = settings.MODEL_OVERRIDES.get("vllm")
        self.curated = _extra(self.id)


# ------------------------------------------------------------ native SDKs


class GeminiProvider(BaseProvider):
    id = "gemini"
    label = "Google Gemini"
    docs_url = "https://aistudio.google.com/apikey"
    key_names = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    max_attempts = 3
    model_cap = 18

    def __init__(self) -> None:
        self.env_key = settings.GOOGLE_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("gemini")
        self.curated = [
            ModelSpec("gemini-2.5-flash", "Gemini 2.5 Flash",
                      note="Balanced default, generous free tier"),
            ModelSpec("gemini-2.5-flash-lite", "Gemini 2.5 Flash Lite",
                      note="Fastest and cheapest", cheap=True),
            ModelSpec("gemini-2.5-pro", "Gemini 2.5 Pro", note="Deepest reasoning"),
        ] + _extra(self.id)
        # One SDK client per credential, never one per adapter, so a visitor's
        # key can never be reused for someone else's request.
        self._clients: dict[str, Any] = {}
        self._labels: dict[str, str] = {}

    def is_available(self) -> bool:
        return bool(self.api_key)

    @staticmethod
    def _normalise(model: str) -> str:
        model = (model or "").strip()
        return model[len("models/") :] if model.startswith("models/") else model

    def _discover(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            ids: list[str] = []
            for item in self._client().models.list():
                actions = getattr(item, "supported_actions", None) or []
                if "generateContent" not in actions:
                    continue
                name = self._normalise(getattr(item, "name", "") or "")
                if not name:
                    continue
                ids.append(name)
                if display := (getattr(item, "display_name", "") or ""):
                    self._labels[name] = display
            return ids
        except Exception:  # noqa: BLE001 - discovery is best effort
            return []

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        live = _cache_get(self.id, self.api_key)
        if live is None:
            live = self._discover()
            _cache_put(self.id, live, self.api_key)
        merged = _merge_catalogue(list(self.curated), live, cap=self.model_cap)
        merged = [
            ModelSpec(spec.id, self._labels.get(spec.id, spec.label), spec.note, spec.cheap)
            for spec in merged
        ]
        return _promote(merged, self.preferred, live)

    def _client(self):
        api_key = self.api_key
        if not api_key:
            raise LLMError(self.unavailable_reason(), provider=self.id, hint="Missing API key.")
        tag = keyring.fingerprint(api_key)
        if (cached := self._clients.get(tag)) is not None:
            return cached
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "The google-genai package is not installed on the server.",
                provider=self.id,
            ) from exc
        if len(self._clients) >= 16:
            self._clients.clear()
        client = genai.Client(api_key=api_key)
        self._clients[tag] = client
        return client

    def verify(self) -> list[str]:
        if not self.api_key:
            raise LLMError(self.unavailable_reason(), provider=self.id)
        try:
            ids = [
                name
                for item in self._client().models.list()
                if (name := self._normalise(getattr(item, "name", "") or ""))
                and "generateContent" in (getattr(item, "supported_actions", None) or [])
            ]
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            raise LLMError(
                f"{self.label} did not accept that key.",
                provider=self.id,
                hint=_hint_for(str(exc), self.label),
            ) from exc
        _cache_put(self.id, ids, self.api_key)
        return ids

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> Completion:
        from google.genai import types

        client = self._client()
        model_name = self._normalise(model)
        last_error: Exception | None = None

        for attempt in range(self.max_attempts):
            try:
                config_kwargs: dict[str, Any] = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "system_instruction": _compose_system(system),
                }
                # Summarising a news cluster needs no hidden reasoning, and
                # paying for it would double the cost of every call. Older SDKs
                # have no thinking config at all, which is equally fine.
                with suppress(Exception):
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                if json_mode:
                    config_kwargs["response_mime_type"] = "application/json"

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = sanitise_output((getattr(response, "text", "") or "").strip())
                if not text:
                    raise RuntimeError(
                        "Gemini returned an empty response. The prompt may have been "
                        "blocked by a safety filter, or the token budget was too small."
                    )
                meta = getattr(response, "usage_metadata", None)
                return Completion(
                    text=text,
                    usage=Usage(
                        provider=self.id,
                        model=model,
                        input_tokens=int(getattr(meta, "prompt_token_count", 0) or 0),
                        output_tokens=int(getattr(meta, "candidates_token_count", 0) or 0),
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - normalised below
                last_error = exc
                message = str(exc)
                if json_mode and "response_mime_type" in message:
                    json_mode = False
                    continue
                if _is_transient(message) and attempt < self.max_attempts - 1:
                    time.sleep(_retry_delay(message, attempt))
                    continue
                break

        message = str(last_error) if last_error else "Unknown provider error."
        raise LLMError(
            f"Gemini call failed: {message[:400]}",
            provider=self.id,
            model=model,
            retryable=_is_transient(message),
            hint=_hint_for(message, self.label),
        )


class AnthropicProvider(BaseProvider):
    """Claude, through the official Anthropic SDK.

    Deliberately not routed through an OpenAI compatibility shim: the native
    SDK is the supported path, and the system prompt is a top level field here
    rather than a message, which the shim would have to fake.
    """

    id = "anthropic"
    label = "Anthropic Claude"
    docs_url = "https://console.anthropic.com/settings/keys"
    key_names = ("ANTHROPIC_API_KEY",)
    max_attempts = 3
    model_cap = 12

    def __init__(self) -> None:
        self.env_key = settings.ANTHROPIC_API_KEY
        self.preferred = settings.MODEL_OVERRIDES.get("anthropic")
        self.curated = [
            ModelSpec("claude-sonnet-5", "Claude Sonnet 5",
                      note="Best balance of quality and cost for digests"),
            ModelSpec("claude-opus-5", "Claude Opus 5", note="Highest quality"),
            ModelSpec("claude-haiku-4-5", "Claude Haiku 4.5",
                      note="Cheapest, good for classification", cheap=True),
        ] + _extra(self.id)
        self._clients: dict[str, Any] = {}

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        api_key = self.api_key
        if not api_key:
            raise LLMError(self.unavailable_reason(), provider=self.id, hint="Missing API key.")
        tag = keyring.fingerprint(api_key)
        if (cached := self._clients.get(tag)) is not None:
            return cached
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMError(
                "The anthropic package is not installed on the server.", provider=self.id
            ) from exc
        if len(self._clients) >= 16:
            self._clients.clear()
        client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self._clients[tag] = client
        return client

    def _discover(self) -> list[str]:
        if not self.api_key:
            return []
        try:
            return [
                item.id for item in self._client().models.list() if getattr(item, "id", None)
            ]
        except Exception:  # noqa: BLE001 - discovery is best effort
            return []

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        live = _cache_get(self.id, self.api_key)
        if live is None:
            live = self._discover()
            _cache_put(self.id, live, self.api_key)
        return _promote(
            _merge_catalogue(list(self.curated), live, cap=self.model_cap),
            self.preferred,
            live,
        )

    def verify(self) -> list[str]:
        if not self.api_key:
            raise LLMError(self.unavailable_reason(), provider=self.id)
        try:
            ids = [
                item.id for item in self._client().models.list() if getattr(item, "id", None)
            ]
        except Exception as exc:  # noqa: BLE001 - reported to the caller
            raise LLMError(
                f"{self.label} did not accept that key.",
                provider=self.id,
                hint=_hint_for(str(exc), self.label),
            ) from exc
        _cache_put(self.id, ids, self.api_key)
        return ids

    def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> Completion:
        client = self._client()
        system_text = _compose_system(system)
        if json_mode:
            system_text += (
                "\n\nReturn only a single valid JSON value. No prose, no markdown fences."
            )

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=system_text,
                    messages=[{"role": "user", "content": prompt}],
                )
                # A safety classifier may decline the request with a 200, so the
                # stop reason has to be read before the content is trusted.
                if getattr(response, "stop_reason", "") == "refusal":
                    details = getattr(response, "stop_details", None)
                    category = getattr(details, "category", "") or "unspecified"
                    raise LLMError(
                        f"Claude declined this request (category: {category}).",
                        provider=self.id,
                        model=model,
                        hint="Rephrase the prompt, or use a different provider for this stage.",
                    )
                text = "".join(
                    block.text
                    for block in (getattr(response, "content", None) or [])
                    if getattr(block, "type", "") == "text"
                ).strip()
                if not text:
                    raise RuntimeError(
                        "Claude returned no text block "
                        f"(stop reason: {getattr(response, 'stop_reason', 'none')})."
                    )
                usage = getattr(response, "usage", None)
                return Completion(
                    text=sanitise_output(text),
                    usage=Usage(
                        provider=self.id,
                        model=model,
                        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                    ),
                )
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalised below
                last_error = exc
                message = str(exc)
                if _is_transient(message) and attempt < self.max_attempts - 1:
                    time.sleep(_retry_delay(message, attempt))
                    continue
                break

        message = str(last_error) if last_error else "Unknown provider error."
        raise LLMError(
            f"Anthropic call failed: {message[:400]}",
            provider=self.id,
            model=model,
            retryable=_is_transient(message),
            hint=_hint_for(message, self.label),
        )
