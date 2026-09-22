"""Provider registry and the single entry point every stage calls.

Routing is strict: the provider and model chosen in the UI are the source of
truth. There is no silent cross provider fallback, so a quota error from the
selected provider surfaces with an actionable hint instead of quietly costing
money somewhere else. Transient errors are retried on the same provider only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ops import meter
from . import keyring
from .base import BaseProvider, Completion, LLMError, coerce_json
from .providers import (
    AnthropicProvider,
    CerebrasProvider,
    CloudflareProvider,
    DeepSeekProvider,
    FireworksProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    LlamaCppProvider,
    LMStudioProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    PerplexityProvider,
    SambaNovaProvider,
    TogetherProvider,
    VLLMProvider,
    XAIProvider,
)

# Display order in the picker. Providers with a usable free tier come first,
# because the common case is someone trying this out with no budget.
_ORDER: tuple[type[BaseProvider], ...] = (
    GeminiProvider,
    GroqProvider,
    CloudflareProvider,
    OpenRouterProvider,
    CerebrasProvider,
    SambaNovaProvider,
    MistralProvider,
    DeepSeekProvider,
    TogetherProvider,
    FireworksProvider,
    AnthropicProvider,
    OpenAIProvider,
    XAIProvider,
    PerplexityProvider,
    HuggingFaceProvider,
    OllamaProvider,
    LMStudioProvider,
    LlamaCppProvider,
    VLLMProvider,
)

_registry: dict[str, BaseProvider] | None = None


def provider_map() -> dict[str, BaseProvider]:
    global _registry
    if _registry is None:
        built: dict[str, BaseProvider] = {}
        for cls in _ORDER:
            instance = cls()
            built[instance.id] = instance
        _registry = built
    return _registry


def reset_registry() -> None:
    """Rebuild adapters, used after the environment changes in tests."""
    global _registry
    _registry = None


def all_status() -> list[dict[str, Any]]:
    """Every provider's status, gathered concurrently.

    Each provider's status costs a network round trip: a liveness probe for a
    local server, a model listing for a hosted one. Nineteen of those in
    sequence made a cold /api/config take over twenty seconds. They are
    independent, so they run together and the page waits for the slowest rather
    than the sum.
    """
    from ..util import map_concurrently

    return [
        status.as_dict()
        for status in map_concurrently(
            lambda item: item.status(), list(provider_map().values())
        )
    ]


def server_providers() -> list[str]:
    return [p.id for p in provider_map().values() if p.has_server_key()]


def client_providers() -> list[str]:
    supplied = keyring.supplied()
    return [p.id for p in provider_map().values() if p.id in supplied]


def available_providers() -> list[BaseProvider]:
    return [p for p in provider_map().values() if p.is_available()]


def any_available() -> bool:
    return bool(available_providers())


def default_selection() -> tuple[str, str]:
    for provider in available_providers():
        if model := provider.default_model():
            return provider.id, model
    return "", ""


@dataclass
class Selection:
    provider: BaseProvider
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider.label} / {self.model}"


def resolve(
    provider_id: str | None = None,
    model: str | None = None,
    *,
    cheap: bool = False,
) -> Selection:
    """Turn a client supplied provider and model into a usable Selection.

    `cheap` asks for the provider's small model instead of its default. The
    classification stage runs once per article and the summarisation stage once
    per event, so letting the caller ask for the cheaper model is the main
    lever on what a run costs.
    """
    providers = provider_map()
    usable = available_providers()

    if not usable:
        raise LLMError(
            "No AI provider is available for this request.",
            hint=(
                "Open Settings and paste an API key for Google Gemini, Groq, "
                "Cloudflare Workers AI or any other listed provider. The key stays "
                "in your browser and is sent only with your own requests. Running "
                "locally, Ollama or LM Studio needs no key at all."
            ),
        )

    provider_id = (provider_id or "").strip().lower()
    model = (model or "").strip()

    if provider_id:
        provider = providers.get(provider_id)
        if provider is None:
            raise LLMError(
                f"Unknown provider '{provider_id}'.",
                hint=f"Available providers: {', '.join(p.id for p in usable)}.",
            )
        if not provider.is_available():
            raise LLMError(
                f"{provider.label} is not usable in this deployment.",
                provider=provider.id,
                hint=provider.unavailable_reason(),
            )
    else:
        provider = usable[0]

    if not model:
        model = provider.cheap_model() if cheap else provider.default_model()

    if not model:
        raise LLMError(
            f"No model available for {provider.label}.",
            provider=provider.id,
            hint="Set a model id in the picker or through the environment.",
        )

    return Selection(provider=provider, model=model)


def generate(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    cheap: bool = False,
    operation: str = "",
) -> str:
    selection = resolve(provider, model, cheap=cheap)
    completion = selection.provider.generate(
        prompt,
        model=selection.model,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _meter(completion, operation)
    return completion.text


def _meter(completion: Completion, operation: str) -> None:
    meter.record(
        provider=completion.usage.provider,
        model=completion.usage.model,
        kind="llm",
        operation=operation,
        input_tokens=completion.usage.input_tokens,
        output_tokens=completion.usage.output_tokens,
    )


# Keys a model might reasonably choose when asked to wrap a list in an object.
_LIST_KEYS = (
    "items", "rows", "results", "data", "list", "entries", "records",
    "articles", "stories", "events", "themes", "entities", "bullets", "output",
)


def unwrap_list(payload: Any) -> list:
    """Pull a list out of whatever shape the model actually returned.

    A JSON mode that only permits a top level object is common, so a stage that
    wants an array has to ask for one wrapped in an object. Models then vary in
    what they name the key, and a smaller model sometimes returns a single bare
    object instead of a list of one. All three cases end up as a list here.
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    for key in _LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return value

    lists = [v for v in payload.values() if isinstance(v, list)]
    if len(lists) == 1:
        return lists[0]

    # A lone object that is itself one row, which is what a strict object mode
    # produces when the model gives up on the array.
    if any(not isinstance(v, (list, dict)) for v in payload.values()):
        return [payload]
    return []


def generate_json(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    schema_hint: str = "",
    expect_list: bool = False,
    cheap: bool = False,
    operation: str = "",
) -> Any:
    """Ask for JSON and parse it defensively.

    Providers differ in how well they honour a JSON mode, so the schema is
    always described in the prompt as well and the response is parsed leniently.

    Set expect_list when the stage wants an array. Several providers constrain
    JSON mode to a single top level object, so asking for a bare array there
    yields one object holding only the first row. The request is therefore made
    for an object wrapping the array, and the array is unwrapped on the way out.
    """
    selection = resolve(provider, model, cheap=cheap)
    full_system = system or ""
    if schema_hint:
        full_system = (
            f"{full_system}\n\nReturn JSON matching this shape exactly:\n{schema_hint}"
        ).strip()
    if expect_list:
        full_system = (
            f'{full_system}\n\nReturn a single JSON object with one key, "items", '
            "whose value is the complete array. Put every entry in that one array. "
            "Never return a bare object for a single entry."
        ).strip()

    # A reasoning model occasionally emits its own analysis in place of the
    # JSON, which is a per call accident rather than a property of the model,
    # so an unparseable answer is worth asking for again before giving up.
    last_error: ValueError | None = None
    for attempt in range(3):
        system_for_attempt = full_system
        if attempt:
            system_for_attempt = (
                f"{full_system}\n\nYour previous answer could not be parsed. Reply with "
                "the JSON value and nothing else. No analysis, no explanation, no prose "
                "before or after it."
            )
        completion = selection.provider.generate(
            prompt,
            model=selection.model,
            system=system_for_attempt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        _meter(completion, operation)
        try:
            parsed = coerce_json(completion.text)
        except ValueError as exc:
            last_error = exc
            continue
        return unwrap_list(parsed) if expect_list else parsed

    raise LLMError(
        str(last_error),
        provider=selection.provider.id,
        model=selection.model,
        hint=(
            "The selected model did not return structured output after three "
            "attempts. A larger model usually fixes this."
        ),
    ) from last_error
