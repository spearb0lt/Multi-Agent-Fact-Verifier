"""The only way an agent reaches a model.

Agents never touch the provider registry directly. Everything goes through
here, because four things have to happen around every single call and none of
them can be left to the caller to remember:

* the budget is consulted before the call and charged after it,
* the tokens are attributed to the role and the step that spent them,
* the choice of model follows the run's policy rather than the agent's whim,
* a transient provider failure is retried, and a permanent one is turned into
  something the agent can read.

The last point is worth being specific about. Free tier providers rate limit
aggressively and the retry logic in the vendored provider layer already handles
the per-call case. What it cannot do is decide that a run has now been rate
limited so many times that continuing is pointless. That judgement needs the
run's budget, which is why it lives here.
"""
from __future__ import annotations

import threading
from typing import Any

from ..core.llm import registry as llm
from ..core.llm.base import Completion, LLMError
from ..core.ops.meter import Entry
from ..core.util import truncate
from .contracts import BudgetExceeded, EventKind, ProviderExhausted
from .pacer import pacer

# Roughly four characters to a token for English prose. Only used to refuse a
# call that obviously will not fit the remaining ceiling, so being approximate
# is fine and being cheap to compute is not negotiable.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN)


class RunLLM:
    """A model client bound to one run, one policy and one budget."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        # Parallel researchers share this client, and each of them charges the
        # budget and writes a usage row. Without the lock two concurrent
        # charges read the same total and one of them is lost, which shows up
        # as a run that quietly overspends its ceiling.
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- calls

    def complete(
        self,
        prompt: str,
        *,
        role: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        purpose: str = "",
    ) -> str:
        completion = self._call(
            prompt,
            role=role,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose or role.lower(),
            json_mode=False,
        )
        return completion.text

    def complete_json(
        self,
        prompt: str,
        *,
        role: str,
        system: str = "",
        schema_hint: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        purpose: str = "",
        expect_list: bool = False,
        default: Any = None,
    ) -> Any:
        """Ask for JSON and parse it leniently.

        The schema is described in the prompt as well as requested through the
        provider's JSON mode, because the smaller models this system is built
        to run on honour the prompt more reliably than the mode.
        """
        from ..core.llm.base import coerce_json
        from ..core.llm.registry import unwrap_list

        body = prompt
        if schema_hint:
            body = (
                f"{prompt}\n\n"
                f"Reply with JSON matching this shape and nothing else:\n{schema_hint}"
            )

        completion = self._call(
            body,
            role=role,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose or role.lower(),
            json_mode=True,
        )

        try:
            parsed = coerce_json(completion.text)
        except Exception:
            parsed = None

        if parsed is None:
            self.ctx.bus.emit(
                EventKind.LOG,
                f"{role} returned something that was not JSON, so the default was used.",
                agent=role,
                level="warning",
                sample=truncate(completion.text, 300),
            )
            return [] if expect_list else (default if default is not None else {})

        return unwrap_list(parsed) if expect_list else parsed

    def supports_native_tools(self, role: str) -> bool:
        choice = self.ctx.policy.resolve(role, self.ctx)
        provider = llm.provider_map().get(choice.provider)
        return bool(provider and getattr(provider, "supports_tools", False))

    def complete_with_tools(
        self,
        prompt: str,
        *,
        role: str,
        system: str = "",
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> Completion:
        """One turn with tool schemas declared to the provider.

        Used where the provider has a real function calling API. That is not
        an optimisation: models trained for native tool use will emit a native
        call whether or not tools were declared, and a provider that was not
        told about them rejects the request outright.
        """
        return self._call(
            prompt,
            role=role,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            purpose=purpose or f"{role.lower()}.react",
            json_mode=False,
            tools=tools,
        )

    # ------------------------------------------------------------- internals

    def _call(
        self,
        prompt: str,
        *,
        role: str,
        system: str,
        temperature: float,
        max_tokens: int,
        purpose: str,
        json_mode: bool,
        tools: list[dict[str, Any]] | None = None,
    ) -> Completion:
        ctx = self.ctx
        choice = ctx.policy.resolve(role, ctx)

        estimated_in = estimate_tokens(prompt) + estimate_tokens(system)
        ctx.budget.check_headroom(tokens=estimated_in + max_tokens)

        provider = llm.provider_map()[choice.provider]

        # Wait for a slot before spending one. A free tier refuses a burst that
        # it would have accepted spread over a minute, so the delay here buys
        # the run far more than it costs.
        # The reservation covers the prompt and the whole reply allowance,
        # which is what the provider counts against a tokens per minute limit.
        waited = pacer.wait(choice.provider, tokens=estimated_in + max_tokens)
        if waited > 2.0:
            ctx.bus.emit(
                EventKind.LOG,
                f"Waited {waited:.0f}s for a {choice.provider} rate limit slot.",
                agent=role,
                level="info",
                provider=choice.provider,
                waited_seconds=round(waited, 1),
            )

        call_kwargs: dict[str, Any] = {
            "model": choice.model,
            "system": system or None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        }
        # Only providers that advertise it are handed tool schemas; the rest
        # would raise a TypeError on an argument their adapter never declared.
        if tools and getattr(provider, "supports_tools", False):
            call_kwargs["tools"] = tools

        try:
            completion = provider.generate(prompt, **call_kwargs)
        except LLMError as exc:
            # The call cost something even when it failed on the provider's
            # side, but nothing reliable is reported back, so only the attempt
            # is counted. Leaving it uncounted would let a run retry forever
            # against a rate limited provider without the ceiling ever binding.
            with self._lock:
                ctx.budget.charge_llm(tokens_in=estimated_in, tokens_out=0, usd=0.0)
            ctx.bus.emit(
                EventKind.LOG,
                f"{choice.label} refused the call for {role}: {exc}",
                agent=role,
                level="error",
                provider=choice.provider,
                model=choice.model,
                hint=getattr(exc, "hint", ""),
            )
            if is_exhausted(exc):
                raise ProviderExhausted(
                    choice.provider,
                    f"{choice.label} is out of quota or rate limiting harder than "
                    f"retrying can absorb. The run kept everything it had.",
                    hint=getattr(exc, "hint", "")
                    or "Wait for the quota window to reset and resume, or resume with "
                    "a different provider.",
                ) from exc
            raise

        usage = completion.usage
        entry = Entry(
            provider=usage.provider or choice.provider,
            model=usage.model or choice.model,
            kind="llm",
            operation=purpose,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        )
        cost = entry.cost_usd

        # Replace the reservation with what was really consumed, so the next
        # caller is paced against the truth rather than the allowance.
        pacer.record_actual(choice.provider, usage.input_tokens + usage.output_tokens)

        with self._lock:
            ctx.budget.charge_llm(
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                usd=cost or 0.0,
            )
            ctx.store.record_usage(
                ctx.run_id,
                step_seq=ctx.seq,
                agent=role,
                purpose=purpose,
                provider=entry.provider,
                model=entry.model,
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                cost_usd=cost or 0.0,
                priced=entry.priced,
            )

        for reading in ctx.budget.newly_warned():
            ctx.bus.emit(
                EventKind.BUDGET_WARN,
                f"{int(reading.fraction * 100)} percent of the run's {reading.name} "
                f"ceiling is spent.",
                level="warning",
                limit=reading.name,
                used=reading.used,
                ceiling=reading.limit,
            )

        return completion

    # ------------------------------------------------------------- utilities

    def guard(self) -> None:
        """Raise if the run can no longer afford to continue."""
        self.ctx.budget.check()


# What a provider says when the account is out of room rather than briefly busy.
_EXHAUSTED_MARKERS = (
    "resource_exhausted", "quota", "rate limit", "rate_limit", "429",
    "too many requests", "insufficient_quota", "billing", "credit",
)


def is_exhausted(exc: LLMError) -> bool:
    """Whether this failure is 'come back later' rather than 'this is broken'.

    Checked after the provider layer has already retried, so reaching here at
    all means its own backoff did not clear the problem.
    """
    if getattr(exc, "status", None) in {429, 402}:
        return True
    text = f"{exc} {getattr(exc, 'hint', '')}".lower()
    return any(marker in text for marker in _EXHAUSTED_MARKERS)


def summarise_error(exc: Exception) -> str:
    """One line an agent or the UI can read, out of any provider failure."""
    if isinstance(exc, BudgetExceeded):
        return exc.detail
    if isinstance(exc, LLMError):
        hint = getattr(exc, "hint", "")
        return f"{exc} {hint}".strip()
    return f"{type(exc).__name__}: {truncate(str(exc), 240)}"
