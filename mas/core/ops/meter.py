"""Records what each model call consumed, without threading it through the code.

Token counts are needed in exactly one place, the run's cost report, but they
are produced in a dozen places scattered across the pipeline. Passing a usage
object through every function signature would distort all of them, so calls
append to a context bound collector instead and the runner drains it at the end
of each stage.

The collector is a context variable, so a background thread or a second
concurrent request gets its own and the totals never mix.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# Published prices per million tokens, used only to show an estimate. A model
# that is not listed is reported with a zero cost and a note, rather than a
# guessed number that would read as authoritative.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # (input, output) USD per million tokens, from each provider's own pricing
    # page, checked 2026-09-21. A model is listed only when its price is
    # published: anything missing reports as unpriced rather than as free,
    # because a zero that means "nobody entered a number" reads exactly like a
    # zero that means "this cost nothing".
    #
    # Two known simplifications. Gemini 2.5 Pro and xAI Grok charge more above
    # a context threshold and the lower band is used here, so a very long
    # prompt is under reported. Perplexity also bills per request on top of
    # tokens, which this cannot express at all.

    # Anthropic, platform.claude.com/docs/en/about-claude/pricing
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),

    # Google, ai.google.dev/gemini-api/docs/pricing. Free of charge on the free
    # tier; these are the paid rates.
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),

    # Groq, console.groq.com/docs/models
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "qwen/qwen3.8-27b": (0.80, 4.00),

    # Cloudflare Workers AI, which bills in Neurons at $0.011 per 1,000 and
    # publishes the USD equivalent alongside. 10,000 Neurons a day are free.
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": (0.293, 2.253),
    "@cf/openai/gpt-oss-120b": (0.350, 0.750),
    "@cf/meta/llama-3.1-8b-instruct-fp8": (0.152, 0.287),
    "@cf/meta/llama-3.2-3b-instruct": (0.051, 0.335),

    # OpenAI, developers.openai.com/api/docs/pricing
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),

    # Others, headline chat model
    "Meta-Llama-3.3-70B-Instruct": (0.60, 1.20),   # SambaNova
    "grok-4.6": (2.00, 6.00),                       # xAI, under 200k context
    "sonar": (1.00, 1.00),                          # Perplexity, plus per request
    "sonar-pro": (3.00, 15.00),                     # Perplexity, plus per request

    # Embeddings. Output is always zero: nothing is generated.
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
    "gemini-embedding-2": (0.20, 0.0),
    "@cf/baai/bge-base-en-v1.5": (0.067, 0.0),
    "@cf/baai/bge-m3": (0.012, 0.0),
    "jina-embeddings-v5-text-small": (0.05, 0.0),
    "jina-embeddings-v5-text-nano": (0.02, 0.0),
    "voyage-4": (0.06, 0.0),
    "voyage-4-lite": (0.02, 0.0),
    "voyage-4-large": (0.12, 0.0),
    "embed-v4.0": (0.12, 0.0),

    # Runs on this machine, so it genuinely costs nothing per token.
    "all-MiniLM-L6-v2-onnx": (0.0, 0.0),
    "hashed-ngrams-1024": (0.0, 0.0),
}


@dataclass
class Entry:
    provider: str
    model: str
    kind: str = "llm"  # llm or embed
    operation: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    units: int = 0  # items embedded, for providers that do not report tokens

    @property
    def priced(self) -> bool:
        return self.model in PRICES_PER_MTOK

    @property
    def cost_usd(self) -> float | None:
        """What this call cost, or None when the model has no published price.

        Returning None rather than zero matters. A model that is genuinely free
        and a model nobody has entered a price for both rendered as $0.00, so
        a page showing the total spend read as "you have spent nothing" when
        the honest answer was "this is not being measured".
        """
        price = PRICES_PER_MTOK.get(self.model)
        if price is None:
            return None
        prompt_price, completion_price = price
        return (
            self.input_tokens * prompt_price + self.output_tokens * completion_price
        ) / 1_000_000

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "operation": self.operation,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "units": self.units,
            "priced": self.priced,
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
        }


@dataclass
class Collector:
    entries: list[Entry] = field(default_factory=list)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)

    def drain(self) -> list[Entry]:
        out = self.entries
        self.entries = []
        return out

    def totals(self) -> dict[str, Any]:
        priced = [e for e in self.entries if e.priced]
        return {
            "calls": len(self.entries),
            "input_tokens": sum(e.input_tokens for e in self.entries),
            "output_tokens": sum(e.output_tokens for e in self.entries),
            "units": sum(e.units for e in self.entries),
            "cost_usd": round(sum(e.cost_usd or 0.0 for e in priced), 6),
            # Counted separately so a total can never be read as complete when
            # part of the spend is simply not measured.
            "unpriced_calls": len(self.entries) - len(priced),
        }


_collector: ContextVar[Collector | None] = ContextVar("usage_collector", default=None)


def start() -> Collector:
    collector = Collector()
    _collector.set(collector)
    return collector


def active() -> Collector | None:
    return _collector.get()


def record(
    *,
    provider: str,
    model: str,
    kind: str = "llm",
    operation: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    units: int = 0,
) -> None:
    """Note one call. A no-op when nothing is collecting, so callers need no guard."""
    collector = _collector.get()
    if collector is None:
        return
    collector.add(
        Entry(
            provider=provider,
            model=model,
            kind=kind,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            units=units,
        )
    )


def stop() -> None:
    _collector.set(None)
