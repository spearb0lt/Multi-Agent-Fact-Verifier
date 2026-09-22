"""Keeping a run inside a provider's free tier allowance.

Three researchers each taking up to eight reason and act turns is two dozen
model calls issued as fast as the network allows. That burst trips every free
tier in the registry within seconds, failing a run that had plenty of quota
left and simply asked for it too quickly.

So calls are paced rather than throttled after the fact. A caller waits its
turn before the request goes out, which turns a failure into a delay. That is
the right trade here: the whole system is built to run on free tiers, and a run
that takes four minutes instead of two is a much better outcome than one that
stops at the first researcher.

Both limits are enforced, and the second one is the one that actually bites.
Requests per minute is the limit everyone quotes, but tokens per minute is what
a research agent hits first: Groq allows 8,000 tokens a minute on its small
models, and one researcher's loop exhausts that in three turns while using
three of its twenty five requests. A pacer that counted only requests would
look like it was working and would not be.

Tokens are reserved before the call, using the prompt estimate plus the whole
reply allowance, and corrected afterwards to what the provider actually
counted. Reserving on the way in is what makes it safe with several researchers
in flight; correcting on the way out is what stops it being so conservative
that the run spends its time asleep.

The limits are per provider and per process, and are set a little under the
published allowances, because the published number is enforced on the
provider's clock rather than this one.
"""
from __future__ import annotations

import threading
import time
from collections import deque

from ..core import settings

# Requests per minute, set a little under each provider's published free tier.
# A provider absent from here is not paced at all, which is right for a local
# model server and for any account paying for capacity.
DEFAULT_RPM: dict[str, int] = {
    # Gemini's free tier is the tightest of the lot on requests, and three
    # researchers in parallel exhausted it in under a minute at 12. It is also
    # limited per day, which no amount of pacing helps with, so a run that
    # stops on Gemini is usually told to continue somewhere else.
    "gemini": 8,
    "groq": 25,
    "cloudflare": 250,
    "huggingface": 45,
    "cerebras": 25,
    "sambanova": 25,
    "mistral": 45,
    "openrouter": 15,
    "perplexity": 45,
}

# Tokens per minute. This is the limit that actually binds on a free tier, and
# missing it was the first thing that stopped a working run: Groq allows 8,000
# tokens a minute on the small models, which a single researcher's reason and
# act loop exhausts in three turns while using only three of its 25 requests.
# Pacing requests alone therefore looks like it is working and is not.
#
# Set under the published number, because the count here is an estimate of the
# prompt plus a reservation for the reply, and the provider counts what it
# actually processed.
DEFAULT_TPM: dict[str, int] = {
    "groq": 7_000,
    "gemini": 200_000,
    "cerebras": 60_000,
    "sambanova": 40_000,
    "mistral": 400_000,
    "openrouter": 40_000,
    "huggingface": 40_000,
    "cloudflare": 0,  # Billed in neurons per day rather than tokens per minute.
}

# A local model server has no quota, only a queue, so pacing it just makes it
# slower for no benefit.
UNPACED = {"ollama", "lmstudio", "llamacpp", "vllm"}


def _configured(defaults: dict[str, int], variable: str) -> dict[str, int]:
    """Env overrides, written as PROVIDER_RPM=gemini:10,groq:30."""
    limits = dict(defaults)
    raw = settings.env(variable, default="") or ""
    for part in raw.split(","):
        if ":" not in part:
            continue
        name, _, value = part.partition(":")
        try:
            limits[name.strip().lower()] = max(0, int(value.strip()))
        except ValueError:
            continue
    return limits


class Pacer:
    """Sliding window limiters, per provider, on both requests and tokens."""

    # The width of the window. An attribute rather than a literal so a test can
    # shrink it and exercise the blocking path without sleeping for a minute.
    WINDOW_SECONDS = 60.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, deque[float]] = {}
        # (timestamp, tokens) pairs, so the token window can be summed and aged
        # out the same way the request window is.
        self._tokens: dict[str, deque[tuple[float, int]]] = {}
        self._rpm = _configured(DEFAULT_RPM, "PROVIDER_RPM")
        self._tpm = _configured(DEFAULT_TPM, "PROVIDER_TPM")
        self.waited_seconds = 0.0
        self.waits = 0

    def limit_for(self, provider: str) -> int:
        if provider in UNPACED:
            return 0
        return int(self._rpm.get(provider, 0))

    def token_limit_for(self, provider: str) -> int:
        if provider in UNPACED:
            return 0
        return int(self._tpm.get(provider, 0))

    def _prune(self, provider: str, now: float) -> tuple[deque, deque, int]:
        calls = self._calls.setdefault(provider, deque())
        while calls and now - calls[0] >= self.WINDOW_SECONDS:
            calls.popleft()
        tokens = self._tokens.setdefault(provider, deque())
        while tokens and now - tokens[0][0] >= self.WINDOW_SECONDS:
            tokens.popleft()
        return calls, tokens, sum(count for _, count in tokens)

    def wait(self, provider: str, *, tokens: int = 0, timeout: float = 180.0) -> float:
        """Block until this call fits both windows. Returns how long it waited.

        The windows are inspected under the lock and the sleep happens outside
        it, so a thread waiting on Groq does not also stop a thread calling
        Gemini. The reservation is made on the way in rather than the way out,
        because two threads that both checked before either recorded would
        both pass a limit only one of them fits under.
        """
        rpm = self.limit_for(provider)
        tpm = self.token_limit_for(provider)
        if rpm <= 0 and tpm <= 0:
            return 0.0

        deadline = time.monotonic() + timeout
        slept = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                calls, token_window, used = self._prune(provider, now)

                over_requests = rpm > 0 and len(calls) >= rpm
                over_tokens = tpm > 0 and tokens > 0 and used + tokens > tpm

                if not over_requests and not over_tokens:
                    calls.append(now)
                    if tokens:
                        token_window.append((now, tokens))
                    return slept

                waits = []
                if over_requests and calls:
                    waits.append(self.WINDOW_SECONDS - (now - calls[0]))
                if over_tokens and token_window:
                    # Waiting for the oldest block of tokens to age out frees
                    # the most room for the least delay.
                    waits.append(self.WINDOW_SECONDS - (now - token_window[0][0]))
                wait_for = max(0.1, min(waits) + 0.1) if waits else 1.0

            if time.monotonic() + wait_for > deadline:
                # Give up on pacing rather than on the call: the provider may
                # still accept it, and its own error is something the caller
                # already knows how to turn into a recoverable pause.
                with self._lock:
                    now = time.monotonic()
                    self._calls.setdefault(provider, deque()).append(now)
                    if tokens:
                        self._tokens.setdefault(provider, deque()).append((now, tokens))
                return slept

            time.sleep(wait_for)
            slept += wait_for
            with self._lock:
                self.waited_seconds += wait_for
                self.waits += 1

    def record_actual(self, provider: str, tokens: int) -> None:
        """Correct the token window once the provider reports what it counted.

        The reservation made before the call is an estimate of the prompt plus
        the whole reply allowance, and the reply is usually far shorter than
        the allowance. Leaving the estimate in place makes the pacer far more
        conservative than the provider, which shows up as a run that spends
        most of its wall clock asleep.
        """
        tpm = self.token_limit_for(provider)
        if tpm <= 0 or tokens <= 0:
            return
        with self._lock:
            window = self._tokens.setdefault(provider, deque())
            if window:
                stamp, _ = window[-1]
                window[-1] = (stamp, tokens)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            now = time.monotonic()
            usage = {}
            for provider in set(self._calls) | set(self._tokens):
                calls = [t for t in self._calls.get(provider, ()) if now - t < self.WINDOW_SECONDS]
                tokens = sum(
                    count for stamp, count in self._tokens.get(provider, ())
                    if now - stamp < self.WINDOW_SECONDS
                )
                usage[provider] = {"requests": len(calls), "tokens": tokens}
            return {
                "limits": {k: v for k, v in self._rpm.items() if v},
                "token_limits": {k: v for k, v in self._tpm.items() if v},
                "in_window": usage,
                "total_wait_seconds": round(self.waited_seconds, 1),
                "waits": self.waits,
            }


pacer = Pacer()
