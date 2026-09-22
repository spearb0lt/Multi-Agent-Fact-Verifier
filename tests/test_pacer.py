"""Rate pacing, against both limits that a free tier enforces.

The token limit is the one that matters and the one that was missed at first,
so it is tested first. Every case shrinks the window rather than sleeping for a
real minute.
"""
from __future__ import annotations

import threading
import time

from mas.kernel.pacer import Pacer


def fast_pacer(*, rpm: int = 1000, tpm: int = 1000, window: float = 1.0) -> Pacer:
    pacer = Pacer()
    pacer.WINDOW_SECONDS = window
    pacer._rpm = {"p": rpm}
    pacer._tpm = {"p": tpm}
    return pacer


def test_token_limit_blocks_before_the_request_limit_does():
    """The failure that stopped a real run: plenty of requests, no tokens."""
    pacer = fast_pacer(rpm=1000, tpm=1000)

    assert pacer.wait("p", tokens=400, timeout=5) == 0
    assert pacer.wait("p", tokens=400, timeout=5) == 0
    # 800 used, 400 more would exceed 1000, so this one waits for the window.
    waited = pacer.wait("p", tokens=400, timeout=5)
    assert waited > 0


def test_request_limit_still_applies_on_its_own():
    pacer = fast_pacer(rpm=2, tpm=0)
    assert pacer.wait("p", tokens=10, timeout=5) == 0
    assert pacer.wait("p", tokens=10, timeout=5) == 0
    assert pacer.wait("p", tokens=10, timeout=5) > 0


def test_the_window_slides_so_waiting_actually_frees_room():
    pacer = fast_pacer(rpm=1, tpm=0, window=0.3)
    pacer.wait("p", tokens=1)
    start = time.monotonic()
    pacer.wait("p", tokens=1, timeout=5)
    # It waited for the first call to age out, rather than waiting forever.
    assert 0.2 < time.monotonic() - start < 2.0


def test_the_reservation_is_corrected_to_what_was_really_spent():
    pacer = fast_pacer(rpm=1000, tpm=1000)
    # Reserve almost the whole window, the way a 4k reply allowance does.
    pacer.wait("p", tokens=900)
    assert pacer.snapshot()["in_window"]["p"]["tokens"] == 900

    pacer.record_actual("p", 100)
    assert pacer.snapshot()["in_window"]["p"]["tokens"] == 100
    # The freed room is usable straight away, so a short reply does not cost
    # the run a minute of sleeping.
    assert pacer.wait("p", tokens=500, timeout=1) == 0


def test_an_unlimited_provider_is_never_paced():
    pacer = fast_pacer(rpm=0, tpm=0)
    assert pacer.wait("p", tokens=10_000_000, timeout=1) == 0


def test_a_local_model_server_is_never_paced():
    pacer = Pacer()
    assert pacer.limit_for("ollama") == 0
    assert pacer.token_limit_for("lmstudio") == 0
    assert pacer.wait("ollama", tokens=10_000_000, timeout=1) == 0


def test_giving_up_lets_the_call_through_rather_than_failing():
    """Pacing is a courtesy. The provider's own error is the real backstop."""
    pacer = fast_pacer(rpm=1, tpm=0, window=60.0)
    pacer.wait("p", tokens=1)
    # No room for a minute, but the caller only allowed a second.
    assert pacer.wait("p", tokens=1, timeout=0.5) == 0


def test_concurrent_callers_cannot_both_pass_the_same_slot():
    """Two researchers checking at once must not both fit through one opening."""
    pacer = fast_pacer(rpm=1000, tpm=1000, window=30.0)
    passed: list[float] = []
    barrier = threading.Barrier(4)

    def attempt() -> None:
        barrier.wait()
        passed.append(pacer.wait("p", tokens=400, timeout=0.2))

    threads = [threading.Thread(target=attempt) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Reserving under the lock means the window holds what was actually
    # granted, not four simultaneous optimistic reads of an empty one.
    tokens = pacer.snapshot()["in_window"]["p"]["tokens"]
    assert tokens == 4 * 400
    assert len(passed) == 4


def test_limits_are_per_provider():
    pacer = Pacer()
    pacer.WINDOW_SECONDS = 1.0
    pacer._rpm = {"slow": 1, "fast": 100}
    pacer._tpm = {"slow": 0, "fast": 0}
    pacer.wait("slow", tokens=1)
    # A saturated provider does not hold up a different one.
    assert pacer.wait("fast", tokens=1, timeout=1) == 0


def test_env_overrides_are_read(monkeypatch):
    monkeypatch.setenv("PROVIDER_RPM", "gemini:3,groq:99")
    monkeypatch.setenv("PROVIDER_TPM", "groq:1234")
    pacer = Pacer()
    assert pacer.limit_for("gemini") == 3
    assert pacer.limit_for("groq") == 99
    assert pacer.token_limit_for("groq") == 1234
