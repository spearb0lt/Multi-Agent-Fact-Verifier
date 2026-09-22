"""Ceilings on what one run may consume, and the pressure reading below them.

Two jobs, and the second matters as much as the first.

The obvious job is stopping a runaway. A loop that keeps deciding it needs one
more search is the characteristic failure of an agent system, and the only
reliable defence is an accountant outside the loop that counts what actually
happened rather than what the agent intended.

The subtler job is `pressure`. A guard that only ever says "fine" and then
"stop" makes a run fail at 100 percent of its budget having produced nothing.
Reporting how much of the budget is gone lets the policy layer degrade on the
way up: drop to the cheap model, search less widely, verify fewer claims, and
arrive at a smaller finished report instead of a larger unfinished one.

A ceiling of 0 means unlimited. That is deliberate rather than a sentinel to be
embarrassed about: "no limit" is a real thing an operator asks for, and writing
it as a very large number makes the intent unreadable at the call site.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .contracts import Budget, BudgetExceeded, Spend

# Pressure past which the policy layer starts trading quality for completion.
DEGRADE_AT = 0.70
# Pressure past which only work that finishes the run is worth starting.
CRITICAL_AT = 0.88


@dataclass
class Reading:
    """One ceiling's utilisation, used for both the guard and the UI."""

    name: str
    used: float
    limit: float

    @property
    def fraction(self) -> float:
        # An unlimited ceiling never contributes pressure.
        return 0.0 if self.limit <= 0 else self.used / self.limit

    @property
    def breached(self) -> bool:
        return self.limit > 0 and self.used >= self.limit


class BudgetGuard:
    """Counts what a run has spent and refuses to let it spend more.

    The guard is consulted before an expensive action, not after, because the
    point is to avoid paying for the call that breaches the ceiling rather than
    to notice afterwards that it was paid for.
    """

    def __init__(self, budget: Budget, spent: Spend | None = None) -> None:
        self.budget = budget
        self.spent = spent or Spend()
        # Wall clock is the one dimension that accrues without anyone spending
        # anything, so it is measured from when this guard started rather than
        # accumulated into `spent` by the callers.
        self._started = time.monotonic()
        self._carried_seconds = self.spent.seconds
        self.warned: set[str] = set()
        # Researchers run in a thread pool and every one of them charges this
        # guard. A read, modify, write on a counter without the lock loses one
        # of two concurrent charges, and a lost charge is a ceiling that never
        # binds, which is the one failure this class exists to prevent.
        self._lock = threading.RLock()

    # ------------------------------------------------------------ accounting

    def elapsed(self) -> float:
        return self._carried_seconds + (time.monotonic() - self._started)

    def sync_seconds(self) -> None:
        with self._lock:
            self.spent.seconds = self.elapsed()

    def readings(self) -> list[Reading]:
        with self._lock:
            b, s = self.budget, self.spent
            return [
                Reading("steps", s.steps, b.max_steps),
                Reading("tokens", s.tokens, b.max_tokens),
                Reading("usd", s.usd, b.max_usd),
                Reading("seconds", self.elapsed(), b.max_seconds),
                Reading("tool_calls", s.tool_calls, b.max_tool_calls),
            ]

    def pressure(self) -> float:
        """How close the tightest ceiling is, from 0 to 1.

        The maximum rather than the mean: a run that has spent 95 percent of
        its time and 5 percent of its tokens is 95 percent done, and averaging
        would report it as comfortable right up to the moment it stopped.
        """
        return max((r.fraction for r in self.readings()), default=0.0)

    @property
    def degraded(self) -> bool:
        return self.pressure() >= DEGRADE_AT

    @property
    def critical(self) -> bool:
        return self.pressure() >= CRITICAL_AT

    def remaining(self, name: str) -> float:
        for reading in self.readings():
            if reading.name == name:
                return float("inf") if reading.limit <= 0 else max(0.0, reading.limit - reading.used)
        return float("inf")

    # ---------------------------------------------------------------- checks

    def check(self) -> None:
        """Raise if any ceiling is already met. The orchestrator pauses on this."""
        self.sync_seconds()
        for reading in self.readings():
            if reading.breached:
                raise BudgetExceeded(
                    reading.name,
                    f"The {reading.name} ceiling of {_fmt(reading.limit)} is spent "
                    f"({_fmt(reading.used)} used). Raise it and resume to carry on.",
                )

    def check_headroom(self, *, tokens: int = 0, usd: float = 0.0) -> None:
        """Raise if the action about to be taken would breach a ceiling.

        Estimates are approximate by nature, so this is deliberately not exact:
        it exists to stop a single enormous call from blowing through a ceiling
        that `check` would only have noticed afterwards.
        """
        self.check()
        b = self.budget
        if b.max_tokens > 0 and tokens and self.spent.tokens + tokens > b.max_tokens:
            raise BudgetExceeded(
                "tokens",
                f"This call needs about {tokens:,} tokens and only "
                f"{int(self.remaining('tokens')):,} remain in the run's ceiling.",
            )
        if b.max_usd > 0 and usd and self.spent.usd + usd > b.max_usd:
            raise BudgetExceeded(
                "usd",
                f"This call is estimated at ${usd:.4f} and only "
                f"${self.remaining('usd'):.4f} remains in the run's ceiling.",
            )

    def can_afford_step(self) -> bool:
        """Whether another step fits, without raising. Used to decide, not to stop."""
        try:
            self.check()
        except BudgetExceeded:
            return False
        return True

    # --------------------------------------------------------------- charges

    def charge_step(self) -> None:
        with self._lock:
            self.spent.steps += 1

    def charge_tool(self) -> None:
        with self._lock:
            self.spent.tool_calls += 1

    def charge_llm(self, *, tokens_in: int, tokens_out: int, usd: float) -> None:
        with self._lock:
            self.spent.llm_calls += 1
            self.spent.tokens_in += max(0, tokens_in)
            self.spent.tokens_out += max(0, tokens_out)
            self.spent.usd += max(0.0, usd)

    # ------------------------------------------------------------- reporting

    def newly_warned(self) -> list[Reading]:
        """Ceilings that have just crossed the degrade line, reported once each."""
        out = []
        with self._lock:
            for reading in self.readings():
                if reading.fraction >= DEGRADE_AT and reading.name not in self.warned:
                    self.warned.add(reading.name)
                    out.append(reading)
        return out

    def snapshot(self) -> dict[str, object]:
        self.sync_seconds()
        return {
            "budget": self.budget.to_dict(),
            "spent": self.spent.to_dict(),
            "pressure": round(self.pressure(), 4),
            "degraded": self.degraded,
            "critical": self.critical,
            "readings": [
                {
                    "name": r.name,
                    "used": round(r.used, 6),
                    "limit": r.limit,
                    "fraction": round(r.fraction, 4),
                }
                for r in self.readings()
            ],
        }


def _fmt(value: float) -> str:
    if value >= 1000:
        return f"{value:,.0f}"
    if value == int(value):
        return str(int(value))
    return f"{value:.4f}".rstrip("0").rstrip(".")
