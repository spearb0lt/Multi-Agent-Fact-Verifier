"""Which model each role gets, and what happens to that choice under pressure.

Not every role needs the same model. Deciding whether a search result is on
topic is a mechanical judgement a small model makes as well as a large one, and
it happens dozens of times per run. Deciding how to decompose a brief happens
once and sets the shape of everything downstream. Spending the same model on
both is the single most expensive mistake available here, so roles declare a
tier and the tier picks the model.

The second half is degradation. When the budget guard reports pressure, strong
roles fall back to the cheap model rather than the run stopping at full price
with nothing finished. A smaller model writing the final section is a worse
report; no final section is not a report at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ..core.llm import registry as llm
from .budget import DEGRADE_AT
from .contracts import EventKind

if TYPE_CHECKING:  # pragma: no cover
    from .context import RunContext


class Tier(StrEnum):
    """How much model judgement a role's work actually needs."""

    # High volume, low judgement. Filtering, extraction, classification.
    CHEAP = "cheap"
    # Once or twice per run, and the quality of the whole run turns on it.
    STRONG = "strong"


# The default tier per role. A run's config can override any of these, which is
# what lets someone with a paid key put the strong model everywhere and someone
# on a free tier put the cheap model everywhere, without touching code.
ROLE_TIERS: dict[str, Tier] = {
    "Supervisor": Tier.CHEAP,
    "Planner": Tier.STRONG,
    "Researcher": Tier.CHEAP,
    "Analyst": Tier.STRONG,
    "FactChecker": Tier.CHEAP,
    "Writer": Tier.STRONG,
    "Editor": Tier.STRONG,
    "Critic": Tier.STRONG,
}


@dataclass
class Choice:
    provider: str
    model: str
    tier: Tier
    downgraded: bool = False
    reason: str = ""

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model}"


class ModelPolicy:
    """Resolves a role to a concrete provider and model for this run."""

    def __init__(
        self,
        *,
        provider: str = "",
        model: str = "",
        cheap_model: str = "",
        role_tiers: dict[str, str] | None = None,
        allow_degrade: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.cheap_model = cheap_model
        self.allow_degrade = allow_degrade
        self.overrides: dict[str, Tier] = {}
        for role, value in (role_tiers or {}).items():
            try:
                self.overrides[role] = Tier(str(value))
            except ValueError:
                continue
        self._announced: set[str] = set()

    def tier_for(self, role: str) -> Tier:
        return self.overrides.get(role) or ROLE_TIERS.get(role, Tier.CHEAP)

    def resolve(self, role: str, ctx: RunContext | None = None) -> Choice:
        tier = self.tier_for(role)
        downgraded = False
        reason = ""

        pressure = ctx.budget.pressure() if ctx is not None else 0.0
        if tier is Tier.STRONG and self.allow_degrade and pressure >= DEGRADE_AT:
            tier = Tier.CHEAP
            downgraded = True
            reason = (
                f"{int(pressure * 100)} percent of the run's budget is spent, so "
                f"{role} is using the cheap model to reach a finished result."
            )

        wants_cheap = tier is Tier.CHEAP
        pinned = self.cheap_model if wants_cheap else self.model
        selection = llm.resolve(self.provider or None, pinned or None, cheap=wants_cheap)

        choice = Choice(
            provider=selection.provider.id,
            model=selection.model,
            tier=tier,
            downgraded=downgraded,
            reason=reason,
        )

        if downgraded and ctx is not None and role not in self._announced:
            self._announced.add(role)
            ctx.bus.emit(
                EventKind.MODEL_DOWNGRADE,
                reason,
                agent=role,
                level="warning",
                model=choice.model,
                provider=choice.provider,
            )
        return choice

    def describe(self) -> dict[str, str]:
        """The role to tier map, for the UI and the run report."""
        roles = set(ROLE_TIERS) | set(self.overrides)
        return {role: self.tier_for(role).value for role in sorted(roles)}
