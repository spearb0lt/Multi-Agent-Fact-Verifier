"""Request and response shapes for the HTTP API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BudgetIn(BaseModel):
    """Ceilings for one run. Omitted fields fall back to the deployment's."""

    max_steps: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_usd: float | None = Field(default=None, ge=0)
    max_seconds: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)

    def merged(self) -> Any:
        from ..core import settings
        from ..kernel.contracts import Budget

        return Budget(
            max_steps=self.max_steps if self.max_steps is not None else settings.MAX_RUN_STEPS,
            max_tokens=(
                self.max_tokens if self.max_tokens is not None else settings.MAX_RUN_TOKENS
            ),
            max_usd=self.max_usd if self.max_usd is not None else settings.MAX_RUN_USD,
            max_seconds=(
                self.max_seconds if self.max_seconds is not None else settings.MAX_RUN_SECONDS
            ),
            max_tool_calls=(
                self.max_tool_calls
                if self.max_tool_calls is not None
                else settings.MAX_TOOL_CALLS
            ),
        )


class Credentials(BaseModel):
    """A visitor's own provider keys.

    Held in the worker's memory for the life of the run and never written to
    the database. A run resumed after a restart therefore falls back to the
    server's keys, which the API reports rather than hiding.
    """

    keys: dict[str, str] = Field(default_factory=dict)
    base_urls: dict[str, str] = Field(default_factory=dict)
    accounts: dict[str, str] = Field(default_factory=dict)


class RunIn(BaseModel):
    brief: str = Field(min_length=8, max_length=4000)
    workflow: str = "research_report"
    provider: str = ""
    model: str = ""
    depth: str = Field(default="standard", pattern="^(quick|standard|deep)$")
    budget: BudgetIn | None = None
    max_revisions: int | None = Field(default=None, ge=0, le=5)
    max_research_rounds: int | None = Field(default=None, ge=1, le=4)
    agent_concurrency: int | None = Field(default=None, ge=1, le=8)
    role_tiers: dict[str, str] | None = None
    allow_degrade: bool = True
    # Park the run after planning until someone approves, before research
    # spends most of the run's tokens.
    approve_plan: bool = False
    # Start executing straight away. False creates the run and leaves it
    # pending, which is what a caller wants when queueing work for later.
    start: bool = True
    credentials: Credentials | None = None


class ResumeIn(BaseModel):
    budget: BudgetIn | None = None
    credentials: Credentials | None = None
    # Continuing on a different provider is the direct answer to the most
    # common reason a run stops on a free tier. The blackboard does not care
    # which model produced what is already on it.
    provider: str = ""
    model: str = ""


class AnswerIn(BaseModel):
    reply: str = Field(min_length=1, max_length=2000)
    note: str = ""
    resume: bool = True


class KeysIn(BaseModel):
    credentials: Credentials
