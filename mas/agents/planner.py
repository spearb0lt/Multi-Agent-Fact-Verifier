"""The Planner: turn a brief into a set of separately researchable questions.

The quality of a run is decided here more than anywhere else. Researchers are
good at answering a specific question and hopeless at answering a vague one, so
a plan of five sharp subquestions produces a report, and a plan of five
restatements of the brief produces five copies of the same shallow search.

The Planner gets the strong model for that reason, and it runs once.
"""
from __future__ import annotations

from typing import Any

from ..core.util import truncate
from .base import HOUSE_STYLE, Agent

SCHEMA = """{
  "title": "a short factual title for the finished report",
  "angle": "one sentence on what this report will actually establish",
  "subquestions": [
    {
      "question": "a specific, separately answerable question",
      "why": "what the report needs it for",
      "priority": "high | medium | low",
      "search_hints": ["a search query someone could type", "another"]
    }
  ],
  "success_criteria": ["what must be established for this report to be useful"],
  "risks": ["what is likely to be hard, contested or badly sourced"]
}"""


class Planner(Agent):
    role = "Planner"
    goal = "Decompose the brief into questions that can each be researched on their own."
    temperature = 0.3
    max_tokens = 2000

    def system_prompt(self, ctx: Any) -> str:
        return f"""You are the Planner in a research team. You do not do the research.
You decide what the researchers should go and find out.

A good subquestion:
- can be answered by reading a handful of specific pages,
- has a checkable answer, not an opinion,
- does not overlap with the other subquestions,
- names the specific thing to look for: the figure, the date, the decision, the party.

A bad subquestion restates the brief, asks for "an overview", or bundles three
questions into one sentence.

Cover the brief completely between them. Include the background a reader needs,
the current state of the matter, and whatever is disputed about it.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        return self._prompt(ctx)

    def _prompt(self, ctx: Any) -> str:
        lessons = ctx.recall(kind="lesson", limit=6)
        memory_note = ""
        if lessons:
            memory_note = "\n\nWhat earlier runs learned:\n" + "\n".join(
                f"- {truncate(row['content'], 200)}" for row in lessons
            )
        gaps = ctx.board.get("gaps", [])
        gap_note = ""
        if gaps:
            gap_note = (
                "\n\nA first round of research has already happened and left these gaps. "
                "Plan ONLY to close them, and do not repeat what is already answered:\n"
                + "\n".join(f"- {truncate(str(g), 300)}" for g in gaps)
            )
        return (
            f"Brief:\n{ctx.brief}\n\n"
            f"Produce exactly {ctx.breadth} subquestions."
            f"{gap_note}{memory_note}"
        )

    def final_schema(self, ctx: Any) -> str:
        return SCHEMA

    def plan(self, ctx: Any) -> dict[str, Any]:
        payload = self.ask(
            ctx,
            prompt=self._prompt(ctx),
            schema_hint=SCHEMA,
            default={},
            purpose="planner.plan",
        )
        if not isinstance(payload, dict):
            payload = {}

        raw = payload.get("subquestions") or []
        subquestions = []
        for index, item in enumerate(raw[: max(1, ctx.breadth)]):
            if isinstance(item, str):
                item = {"question": item}
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if len(question) < 10:
                continue
            subquestions.append(
                {
                    "id": f"Q{index + 1}",
                    "question": question,
                    "why": str(item.get("why") or "").strip(),
                    "priority": str(item.get("priority") or "medium").lower(),
                    "search_hints": [
                        str(h) for h in (item.get("search_hints") or []) if str(h).strip()
                    ][:4],
                }
            )

        if not subquestions:
            # A plan is required for the run to proceed at all, so a model that
            # produced nothing usable is backstopped with the brief itself
            # rather than failing the run before any research happens.
            subquestions = [
                {
                    "id": "Q1",
                    "question": ctx.brief,
                    "why": "The planner returned no usable subquestions.",
                    "priority": "high",
                    "search_hints": [ctx.brief[:120]],
                }
            ]

        return {
            "title": str(payload.get("title") or "").strip() or truncate(ctx.brief, 80),
            "angle": str(payload.get("angle") or "").strip(),
            "subquestions": subquestions,
            "success_criteria": [
                str(c) for c in (payload.get("success_criteria") or []) if str(c).strip()
            ][:6],
            "risks": [str(r) for r in (payload.get("risks") or []) if str(r).strip()][:6],
        }
