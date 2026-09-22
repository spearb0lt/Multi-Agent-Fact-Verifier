"""The Analyst: turn a pile of findings into a short list of checkable claims.

Researchers produce overlapping, uneven material: the same fact from three
angles, two findings that quietly contradict each other, and several that are
really the same statement with different wording. Handing that directly to a
Fact Checker would mean verifying the same thing three times and verifying
nothing that matters once.

So this role consolidates. It also does the one thing no other role is
positioned to do: notice that two findings disagree. A researcher only saw its
own question, and the Writer will only see what survived verification, so a
contradiction spotted here is a contradiction the report can report honestly
instead of picking one side of by accident.
"""
from __future__ import annotations

from typing import Any

from .base import HOUSE_STYLE, Agent, findings_digest

SCHEMA = """{
  "claims": [
    {
      "text": "one specific checkable statement, with the figure or date in it",
      "sources": ["S1", "S4"],
      "kind": "fact | figure | event | position | prediction",
      "importance": "core | supporting | background",
      "contested": false,
      "contest_note": "if two findings disagree, what the disagreement is"
    }
  ],
  "themes": ["the two to five threads this report should be organised around"],
  "gaps": ["what the findings do not establish that the brief needs"]
}"""


class Analyst(Agent):
    role = "Analyst"
    goal = "Consolidate findings into distinct, checkable claims and name the contradictions."
    temperature = 0.2
    max_tokens = 3000

    def system_prompt(self, ctx: Any) -> str:
        return f"""You are the Analyst. Researchers have gathered findings, each citing
sources they actually read. You turn those into the claims the report will be built from.

What you do:
- Merge findings that say the same thing into one claim, carrying every source
  reference from all of them.
- Split a finding that bundles two facts into two claims.
- Mark a claim contested when findings disagree, and say what the disagreement
  is. Never silently pick a side.
- Drop anything that is opinion, restatement of the brief, or unsupported.
- Keep every source reference exactly as written. Never invent one, never
  renumber, never cite a reference that was not on the finding you merged.

Order the claims so the core ones come first.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        return self._prompt(ctx)

    def _prompt(self, ctx: Any) -> str:
        plan = ctx.board.plan
        answers = ctx.board.items("research_answers")
        answer_note = ""
        if answers:
            answer_note = "\n\nWhat each researcher concluded:\n" + "\n".join(
                f"- {a.get('subquestion_id', '?')}: {a.get('answer', '')}" for a in answers
            )
        return (
            f"Brief:\n{ctx.brief}\n\n"
            f"What the report is meant to establish: {plan.get('angle', '')}\n\n"
            f"Findings on the board:\n{findings_digest(ctx.board.findings)}"
            f"{answer_note}\n\n"
            f"Produce the claims. Aim for between 5 and 18, fewer if the findings "
            f"do not support more."
        )

    def analyse(self, ctx: Any) -> dict[str, Any]:
        payload = self.ask(
            ctx, prompt=self._prompt(ctx), schema_hint=SCHEMA, default={}, purpose="analyst.claims"
        )
        if not isinstance(payload, dict):
            payload = {}

        known = {row["ref"] for row in ctx.evidence()}
        claims: list[dict[str, Any]] = []
        for index, item in enumerate(payload.get("claims") or []):
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if len(text) < 15:
                continue
            # A claim citing a reference that does not exist is the model
            # inventing a source, which is the failure this whole verification
            # stage exists to catch. Drop the phantom references here rather
            # than letting the Fact Checker try to read them.
            sources = [
                ref for ref in (str(s).strip().upper() for s in item.get("sources") or [])
                if ref in known
            ]
            if not sources:
                continue
            claims.append(
                {
                    "id": f"C{index + 1}",
                    "text": text,
                    "sources": sources,
                    "kind": str(item.get("kind") or "fact").lower(),
                    "importance": str(item.get("importance") or "supporting").lower(),
                    "contested": bool(item.get("contested")),
                    "contest_note": str(item.get("contest_note") or "").strip(),
                }
            )

        return {
            "claims": claims,
            "themes": [str(t) for t in (payload.get("themes") or []) if str(t).strip()][:6],
            "gaps": [str(g) for g in (payload.get("gaps") or []) if str(g).strip()][:6],
        }
