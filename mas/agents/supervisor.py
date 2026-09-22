"""The Supervisor: decide what the team does next.

Every other role does a job. This one decides whether the job was good enough
to move on from, and that is where the system's autonomy actually lives. After
verification it looks at what survived and chooses between writing the report
and sending the researchers back out for another round against the specific
gaps that remain.

The decision is deliberately not left entirely to the model. Some of it is
arithmetic: if two claims passed verification out of fourteen, no amount of
model judgement makes that a report, and if the budget is nearly spent then
another research round is a way to run out of money with nothing to show. So
the hard constraints are evaluated first and the model is asked only when the
answer is genuinely a judgement call. A supervisor that can be talked into
anything is not a control.
"""
from __future__ import annotations

from typing import Any

from ..core.util import truncate
from .base import HOUSE_STYLE, Agent, claims_digest

SCHEMA = """{
  "decision": "write | research_more",
  "reasoning": "one or two sentences on why",
  "gaps": ["the specific things another round should go and find"]
}"""


class Supervisor(Agent):
    role = "Supervisor"
    goal = "Decide whether the evidence is good enough to write from."
    temperature = 0.1
    max_tokens = 900

    def system_prompt(self, ctx: Any) -> str:
        return f"""You supervise a research team. Research and verification have
finished a round. You decide whether to write the report now or send the
researchers back out.

Choose "research_more" only when there is a specific, nameable gap that another
round could realistically close, and closing it would materially change the
report. Name the gaps concretely: what to look for, not "more detail".

Choose "write" when the verified claims answer the brief well enough to produce
an honest report, even if it has to state clearly what remains unsettled. A
report that reports its own gaps is a good outcome. Another research round that
finds nothing is a bad one.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        return self._prompt(ctx)

    def _prompt(self, ctx: Any) -> str:
        supported = ctx.board.supported_claims()
        rejected = ctx.board.rejected_claims()
        analysis = ctx.board.mapping("analysis")
        unanswered = [
            a for a in ctx.board.items("research_answers") if a.get("gaps")
        ]
        gap_note = ""
        if unanswered:
            gap_note = "\n\nGaps the researchers reported:\n" + "\n".join(
                f"- {a.get('subquestion_id', '?')}: {'; '.join(a.get('gaps', []))}"
                for a in unanswered
            )
        analysis_gaps = analysis.get("gaps") or []
        if analysis_gaps:
            gap_note += "\n\nGaps the Analyst reported:\n" + "\n".join(
                f"- {g}" for g in analysis_gaps
            )

        return (
            f"Brief:\n{ctx.brief}\n\n"
            f"Research rounds completed: {ctx.board.get('research_rounds', 1)}\n"
            f"Sources gathered: {len(ctx.evidence())}\n"
            f"Claims that passed verification: {len(supported)}\n"
            f"Claims that failed: {len(rejected)}\n"
            f"Budget spent: {int(ctx.budget.pressure() * 100)} percent\n\n"
            f"The verified claims:\n{claims_digest(supported)}"
            f"{gap_note}"
        )

    def decide(self, ctx: Any) -> dict[str, Any]:
        supported = ctx.board.supported_claims()
        rounds = int(ctx.board.get("research_rounds", 1))
        max_rounds = int(ctx.option("max_research_rounds", 2))

        # The hard constraints first. Each of these is a fact about the run
        # rather than a judgement, and letting the model overrule them is how a
        # budget ceiling turns into a run that stops with nothing written.
        if ctx.budget.critical:
            return self._forced(
                ctx, "write",
                "The budget is nearly spent, so the report is written from what is "
                "already verified rather than risking finishing with nothing.",
            )
        if rounds >= max_rounds:
            return self._forced(
                ctx, "write",
                f"{rounds} research round(s) have run, which is the limit, so the "
                f"report is written from what is verified.",
            )
        if not supported:
            if rounds < max_rounds:
                return self._forced(
                    ctx, "research_more",
                    "No claim survived verification, so there is nothing to write from.",
                    gaps=[
                        f"Nothing verified for: {q.get('question', '')}"
                        for q in ctx.board.subquestions[:4]
                    ],
                )
            return self._forced(
                ctx, "write",
                "No claim survived verification and the research rounds are spent. "
                "The report will state that nothing could be established.",
            )

        payload = self.ask(
            ctx, prompt=self._prompt(ctx), schema_hint=SCHEMA, default={}, purpose="supervisor.route"
        )
        if not isinstance(payload, dict):
            payload = {}

        decision = str(payload.get("decision") or "write").strip().lower()
        if decision not in {"write", "research_more"}:
            decision = "write"

        gaps = [
            truncate(str(g).strip(), 300)
            for g in (payload.get("gaps") or [])
            if str(g).strip()
        ][:5]
        if decision == "research_more" and not gaps:
            # Another round with no stated target is the model deferring, not
            # deciding, and it costs a full round of researchers to find out.
            decision = "write"

        return {
            "decision": decision,
            "reasoning": truncate(str(payload.get("reasoning") or "").strip(), 400),
            "gaps": gaps,
            "forced": False,
            "supported": len(supported),
            "rounds": rounds,
        }

    def _forced(
        self, ctx: Any, decision: str, reasoning: str, *, gaps: list[str] | None = None
    ) -> dict[str, Any]:
        ctx.bus.log(f"Supervisor: {reasoning}", level="info", decision=decision, forced=True)
        return {
            "decision": decision,
            "reasoning": reasoning,
            "gaps": gaps or [],
            "forced": True,
            "supported": len(ctx.board.supported_claims()),
            "rounds": int(ctx.board.get("research_rounds", 1)),
        }
