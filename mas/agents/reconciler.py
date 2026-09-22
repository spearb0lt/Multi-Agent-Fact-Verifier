"""The Reconciler: decide whether two verified claims actually conflict.

This is the half of cross-referencing that a prompt cannot honestly do. Asking
the Analyst to "mark a claim contested when findings disagree" relies on one
model noticing a conflict between two statements it produced in the same
breath, and it will usually not. A contradiction that nobody notices is worse
than one nobody looked for, because the report then asserts one side of a real
disagreement as settled fact.

So the mechanism is split. The vectors find which pairs are even about the same
thing, which is arithmetic and free. The model is then asked one narrow
question about each surviving pair: do these conflict, and how. That is a
judgement a model makes well, and there are only a handful of pairs to ask
about because the arithmetic threw the rest away.

Two claims from the same source are skipped upstream. Those disagree because
somebody misread a page, which is a different problem from two outlets
reporting different things.
"""
from __future__ import annotations

from typing import Any

from ..core.util import truncate
from ..kernel.semantic import candidate_pairs
from .base import HOUSE_STYLE, Agent

SCHEMA = """{
  "relation": "contradiction | tension | agreement | unrelated",
  "explanation": "one sentence on what the disagreement actually is",
  "which_is_better_supported": "A | B | neither",
  "why": "one sentence, referring to the sources behind each"
}"""


class Reconciler(Agent):
    role = "Reconciler"
    goal = "Decide whether two claims about the same thing actually conflict."
    temperature = 0.1
    max_tokens = 700

    def system_prompt(self, ctx: Any) -> str:
        return f"""You are given two claims that verification has already passed, and
that are about the same subject. You decide how they relate.

- contradiction: they cannot both be true. One says a thing happened, the other
  says it did not; one gives a figure, the other gives an incompatible figure.
- tension: they can both be true but pull in different directions, or they
  describe the same thing under different conditions, or over different periods.
- agreement: they say the same thing in different words. This is common and is
  not a problem.
- unrelated: they are about different things and the similarity is superficial.

Be strict about what a contradiction is. "Rose sharply" and "rose 12 percent"
agree. "Improves outcomes in mice" and "no effect in humans" are in tension,
not contradiction, because they are different populations. "The rate was cut"
and "the rate was held" is a contradiction.

When you call it a contradiction or a tension, say which claim the sources
support better, judged on how many independent outlets stand behind each and
how directly they state it. Answer "neither" if that is honest.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        pair = task.payload.get("pair") or {}
        return self._prompt(ctx, pair.get("a") or {}, pair.get("b") or {})

    def _prompt(self, ctx: Any, a: dict[str, Any], b: dict[str, Any]) -> str:
        index = ctx.evidence_index()

        def describe(claim: dict[str, Any]) -> str:
            refs = claim.get("sources", [])
            outlets = sorted(
                {index[ref]["domain"] for ref in refs if ref in index and index[ref].get("domain")}
            )
            verdict = claim.get("verdict") or {}
            return (
                f"{claim.get('text', '')}\n"
                f"  sources: {', '.join(refs) or 'none'}"
                f"  outlets: {', '.join(outlets) or 'unknown'}\n"
                f"  verification: {verdict.get('verdict', 'unknown')} "
                f"({verdict.get('confidence', 'unknown')} confidence)"
            )

        return (
            f"Brief, for context:\n{ctx.brief}\n\n"
            f"CLAIM A ({a.get('id', '?')}):\n{describe(a)}\n\n"
            f"CLAIM B ({b.get('id', '?')}):\n{describe(b)}\n\n"
            f"How do they relate?"
        )

    def reconcile(self, ctx: Any) -> list[dict[str, Any]]:
        """Adjudicate every pair worth asking about. Returns the conflicts found."""
        supported = ctx.board.supported_claims()
        pairs = candidate_pairs(
            supported,
            threshold=float(ctx.option("contradiction_threshold", 0.62)),
            cap=int(ctx.option("max_contradiction_checks", 8)),
        )
        if not pairs:
            ctx.bus.log(
                f"No two of the {len(supported)} verified claims were close enough in "
                f"subject to be worth comparing.",
                level="info",
            )
            return []

        ctx.bus.log(
            f"{len(pairs)} claim pair(s) are about the same subject and will be checked "
            f"for conflict.",
            level="info",
            pairs=[(a["id"], b["id"], score) for a, b, score in pairs],
        )

        conflicts: list[dict[str, Any]] = []
        for a, b, score in pairs:
            # The budget is consulted per pair, because a run that ran out here
            # should still keep the conflicts it already found.
            if not ctx.budget.can_afford_step():
                ctx.bus.log(
                    "Stopped comparing claims early: the budget is spent.", level="warning"
                )
                break

            payload = self.ask(
                ctx,
                prompt=self._prompt(ctx, a, b),
                schema_hint=SCHEMA,
                default={},
                purpose="reconciler.compare",
            )
            if not isinstance(payload, dict):
                continue

            relation = str(payload.get("relation") or "").strip().lower()
            if relation not in {"contradiction", "tension", "agreement", "unrelated"}:
                continue
            if relation in {"agreement", "unrelated"}:
                continue

            better = str(payload.get("which_is_better_supported") or "neither").strip().upper()
            conflicts.append(
                {
                    "a": a["id"],
                    "b": b["id"],
                    "a_text": a.get("text", ""),
                    "b_text": b.get("text", ""),
                    "a_sources": a.get("sources", []),
                    "b_sources": b.get("sources", []),
                    "relation": relation,
                    "explanation": truncate(str(payload.get("explanation") or "").strip(), 400),
                    "better_supported": better if better in {"A", "B"} else "neither",
                    "why": truncate(str(payload.get("why") or "").strip(), 300),
                    "similarity": score,
                }
            )

        return conflicts
