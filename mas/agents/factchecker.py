"""The Fact Checker: does the source actually say that?

One checker runs per claim, in parallel. Each one reads the stored text of the
sources the claim cites and rules on whether they support it, and may search
for a second source when the claim rests on one.

The important design decision is what happens to a claim that fails. It is not
flagged in the draft and it is not footnoted as uncertain. It never reaches the
Writer at all, because the Writer is given only the claims that passed. A
system that marks its own unverified assertions still publishes them, and a
reader skims past the marking. Removing them is the only version of this that
actually means anything.

Corroboration is counted in independent domains, not in references. Three
references to the same wire story reprinted by three outlets is one source, and
the dedup in the ranking layer catches most of that, but the count here is what
the verdict actually rests on.
"""
from __future__ import annotations

from typing import Any

from ..core.util import truncate
from .base import HOUSE_STYLE, Agent, evidence_digest

SCHEMA = """{
  "verdict": "supported | partly_supported | unsupported | contradicted",
  "confidence": "high | medium | low",
  "reasoning": "one or two sentences on what the sources do and do not establish",
  "supporting_sources": ["S1"],
  "correction": "if the claim is nearly right but the figure or date is wrong, the corrected statement, otherwise empty"
}"""


class FactChecker(Agent):
    role = "FactChecker"
    goal = "Rule on whether the cited sources establish the claim."
    tools = ("read_evidence", "web_search", "fetch_page")
    temperature = 0.1
    max_tokens = 1200
    max_iterations = 4

    def system_prompt(self, ctx: Any) -> str:
        return f"""You are a Fact Checker. You are given ONE claim and the sources it
cites. You rule on whether those sources actually establish it.

The verdicts:
- supported: a cited source states this, in substance, and you can point to it.
- partly_supported: the direction is right but the claim overstates it, or the
  figure, date or attribution is off. Give the correction.
- unsupported: the sources do not say this. They may not contradict it either.
  Absence of support is unsupported.
- contradicted: a source says the opposite.

How to work:
1. Read the cited sources with read_evidence first. Usually that settles it.
2. Rule. Do not go looking for more unless the claim is important and rests on
   a single source, in which case one search for a second source is worth it.
3. Be strict about specifics. "Rose sharply" is not support for "rose 12
   percent". A 2024 figure is not support for a claim about 2026.
4. Do not rule on whether the claim is plausible, interesting or fair. Only on
   whether the sources establish it.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        claim = task.payload.get("claim") or {}
        refs = tuple(claim.get("sources") or ())
        return (
            f"CLAIM {claim.get('id', '?')}:\n{claim.get('text', '')}\n\n"
            f"It cites: {', '.join(refs) or 'nothing'}\n\n"
            f"The text of those sources:\n{evidence_digest(ctx, refs, chars=3500)}\n\n"
            f"Rule on it."
        )

    def final_schema(self, ctx: Any) -> str:
        return SCHEMA

    def check(self, ctx: Any, task: Any) -> dict[str, Any]:
        claim = task.payload.get("claim") or {}
        result = self.think(ctx, task)
        output = result.output if isinstance(result.output, dict) else {}

        verdict = str(output.get("verdict") or "").strip().lower().replace(" ", "_")
        if verdict not in {"supported", "partly_supported", "unsupported", "contradicted"}:
            # A checker that produced nothing usable must not be read as a pass.
            # Defaulting to unsupported is the safe direction: the claim is
            # dropped rather than published on the strength of a failed check.
            verdict = "unsupported"

        known = {row["ref"] for row in ctx.evidence()}
        supporting = [
            ref
            for ref in (str(s).strip().upper() for s in output.get("supporting_sources") or [])
            if ref in known
        ] or list(claim.get("sources") or [])

        index = ctx.evidence_index()
        domains = {
            index[ref]["domain"]
            for ref in supporting
            if ref in index and index[ref].get("domain")
        }

        return {
            "claim_id": claim.get("id", ""),
            "verdict": verdict,
            "confidence": str(output.get("confidence") or "medium").lower(),
            "reasoning": truncate(str(output.get("reasoning") or "").strip(), 600),
            "sources": supporting,
            "independent_domains": len(domains),
            "domains": sorted(domains),
            "corroborated": len(domains) >= int(ctx.option("corroboration_min", 2)),
            "correction": truncate(str(output.get("correction") or "").strip(), 400),
            "checked_by": self.role,
            "iterations": result.iterations,
        }
