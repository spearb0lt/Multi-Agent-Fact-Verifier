"""The Writer: build the report out of verified claims and nothing else.

The Writer is given the claims that passed verification, the sources behind
them, and the plan's themes. It is not given the raw findings, the rejected
claims or the research transcript. That restriction is the point: a writer with
access to everything will reach for the vivid unverified detail, and the whole
verification stage becomes decorative.

Citations are the other constraint. Every factual sentence carries the
reference of a source that was actually fetched, and the reference numbering
comes from the evidence table rather than from the order things appear in the
draft, so a citation means the same thing in the draft, the report and the
source list.
"""
from __future__ import annotations

from typing import Any

from ..core.util import truncate
from .base import HOUSE_STYLE, Agent, claims_digest, evidence_digest


class Writer(Agent):
    role = "Writer"
    goal = "Write the report from verified claims, citing every fact."
    temperature = 0.35
    max_tokens = 4000

    def system_prompt(self, ctx: Any) -> str:
        return f"""You are the Writer. You produce the report itself, in markdown.

Absolute rules:
1. Use ONLY the verified claims you are given. Do not add a fact, a figure, a
   date, a name or a quotation from your own knowledge. If you find yourself
   writing something that is not in the claims, delete it.
2. Every factual sentence ends with the source reference or references in
   square brackets, like [S3] or [S1][S7]. Use the references exactly as given.
3. Where a claim is marked contested, report the disagreement and cite both
   sides. Do not resolve it.
4. Where the claims do not settle something the brief asked about, say so
   plainly in the "What is not settled" section. Do not fill the gap.

Structure:

# <title>

## Summary
Three to five sentences a reader could stop after and still be correctly
informed. Cited.

## <a theme heading>
## <another theme heading>
As many as the themes need, ordered so the most important comes first. Each is
a few short paragraphs, every factual sentence cited.

## What is not settled
What the sources do not establish, and where they disagree. Write "Nothing
significant" only if that is honestly true.

No preamble before the title. No sign off, no author persona, no "in
conclusion". Do not include a sources list: it is added afterwards.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        return self._prompt(ctx)

    def _prompt(self, ctx: Any) -> str:
        plan = ctx.board.plan
        claims = ctx.board.supported_claims()
        refs = tuple(
            sorted({ref for claim in claims for ref in claim.get("sources", [])})
        )

        critique_note = ""
        critiques = ctx.board.critiques
        if critiques:
            latest = critiques[-1]
            issues = "\n".join(
                f"- [{i.get('severity', 'minor')}] {i.get('issue', '')}"
                + (f" Fix: {i['fix']}" if i.get("fix") else "")
                for i in (latest.get("issues") or [])
            )
            critique_note = (
                f"\n\nThis is revision {ctx.board.revision + 1}. The Critic rejected the "
                f"previous draft, scoring it {latest.get('score', '?')} out of 100. "
                f"Fix exactly these problems and change nothing else:\n{issues}\n\n"
                f"The previous draft:\n{truncate(ctx.board.draft.get('markdown', ''), 6000)}"
            )

        themes = plan.get("themes") or ctx.board.mapping("analysis").get("themes") or []
        theme_note = f"\nThemes to organise around: {'; '.join(themes)}" if themes else ""

        rejected = ctx.board.rejected_claims()
        rejected_note = ""
        if rejected:
            rejected_note = (
                "\n\nThese claims FAILED verification. They must not appear in the "
                "report in any form:\n"
                + "\n".join(f"- {c.get('text', '')}" for c in rejected[:12])
            )

        return (
            f"Brief:\n{ctx.brief}\n\n"
            f"Title to use: {plan.get('title', '')}\n"
            f"What this report establishes: {plan.get('angle', '')}{theme_note}\n\n"
            f"VERIFIED CLAIMS, the only material you may use:\n"
            f"{claims_digest(claims, with_verdicts=True)}\n\n"
            f"The sources behind them, for wording and detail:\n"
            f"{evidence_digest(ctx, refs, chars=900)}"
            f"{rejected_note}{critique_note}"
        )

    def draft(self, ctx: Any) -> dict[str, Any]:
        markdown = self.write(ctx, prompt=self._prompt(ctx), purpose="writer.draft").strip()
        return {
            "markdown": markdown,
            "revision": ctx.board.revision,
            "claims_used": len(ctx.board.supported_claims()),
            "words": len(markdown.split()),
        }
