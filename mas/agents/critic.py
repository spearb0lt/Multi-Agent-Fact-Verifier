"""The Critic: decide whether the report is good enough, and say why not.

This is the role that closes the loop. It scores the edited report against the
brief and the verified claims, and either accepts it or sends it back to the
Writer with a specific list of what to fix. That cycle, bounded by
MAX_REVISIONS, is the self correction in the system, and it is the difference
between a pipeline that emits whatever the writer produced first and one that
holds its own output to a standard.

Two things keep the loop honest. The Critic sees the claims and their verdicts,
so it can catch a sentence asserting something that never passed verification.
And its issues have to be specific and actionable, because "improve the flow"
sends the Writer round the loop for nothing.

The mechanical checks run first and are not the model's opinion. An uncited
factual paragraph or a reference pointing at a source that does not exist is a
fact, and it is fed to the Critic rather than left for it to notice.
"""
from __future__ import annotations

import re
from typing import Any

from ..core.util import truncate
from .base import HOUSE_STYLE, Agent, claims_digest

CITATION_RE = re.compile(r"\[(S\d+)\]")
HEADING_RE = re.compile(r"^#{1,6}\s", re.M)

SCHEMA = """{
  "score": 82,
  "verdict": "accept | revise",
  "strengths": ["what the report does well"],
  "issues": [
    {
      "severity": "critical | major | minor",
      "issue": "what is wrong, specifically, quoting the sentence if relevant",
      "fix": "what the Writer should do about it"
    }
  ],
  "unsupported_statements": ["any sentence asserting something not in the verified claims"]
}"""


class Critic(Agent):
    role = "Critic"
    goal = "Judge the report against the brief and the verified claims."
    temperature = 0.15
    max_tokens = 2200

    def system_prompt(self, ctx: Any) -> str:
        bar = int(ctx.option("quality_bar", 75))
        return f"""You are the Critic. You judge a research report and decide whether it
is fit to publish. You are the last check before it goes out, and you are hard
to please.

Score out of 100 on:
- Faithfulness (40). Does every factual sentence trace to a verified claim? An
  assertion that is not in the claims is a critical issue, however plausible.
- Coverage (25). Does it answer the brief? Are the gaps stated honestly?
- Citation quality (20). Is every factual sentence cited? Are references real?
- Clarity and structure (15). Does it read as one document, without padding?

Verdict is "accept" when the score is {bar} or above AND there is no critical
issue. Otherwise "revise".

Rules for issues:
- Be specific. Quote the offending sentence. "Tighten the prose" is useless.
- Give a fix the Writer can carry out without going back to research.
- Do not ask for facts that are not in the verified claims. If the report is
  thin because the research was thin, that is a coverage note, not a demand
  for invention.
- Do not invent issues to seem rigorous. If it is good, accept it.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        return self._prompt(ctx)

    def _mechanical(self, ctx: Any, text: str) -> dict[str, Any]:
        """Facts about the document, established before any model reads it."""
        known = {row["ref"] for row in ctx.evidence()}
        used = set(CITATION_RE.findall(text))
        phantom = sorted(used - known)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        uncited = []
        for para in paragraphs:
            if para.startswith("#") or para.startswith(">"):
                continue
            # A paragraph with a digit or a proper noun and no citation is the
            # shape an unsourced assertion takes.
            looks_factual = bool(re.search(r"\d", para)) or len(para.split()) > 25
            if looks_factual and not CITATION_RE.search(para):
                uncited.append(truncate(para, 200))

        return {
            "citations_used": sorted(used),
            "phantom_citations": phantom,
            "uncited_paragraphs": uncited[:6],
            "headings": len(HEADING_RE.findall(text)),
            "words": len(text.split()),
        }

    def _prompt(self, ctx: Any) -> str:
        report = ctx.board.draft.get("markdown", "")
        checks = self._mechanical(ctx, report)

        mechanical_note = ""
        if checks["phantom_citations"]:
            mechanical_note += (
                f"\n- These references appear in the report but no such source exists: "
                f"{', '.join(checks['phantom_citations'])}. This is critical."
            )
        if checks["uncited_paragraphs"]:
            joined = "\n  ".join(f'"{p}"' for p in checks["uncited_paragraphs"])
            mechanical_note += f"\n- These paragraphs state things with no citation:\n  {joined}"
        if mechanical_note:
            mechanical_note = "\n\nChecks already run on the document:" + mechanical_note

        return (
            f"Brief:\n{ctx.brief}\n\n"
            f"The verified claims the report was allowed to use:\n"
            f"{claims_digest(ctx.board.supported_claims(), with_verdicts=True)}\n\n"
            f"Revision {ctx.board.revision} of the report:\n\n{report}"
            f"{mechanical_note}\n\n"
            f"Judge it."
        )

    def critique(self, ctx: Any) -> dict[str, Any]:
        report = ctx.board.draft.get("markdown", "")
        checks = self._mechanical(ctx, report)

        payload = self.ask(
            ctx, prompt=self._prompt(ctx), schema_hint=SCHEMA, default={}, purpose="critic.judge"
        )
        if not isinstance(payload, dict):
            payload = {}

        try:
            score = int(float(payload.get("score", 0)))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(100, score))

        issues = []
        for item in payload.get("issues") or []:
            if isinstance(item, str):
                item = {"issue": item}
            if not isinstance(item, dict):
                continue
            text = str(item.get("issue") or "").strip()
            if not text:
                continue
            issues.append(
                {
                    "severity": str(item.get("severity") or "minor").lower(),
                    "issue": truncate(text, 400),
                    "fix": truncate(str(item.get("fix") or "").strip(), 300),
                }
            )

        # A phantom citation is critical whatever the model thought, so it is
        # added rather than left to the model's judgement.
        for ref in checks["phantom_citations"]:
            issues.append(
                {
                    "severity": "critical",
                    "issue": f"The report cites {ref}, which is not a source in this run.",
                    "fix": f"Remove {ref} and cite a real source, or delete the sentence.",
                }
            )

        bar = int(ctx.option("quality_bar", 75))
        has_critical = any(i["severity"] == "critical" for i in issues)
        verdict = str(payload.get("verdict") or "").strip().lower()
        if verdict not in {"accept", "revise"}:
            verdict = "accept" if score >= bar and not has_critical else "revise"
        if has_critical:
            verdict = "revise"
        if score >= bar and not issues:
            verdict = "accept"

        return {
            "score": score,
            "verdict": verdict,
            "bar": bar,
            "strengths": [str(s) for s in (payload.get("strengths") or []) if str(s).strip()][:5],
            "issues": issues[:10],
            "unsupported_statements": [
                truncate(str(s), 300)
                for s in (payload.get("unsupported_statements") or [])
                if str(s).strip()
            ][:6],
            "checks": checks,
            "revision": ctx.board.revision,
        }
