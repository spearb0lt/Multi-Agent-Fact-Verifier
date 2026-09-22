"""The Researcher: one subquestion, real sources, recorded findings.

This is the only role with a full reason and act loop over the open web, and
several of them run at once, one per subquestion. Everything about the design
follows from two facts about that.

The first is that they share an evidence table. Two researchers looking into
neighbouring questions land on the same page constantly, and the table
deduplicates by normalised URL, so the second one is free and both cite the
same reference. That is why findings carry references rather than URLs: the
reference is stable across the whole run and across agents.

The second is that a researcher is cheap and repeated, which is exactly the
profile the cheap model tier exists for. Judging whether a search result is
worth opening is not work that needs the strong model, and there are dozens of
those judgements per run.
"""
from __future__ import annotations

from typing import Any

from ..core.util import truncate
from .base import HOUSE_STYLE, Agent

SCHEMA = """{
  "answer": "what you established, in two or three sentences, with [S1] style references",
  "confidence": "high | medium | low",
  "gaps": ["anything the question still needs that you could not find"],
  "findings_recorded": 3
}"""


class Researcher(Agent):
    role = "Researcher"
    goal = "Answer one subquestion from real sources and record what you establish."
    tools = (
        "web_search",
        "fetch_page",
        "search_evidence",
        "list_evidence",
        "record_finding",
        "calculate",
        "remember",
    )
    temperature = 0.2
    max_tokens = 1600

    def system_prompt(self, ctx: Any) -> str:
        return f"""You are a Researcher. You are given ONE question and you answer it
from sources you actually read.

How to work:
1. Search for the question. Use the search hints if they were given.
2. Open the two or three results most likely to answer it. Opening a page is
   what makes it citable; a search snippet is not a source.
3. Call record_finding for each specific fact you establish, citing the source
   references that fetch_page returned. Record as you go, not at the end.
   Before searching the web again, try search_evidence: a colleague may have
   already fetched the page that answers your question.
4. When the question is answered, or when further searching is clearly not
   helping, give your final_answer.

Rules that matter:
- A fact with no source you opened does not exist. Do not record what you
  merely believe, and do not record what a snippet implied.
- If a page will not load, is a paywall or is off topic, move to another one.
  Do not retry it.
- Prefer primary sources: the regulator, the company, the filing, the paper.
  Prefer two outlets over one for anything contested.
- You are answering YOUR question only. Leave the other subquestions alone.
- Three solid findings beat eight vague ones.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        sub = task.payload.get("subquestion") or {}
        hints = sub.get("search_hints") or []
        hint_note = f"\nSearch queries worth trying: {'; '.join(hints)}" if hints else ""
        why = f"\nWhy the report needs it: {sub.get('why')}" if sub.get("why") else ""
        already = ctx.evidence()
        known = ""
        if already:
            known = (
                "\n\nSources other researchers have already stored, which you may cite "
                "without fetching again:\n"
                + "\n".join(
                    f"- [{row['ref']}] {truncate(row['title'] or row['url'], 90)} ({row['domain']})"
                    for row in already[:15]
                )
            )
        return (
            f"Overall brief, for context only:\n{ctx.brief}\n\n"
            f"YOUR QUESTION ({sub.get('id', 'Q?')}):\n{sub.get('question', ctx.brief)}"
            f"{why}{hint_note}{known}"
        )

    def final_schema(self, ctx: Any) -> str:
        return SCHEMA

    def research(self, ctx: Any, task: Any) -> dict[str, Any]:
        sub = task.payload.get("subquestion") or {}
        # Findings record which subquestion produced them, and the tool reads
        # it from the board rather than taking it as an argument, because the
        # model should not be able to misattribute its own work.
        ctx.board.set("_current_subquestion", sub.get("id", ""))

        before = len(ctx.board.findings)
        result = self.think(ctx, task)
        after = len(ctx.board.findings)

        output = result.output if isinstance(result.output, dict) else {}
        answer = str(output.get("answer") or "").strip()
        if not answer and isinstance(result.output, str):
            answer = result.output.strip()

        gaps = [str(g).strip() for g in (output.get("gaps") or []) if str(g).strip()]
        recorded = after - before

        if recorded == 0:
            # Worth remembering across runs: a question shape that reliably
            # yields nothing is a planning problem, not a research problem.
            gaps.append(f"No findings could be recorded for: {sub.get('question', '')}")

        return {
            "subquestion_id": sub.get("id", ""),
            "question": sub.get("question", ""),
            "answer": answer,
            "confidence": str(output.get("confidence") or "medium").lower(),
            "gaps": gaps[:4],
            "findings_recorded": recorded,
            "iterations": result.iterations,
            "stopped_because": result.stopped_because,
            "tools_used": sorted({c.name for c in result.tool_calls}),
        }
