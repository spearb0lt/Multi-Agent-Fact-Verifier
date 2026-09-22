"""The Editor: make it read as one document, without changing what it says.

The Editor's constraint is narrower than it sounds and it is the reason the
role exists separately from the Writer. It may cut, reorder, retitle, tighten
and fix the prose. It may not add a fact, remove a citation, or change what a
sentence asserts. A model asked to "improve" a report will happily strengthen a
hedged sentence into a confident one, and that is the single easiest way for an
unverified assertion to enter a document that has just been verified.

So the citation check afterwards is mechanical rather than a matter of trust:
every reference in the edited text is compared against the references that were
in the draft, and an invented one is removed.
"""
from __future__ import annotations

import re
from typing import Any

from ..core.util import truncate
from .base import HOUSE_STYLE, Agent

CITATION_RE = re.compile(r"\[(S\d+)\]")


class Editor(Agent):
    role = "Editor"
    goal = "Tighten the draft into a publishable document without changing its claims."
    temperature = 0.25
    max_tokens = 4000

    def system_prompt(self, ctx: Any) -> str:
        return f"""You are the Editor. You are given a draft and you return the edited
markdown, and nothing else. No commentary, no notes about what you changed.

You MAY:
- cut repetition, filler and anything that says nothing,
- reorder sentences and sections so the argument runs properly,
- tighten wording and fix grammar, punctuation and heading levels,
- merge two paragraphs saying the same thing.

You MUST NOT:
- add any fact, figure, date, name or quotation,
- remove or alter a source reference in square brackets,
- make a hedged statement more confident, or a specific one vaguer,
- change what any sentence asserts,
- add a sources list. It is added afterwards.

Keep every [S1] style reference attached to the sentence it belongs to. If you
merge two cited sentences, carry both references.

{HOUSE_STYLE}"""

    def goal_prompt(self, ctx: Any, task: Any) -> str:
        return self._prompt(ctx)

    def _prompt(self, ctx: Any) -> str:
        draft = ctx.board.draft.get("markdown", "")
        return (
            f"Brief, for context:\n{ctx.brief}\n\n"
            f"Edit this draft and return the full edited markdown:\n\n{draft}"
        )

    def edit(self, ctx: Any) -> dict[str, Any]:
        draft = ctx.board.draft.get("markdown", "")
        if not draft.strip():
            return {"markdown": "", "changed": False, "dropped_citations": []}

        edited = self.write(ctx, prompt=self._prompt(ctx), purpose="editor.edit").strip()

        # A model that returned commentary, an apology or a fragment has not
        # edited anything, and shipping that would lose the whole report.
        if len(edited) < len(draft) * 0.4 or not edited.lstrip().startswith("#"):
            ctx.bus.log(
                "The Editor's output did not look like the edited report, so the draft "
                "was kept unchanged.",
                level="warning",
            )
            return {"markdown": draft, "changed": False, "dropped_citations": []}

        allowed = set(CITATION_RE.findall(draft))
        invented = sorted(set(CITATION_RE.findall(edited)) - allowed)
        if invented:
            # An editor is not allowed to cite a source the writer did not.
            edited = CITATION_RE.sub(
                lambda m: "" if m.group(1) in invented else m.group(0), edited
            )
            ctx.bus.log(
                f"The Editor introduced references that were not in the draft "
                f"({', '.join(invented)}), so they were removed.",
                level="warning",
                invented=invented,
            )

        kept = set(CITATION_RE.findall(edited))
        dropped = sorted(allowed - kept)
        if dropped:
            ctx.bus.log(
                f"Editing dropped these references: {', '.join(dropped)}.",
                level="warning",
                dropped=dropped,
            )

        return {
            "markdown": edited,
            "changed": edited != draft,
            "dropped_citations": dropped,
            "invented_citations": invented,
            "words": len(edited.split()),
            "summary": truncate(edited, 300),
        }
