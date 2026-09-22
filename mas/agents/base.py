"""What every role has in common.

An agent here is a role with a goal, a set of tools and a contract about what
it puts on the blackboard. It is deliberately not a subclass hierarchy with
much behaviour in it: the interesting differences between a Researcher and a
Critic are in their prompts and their tools, not in their control flow, and
burying that in inheritance would hide the part a reader actually needs to
understand.

Two things are shared and worth having in one place. The house style, because
a report written by four models in three tiers has to read as one document.
And the announcement protocol, because an agent that finishes without telling
anyone leaves the Supervisor guessing.
"""
from __future__ import annotations

from typing import Any

from ..core.util import truncate
from ..kernel.contracts import AgentResult, Artifact, Message
from ..kernel.react import think

# Applied to every role. The dash rule is enforced again in the provider layer,
# because models ignore it roughly half the time and a non ASCII dash breaks a
# Windows console as well as looking wrong.
HOUSE_STYLE = """Style rules, which apply to everything you write:
- Never use an em dash or an en dash. Use a comma, a colon, a full stop, or the
  word "to" for a range.
- Plain, specific, unhedged prose. No throat clearing, no "it is important to
  note", no summary of what you are about to say.
- Never invent a fact, a figure, a date, a name or a quotation. If the sources
  do not settle something, say that they do not.
- Every factual sentence carries its source reference in square brackets, like
  [S3]. A sentence with a fact and no reference is not acceptable."""


class Agent:
    """One role in the team."""

    role: str = ""
    goal: str = ""
    tools: tuple[str, ...] = ()
    # The reason and act bound for this role specifically. Zero defers to the
    # run's setting, which is what most roles want.
    max_iterations: int = 0
    temperature: float = 0.2
    max_tokens: int = 2048

    def system_prompt(self, ctx: Any) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def goal_prompt(self, ctx: Any, task: Any) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def final_schema(self, ctx: Any) -> str:
        return ""

    # ------------------------------------------------------------------ work

    def think(self, ctx: Any, task: Any) -> AgentResult:
        """Run this role's reason and act loop."""
        ctx.bus.bind(agent=self.role)
        return think(
            ctx,
            role=self.role,
            goal=self.goal_prompt(ctx, task),
            system=self.system_prompt(ctx),
            tools=self.tools,
            max_iterations=self.max_iterations,
            final_schema=self.final_schema(ctx),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def ask(
        self,
        ctx: Any,
        *,
        prompt: str,
        schema_hint: str = "",
        expect_list: bool = False,
        default: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
    ) -> Any:
        """One shot structured reasoning, for a role that needs no tools.

        Most roles downstream of research are judgement over what is already on
        the blackboard. Giving them a tool loop would let them wander back onto
        the web to avoid a hard call, which is exactly the failure the division
        of labour exists to prevent.
        """
        ctx.bus.bind(agent=self.role)
        return ctx.llm.complete_json(
            prompt,
            role=self.role,
            system=self.system_prompt(ctx),
            schema_hint=schema_hint,
            expect_list=expect_list,
            default=default,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            purpose=purpose or f"{self.role.lower()}.reason",
        )

    def write(
        self,
        ctx: Any,
        *,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        purpose: str = "",
    ) -> str:
        """Free prose, for the roles whose product is the document itself."""
        ctx.bus.bind(agent=self.role)
        return ctx.llm.complete(
            prompt,
            role=self.role,
            system=self.system_prompt(ctx),
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            purpose=purpose or f"{self.role.lower()}.write",
        )

    # --------------------------------------------------------------- reporting

    def say(self, ctx: Any, content: Any, *, to: str = "*", topic: str = "") -> Message:
        message = Message(sender=self.role, recipient=to, topic=topic, content=content)
        ctx.send(message)
        return message

    def artifact(
        self,
        kind: str,
        key: str,
        *,
        title: str = "",
        body: str = "",
        content: dict[str, Any] | None = None,
        parent_key: str = "",
    ) -> Artifact:
        return Artifact(
            kind=kind,
            key=key,
            title=title,
            body=body,
            content=content or {},
            parent_key=parent_key,
            produced_by=self.role,
        )


def evidence_digest(ctx: Any, refs: tuple[str, ...] = (), *, chars: int = 1200) -> str:
    """The sources, numbered, as a model reads them.

    Numbered by the reference the evidence table already assigned rather than
    by position in this list. A model told to cite [2] when the source is S7
    produces a report whose citations point at the wrong things, and that is
    not recoverable after the fact.
    """
    rows = ctx.evidence(refs=refs)
    if not rows:
        return "No sources have been gathered."
    parts = []
    for row in rows:
        published = f", {row['published_at'][:10]}" if row.get("published_at") else ""
        parts.append(
            f"[{row['ref']}] {row['title'] or row['url']} ({row['domain']}{published})\n"
            f"{truncate(row['body'] or row['snippet'], chars)}"
        )
    return "\n\n".join(parts)


def findings_digest(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No findings were recorded."
    return "\n".join(
        f"- {f.get('id', '?')} [{', '.join(f.get('sources', []))}] "
        f"({f.get('confidence', 'medium')}) {f.get('statement', '')}"
        + (f"  Note: {f['note']}" if f.get("note") else "")
        for f in findings
    )


def claims_digest(claims: list[dict[str, Any]], *, with_verdicts: bool = False) -> str:
    if not claims:
        return "No claims were made."
    lines = []
    for claim in claims:
        line = f"- {claim.get('id', '?')} [{', '.join(claim.get('sources', []))}] {claim.get('text', '')}"
        if with_verdicts and claim.get("verdict"):
            verdict = claim["verdict"]
            line += (
                f"\n    verdict: {verdict.get('verdict', '?')}"
                f" ({verdict.get('confidence', '?')} confidence)"
                f" across {verdict.get('independent_domains', 0)} domain(s)"
            )
            if verdict.get("reasoning"):
                line += f"\n    reason: {truncate(verdict['reasoning'], 300)}"
        lines.append(line)
    return "\n".join(lines)
