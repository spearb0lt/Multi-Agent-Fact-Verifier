"""The flagship workflow: a brief in, a cited and verified report out.

                   +-------------+
                   |   Planner   |<---------------------+
                   +------+------+                      |
                          | subquestions                | gaps to close
                   +------v------+                      |
                   | Researcher  | x N, in parallel     |
                   +------+------+                      |
                          | findings                    |
                   +------v------+                      |
                   |   Analyst   |                      |
                   +------+------+                      |
                          | claims                      |
                   +------v------+                      |
                   | FactChecker | x N, in parallel     |
                   +------+------+                      |
                          | verdicts                    |
                   +------v------+                      |
                   | Supervisor  +----------------------+
                   +------+------+  research_more
                          | write
                   +------v------+
                   |   Writer    |<---------+
                   +------+------+          |
                          |                 |
                   +------v------+          |
                   |   Editor    |          | revise
                   +------+------+          |
                          |                 |
                   +------v------+          |
                   |   Critic    +----------+
                   +------+------+
                          | accept
                   +------v------+
                   |   Report    |
                   +-------------+

Two cycles, and they are the point. The Supervisor's cycle asks whether the
team gathered enough; the Critic's asks whether the team wrote it well enough.
Each is bounded by a counter on the blackboard rather than by trust, because an
agent asked whether it should try again will say yes indefinitely.

Only claims that passed verification reach the Writer. That single edge is what
the whole verification half of the graph is for.
"""
from __future__ import annotations

from typing import Any

from ..agents import (
    analyst,
    critic,
    editor,
    factchecker,
    planner,
    researcher,
    supervisor,
    writer,
)
from ..core.util import truncate
from ..kernel.contracts import ArtifactKind, EventKind, Message, NodeResult, Task
from ..kernel.graph import Graph, register

graph = Graph(
    name="research_report",
    entry="plan",
    description=(
        "Eight agents: a Planner decomposes the brief, Researchers gather sources in "
        "parallel, an Analyst consolidates claims, Fact Checkers verify them in "
        "parallel, a Supervisor decides whether to write or research further, then a "
        "Writer, an Editor and a Critic produce and hold the report to a standard."
    ),
)


# ---------------------------------------------------------------------- plan


@graph.node("plan", agent="Planner", label="Plan", description="Decompose the brief")
def plan_node(ctx: Any, task: Task) -> NodeResult:
    plan = planner.plan(ctx)
    round_number = int(ctx.board.get("research_rounds", 0)) + 1

    ctx.board.update(
        plan=plan,
        research_rounds=round_number,
        research_expected=len(plan["subquestions"]),
        research_done=0,
    )
    ctx.board.extend("subquestions", plan["subquestions"])
    if not ctx.board.get("title"):
        ctx.board.set("title", plan["title"])

    artifact = planner.artifact(
        ArtifactKind.PLAN.value,
        f"plan-r{round_number}",
        title=plan["title"],
        content=plan,
    )
    messages = [
        Message(
            sender="Planner",
            recipient="Researcher",
            topic=f"subquestion {sub['id']}",
            content=sub["question"],
        )
        for sub in plan["subquestions"]
    ]

    group = f"research-r{round_number}"
    return NodeResult(
        output={"title": plan["title"], "subquestions": len(plan["subquestions"])},
        action=f"planned {len(plan['subquestions'])} subquestion(s)",
        artifacts=[artifact],
        messages=messages,
        next=[
            Task(node="research", payload={"subquestion": sub}, group=group)
            for sub in plan["subquestions"]
        ],
    )


# ------------------------------------------------------------------ research


@graph.node(
    "research",
    agent="Researcher",
    label="Research",
    description="Answer one subquestion from real sources",
    parallel=True,
    retries=1,
)
def research_node(ctx: Any, task: Task) -> NodeResult:
    answer = researcher.research(ctx, task)
    ctx.board.append("research_answers", answer)

    sub = task.payload.get("subquestion") or {}
    done = ctx.board.counter("research_done")
    expected = int(ctx.board.get("research_expected", 1))

    messages = [
        Message(
            sender="Researcher",
            recipient="Analyst",
            topic=f"answer to {sub.get('id', '?')}",
            content=truncate(answer.get("answer", ""), 600),
        )
    ]

    # The last branch to finish carries the run forward. Counting on the
    # blackboard rather than comparing list lengths means a branch that failed
    # and was abandoned does not strand the run waiting for it.
    if done >= expected:
        ctx.board.set("research_done", 0)
        return NodeResult(
            output=answer, action="researched", messages=messages, next=[Task(node="analyse")]
        )
    return NodeResult(output=answer, action="researched", messages=messages)


# ------------------------------------------------------------------- analyse


@graph.node("analyse", agent="Analyst", label="Analyse", description="Consolidate into claims")
def analyse_node(ctx: Any, task: Task) -> NodeResult:
    analysis = analyst.analyse(ctx)
    claims = analysis["claims"]

    # A second round re-analyses every finding, old and new, which is right:
    # the claim set should reflect everything known. Carrying the earlier
    # verdicts across by claim text means only genuinely new claims are paid
    # for again.
    previous = {
        str(v.get("claim_text", "")): v for v in ctx.board.verdicts if v.get("claim_text")
    }
    carried = []
    for claim in claims:
        if claim["text"] in previous:
            verdict = dict(previous[claim["text"]])
            verdict["claim_id"] = claim["id"]
            carried.append(verdict)

    ctx.board.set("claims", claims)
    ctx.board.set("verdicts", carried)
    ctx.board.set("analysis", analysis)

    artifact = analyst.artifact(
        ArtifactKind.CLAIM.value,
        f"claims-r{ctx.board.get('research_rounds', 1)}",
        title=f"{len(claims)} claim(s)",
        content=analysis,
    )

    pending = [c for c in claims if c["text"] not in previous]
    if not pending:
        return NodeResult(
            output={"claims": len(claims), "reused_verdicts": len(carried)},
            action="analysed",
            artifacts=[artifact],
            next=[Task(node="supervise")],
        )

    ctx.board.update(verify_expected=len(pending), verify_done=0)
    group = f"verify-r{ctx.board.get('research_rounds', 1)}"
    return NodeResult(
        output={"claims": len(claims), "to_verify": len(pending), "reused_verdicts": len(carried)},
        action=f"made {len(claims)} claim(s)",
        artifacts=[artifact],
        messages=[
            Message(
                sender="Analyst",
                recipient="FactChecker",
                topic="claims to verify",
                content=f"{len(pending)} claim(s) need checking",
            )
        ],
        next=[
            Task(node="verify", payload={"claim": claim}, group=group) for claim in pending
        ],
    )


# -------------------------------------------------------------------- verify


@graph.node(
    "verify",
    agent="FactChecker",
    label="Verify",
    description="Rule on whether the sources establish one claim",
    parallel=True,
    retries=1,
)
def verify_node(ctx: Any, task: Task) -> NodeResult:
    claim = task.payload.get("claim") or {}
    verdict = factchecker.check(ctx, task)
    verdict["claim_text"] = claim.get("text", "")
    ctx.board.append("verdicts", verdict)

    done = ctx.board.counter("verify_done")
    expected = int(ctx.board.get("verify_expected", 1))

    messages = []
    if verdict["verdict"] in {"unsupported", "contradicted"}:
        messages.append(
            Message(
                sender="FactChecker",
                recipient="Writer",
                topic=f"rejected {claim.get('id', '?')}",
                content=f"Do not use: {truncate(claim.get('text', ''), 200)}. "
                f"{verdict.get('reasoning', '')}",
            )
        )

    if done >= expected:
        ctx.board.set("verify_done", 0)
        return NodeResult(
            output=verdict, action="verified", messages=messages, next=[Task(node="supervise")]
        )
    return NodeResult(output=verdict, action="verified", messages=messages)


# ----------------------------------------------------------------- supervise


@graph.node(
    "supervise",
    agent="Supervisor",
    label="Supervise",
    description="Write now, or research the remaining gaps",
)
def supervise_node(ctx: Any, task: Task) -> NodeResult:
    decision = supervisor.decide(ctx)
    ctx.board.append("decisions", decision)

    ctx.bus.emit(
        EventKind.LOG,
        f"Supervisor chose to {decision['decision'].replace('_', ' ')}: {decision['reasoning']}",
        agent="Supervisor",
        decision=decision["decision"],
        forced=decision["forced"],
    )

    if decision["decision"] == "research_more":
        # The gaps go on the board and the Planner reads them, so the next
        # round is aimed at what is missing rather than repeating round one.
        ctx.board.set("gaps", decision["gaps"])
        ctx.board.set("subquestions", [])
        return NodeResult(
            output=decision,
            action="sent the team back out",
            messages=[
                Message(
                    sender="Supervisor",
                    recipient="Planner",
                    topic="gaps to close",
                    content=decision["gaps"],
                )
            ],
            next=[Task(node="plan")],
        )

    return NodeResult(
        output=decision,
        action="approved writing",
        messages=[
            Message(
                sender="Supervisor",
                recipient="Writer",
                topic="go ahead",
                content=f"{decision['supported']} verified claim(s) to write from.",
            )
        ],
        next=[Task(node="write")],
    )


# --------------------------------------------------------------------- write


@graph.node("write", agent="Writer", label="Write", description="Draft from verified claims")
def write_node(ctx: Any, task: Task) -> NodeResult:
    draft = writer.draft(ctx)
    ctx.board.set("draft", draft)

    artifact = writer.artifact(
        ArtifactKind.DRAFT.value,
        "draft",
        title=f"Draft, revision {draft['revision']}",
        body=draft["markdown"],
        content={"words": draft["words"], "claims_used": draft["claims_used"]},
    )
    return NodeResult(
        output={"words": draft["words"], "revision": draft["revision"]},
        action=f"drafted {draft['words']} words",
        artifacts=[artifact],
        next=[Task(node="edit")],
    )


@graph.node("edit", agent="Editor", label="Edit", description="Tighten without changing meaning")
def edit_node(ctx: Any, task: Task) -> NodeResult:
    edited = editor.edit(ctx)
    if edited["markdown"]:
        ctx.board.put("draft", "markdown", edited["markdown"])
        ctx.board.put("draft", "words", edited.get("words", 0))

    artifact = editor.artifact(
        ArtifactKind.DRAFT.value,
        "edited",
        title=f"Edited, revision {ctx.board.revision}",
        body=edited["markdown"],
        content={k: v for k, v in edited.items() if k != "markdown"},
    )
    return NodeResult(
        output={"changed": edited["changed"], "dropped": edited.get("dropped_citations", [])},
        action="edited",
        artifacts=[artifact],
        next=[Task(node="critique")],
    )


@graph.node("critique", agent="Critic", label="Critique", description="Accept, or send back")
def critique_node(ctx: Any, task: Task) -> NodeResult:
    verdict = critic.critique(ctx)
    ctx.board.append("critiques", verdict)

    artifact = critic.artifact(
        ArtifactKind.CRITIQUE.value,
        f"critique-r{verdict['revision']}",
        title=f"Scored {verdict['score']} out of 100",
        content=verdict,
    )

    revision = ctx.board.revision
    max_revisions = int(ctx.option("max_revisions", 2))
    accept = verdict["verdict"] == "accept"

    if not accept and revision >= max_revisions:
        ctx.bus.log(
            f"The Critic still wants changes, but {max_revisions} revision(s) have been "
            f"used, so the report is finalised as it stands.",
            level="warning",
        )
        accept = True
    if not accept and ctx.budget.critical:
        ctx.bus.log(
            "The Critic wants changes, but the budget is nearly spent, so the report is "
            "finalised as it stands.",
            level="warning",
        )
        accept = True

    if accept:
        return NodeResult(
            output=verdict,
            action=f"accepted at {verdict['score']}",
            artifacts=[artifact],
            messages=[
                Message(
                    sender="Critic",
                    recipient="*",
                    topic="accepted",
                    content=f"Scored {verdict['score']} out of 100.",
                )
            ],
            next=[Task(node="finalise")],
        )

    ctx.board.set("revision", revision + 1)
    return NodeResult(
        output=verdict,
        action=f"sent back at {verdict['score']}",
        artifacts=[artifact],
        messages=[
            Message(
                sender="Critic",
                recipient="Writer",
                topic="revise",
                content=[i["issue"] for i in verdict["issues"]],
            )
        ],
        next=[Task(node="write")],
    )


# ------------------------------------------------------------------ finalise


@graph.node("finalise", agent="Editor", label="Report", description="Attach sources and finish")
def finalise_node(ctx: Any, task: Task) -> NodeResult:
    from .render import build_report

    report = build_report(ctx)
    ctx.board.set("report", report)

    artifact = editor.artifact(
        ArtifactKind.REPORT.value,
        "report",
        title=report["title"],
        body=report["markdown"],
        content={k: v for k, v in report.items() if k != "markdown"},
    )
    ctx.bus.emit(
        EventKind.ARTIFACT,
        f"Report finished: {report['words']} words, {len(report['sources'])} source(s)",
        agent="Editor",
        kind="report",
    )
    return NodeResult(
        output={"words": report["words"], "sources": len(report["sources"])},
        action="finalised",
        artifacts=[artifact],
        done=True,
    )


graph.edge("plan", "research")
graph.edge("research", "analyse")
graph.edge("analyse", "verify")
graph.edge("analyse", "supervise", "all verified")
graph.edge("verify", "supervise")
graph.edge("supervise", "plan", "research_more")
graph.edge("supervise", "write", "write")
graph.edge("write", "edit")
graph.edge("edit", "critique")
graph.edge("critique", "write", "revise")
graph.edge("critique", "finalise", "accept")

register(graph)
