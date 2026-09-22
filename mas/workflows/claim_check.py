"""Check one claim, rather than research a topic.

                   +-------------+
                   |   Planner   |<-------------+
                   +------+------+              |
                          | angles              | one more round
             +------------+------------+        |
             | supporting, refuting,   |        |
             | and background          |        |
                   +------v------+              |
                   | Researcher  | x 3 parallel |
                   +------+------+              |
                          | evidence            |
                   +------v------+              |
                   | FactChecker +--------------+
                   +------+------+  not enough yet
                          | verdict
                   +------v------+
                   |   Writer    |
                   +------+------+
                          |
                   +------v------+
                   |   Report    |
                   +-------------+

The point of this workflow is not that it is elaborate. It is that it exists:
the same kernel, the same roles, the same tools and the same durable run loop,
arranged differently and registered under another name. A framework that only
ever runs one graph has not demonstrated that it is a framework.

It is also the cheaper thing to run. One claim needs a fraction of the tokens a
whole report does, which makes it the right shape for checking something
quickly, and the right shape for a demonstration on a budget.

The one design decision worth naming: the researchers are deliberately sent to
look for refuting evidence as well as supporting evidence, as a separate task
with its own agent. Asking one agent to "research this claim" reliably produces
a search for confirmation, because that is what the claim's own wording
suggests as a query. Splitting the angles is what stops the answer being decided
by the phrasing of the question.
"""
from __future__ import annotations

from typing import Any

from ..agents import factchecker, planner, researcher, writer
from ..core.util import truncate
from ..kernel.contracts import ArtifactKind, Message, NodeResult, Task
from ..kernel.graph import Graph, register

graph = Graph(
    name="claim_check",
    entry="frame",
    description=(
        "Takes one specific claim rather than a topic. Researchers gather evidence "
        "for it, against it and around it in parallel, a Fact Checker weighs all of "
        "it and returns a verdict with its reasoning, and a Writer explains the "
        "finding with citations."
    ),
)

# What the researchers are sent to look for. Separate angles rather than one
# task, because a single agent asked about a claim searches for the claim.
ANGLES = (
    {
        "id": "A1",
        "angle": "supporting",
        "instruction": (
            "Find the strongest evidence that this claim is TRUE. Look for the "
            "primary source: the study, the filing, the official statement."
        ),
    },
    {
        "id": "A2",
        "angle": "refuting",
        "instruction": (
            "Find the strongest evidence that this claim is FALSE, overstated, "
            "out of date or misattributed. Look for corrections, retractions, "
            "fact checks and contrary findings."
        ),
    },
    {
        "id": "A3",
        "angle": "context",
        "instruction": (
            "Find what the claim leaves out: the conditions under which it holds, "
            "who says it, when it was said, and what has changed since."
        ),
    },
)

VERDICT_SCHEMA = """{
  "verdict": "true | mostly_true | mixed | mostly_false | false | unverifiable",
  "confidence": "high | medium | low",
  "reasoning": "two or three sentences on what the evidence establishes, with [S1] references",
  "key_sources": ["S1", "S3"],
  "what_would_settle_it": "what evidence would resolve this, if it is not resolved",
  "enough_evidence": true
}"""


@graph.node("frame", agent="Planner", label="Frame", description="Turn the claim into search angles")
def frame_node(ctx: Any, task: Task) -> NodeResult:
    round_number = int(ctx.board.get("rounds", 0)) + 1
    gaps = ctx.board.items("gaps")

    # The angles are fixed rather than generated. What to look for when
    # checking a claim does not vary by claim, and spending a strong model call
    # to rediscover "look for evidence against it" every time is waste.
    subquestions = []
    for angle in ANGLES:
        question = f"{angle['instruction']}\n\nTHE CLAIM: {ctx.brief}"
        if gaps:
            question += "\n\nA previous round still needs: " + "; ".join(str(g) for g in gaps)
        subquestions.append(
            {
                "id": f"{angle['id']}r{round_number}",
                "question": question,
                "why": f"The {angle['angle']} side of the claim.",
                "priority": "high",
                "search_hints": [truncate(ctx.brief, 110)],
                "angle": angle["angle"],
            }
        )

    ctx.board.update(
        rounds=round_number,
        research_expected=len(subquestions),
        research_done=0,
        claim=ctx.brief,
    )
    ctx.board.extend("subquestions", subquestions)

    return NodeResult(
        output={"angles": [a["angle"] for a in ANGLES], "round": round_number},
        action=f"framed {len(subquestions)} angle(s)",
        artifacts=[
            planner.artifact(
                ArtifactKind.PLAN.value,
                f"angles-r{round_number}",
                title=f"Checking: {truncate(ctx.brief, 70)}",
                content={"angles": subquestions, "round": round_number},
            )
        ],
        messages=[
            Message(
                sender="Planner",
                recipient="Researcher",
                topic=f"{sub['angle']} evidence",
                content=truncate(sub["question"], 200),
            )
            for sub in subquestions
        ],
        next=[
            Task(node="gather", payload={"subquestion": sub}, group=f"gather-r{round_number}")
            for sub in subquestions
        ],
    )


@graph.node(
    "gather",
    agent="Researcher",
    label="Gather",
    description="Find evidence from one angle",
    parallel=True,
    retries=1,
)
def gather_node(ctx: Any, task: Task) -> NodeResult:
    answer = researcher.research(ctx, task)
    sub = task.payload.get("subquestion") or {}
    answer["angle"] = sub.get("angle", "")
    ctx.board.append("research_answers", answer)

    done = ctx.board.counter("research_done")
    expected = int(ctx.board.get("research_expected", 1))

    message = Message(
        sender="Researcher",
        recipient="FactChecker",
        topic=f"{sub.get('angle', 'evidence')} evidence",
        content=truncate(answer.get("answer", ""), 500),
    )

    if done >= expected:
        ctx.board.set("research_done", 0)
        return NodeResult(
            output=answer, action="gathered", messages=[message], next=[Task(node="adjudicate")]
        )
    return NodeResult(output=answer, action="gathered", messages=[message])


@graph.node(
    "adjudicate",
    agent="FactChecker",
    label="Adjudicate",
    description="Weigh the evidence and rule on the claim",
)
def adjudicate_node(ctx: Any, task: Task) -> NodeResult:
    from ..agents.base import evidence_digest, findings_digest

    findings = ctx.board.findings
    by_angle = {}
    for answer in ctx.board.items("research_answers"):
        by_angle.setdefault(answer.get("angle", "other"), []).append(
            answer.get("answer", "")
        )

    angle_note = "\n\n".join(
        f"What the {angle} search found:\n" + "\n".join(f"- {a}" for a in answers if a)
        for angle, answers in by_angle.items()
    )

    prompt = (
        f"THE CLAIM UNDER CHECK:\n{ctx.brief}\n\n"
        f"{angle_note}\n\n"
        f"Findings recorded, with their sources:\n{findings_digest(findings)}\n\n"
        f"The sources themselves:\n{evidence_digest(ctx, chars=2200)}\n\n"
        f"Rule on the claim."
    )

    payload = factchecker.ask(
        ctx, prompt=prompt, schema_hint=VERDICT_SCHEMA, default={}, purpose="claim.adjudicate"
    )
    if not isinstance(payload, dict):
        payload = {}

    allowed = {"true", "mostly_true", "mixed", "mostly_false", "false", "unverifiable"}
    verdict = str(payload.get("verdict") or "").strip().lower().replace(" ", "_")
    if verdict not in allowed:
        # A checker that produced nothing usable must not read as a pass.
        verdict = "unverifiable"

    known = {row["ref"] for row in ctx.evidence()}
    sources = [
        ref
        for ref in (str(s).strip().upper() for s in payload.get("key_sources") or [])
        if ref in known
    ]
    index = ctx.evidence_index()
    domains = sorted({index[ref]["domain"] for ref in sources if ref in index})

    # Confidence is capped by how many independent outlets actually stand
    # behind the ruling. A model handed a single article will call a claim
    # false with high confidence, and on one source about one subpopulation
    # that is not a judgement anybody should act on. The model may be less
    # confident than the evidence allows; it may not be more.
    stated = str(payload.get("confidence") or "low").lower()
    if stated not in {"high", "medium", "low"}:
        stated = "low"
    ceiling = "high" if len(domains) >= int(ctx.option("corroboration_min", 2)) else "medium"
    if len(domains) <= 1:
        ceiling = "low"
    order = {"low": 0, "medium": 1, "high": 2}
    confidence = stated if order[stated] <= order[ceiling] else ceiling
    if confidence != stated:
        ctx.bus.log(
            f"The Fact Checker claimed {stated} confidence on {len(domains)} independent "
            f"outlet(s), so it was capped at {confidence}.",
            level="warning",
            agent="FactChecker",
        )

    ruling = {
        "verdict": verdict,
        "confidence": confidence,
        "stated_confidence": stated,
        "reasoning": truncate(str(payload.get("reasoning") or "").strip(), 900),
        "sources": sources,
        "domains": domains,
        "independent_domains": len(domains),
        "corroborated": len(domains) >= int(ctx.option("corroboration_min", 2)),
        "what_would_settle_it": truncate(
            str(payload.get("what_would_settle_it") or "").strip(), 400
        ),
        "round": int(ctx.board.get("rounds", 1)),
    }
    ctx.board.set("ruling", ruling)

    artifact = factchecker.artifact(
        ArtifactKind.VERDICT.value,
        "ruling",
        title=f"{verdict.replace('_', ' ')} ({ruling['confidence']} confidence)",
        content=ruling,
    )

    # One more round, but only when another round could plausibly change the
    # answer and the budget can pay for it. "Unverifiable with no evidence at
    # all" is worth another look; "unverifiable because the question is not
    # empirical" is not, and the difference is whether anything was found.
    #
    # Whether the evidence is sufficient is not left to the model alone. It
    # will declare one local news article enough and then rule with high
    # confidence, so a ruling that rests on fewer independent outlets than the
    # corroboration threshold triggers another round regardless of what it
    # said, exactly as the confidence cap overrides its own certainty.
    rounds = int(ctx.board.get("rounds", 1))
    max_rounds = int(ctx.option("max_research_rounds", 2))
    minimum = int(ctx.option("corroboration_min", 2))
    enough = bool(payload.get("enough_evidence", True)) and len(domains) >= minimum
    if not enough and rounds < max_rounds and not ctx.budget.critical and findings:
        gap = ruling["what_would_settle_it"] or "more direct evidence"
        if len(domains) < minimum:
            gap = (
                f"{gap} The ruling currently rests on {len(domains)} outlet(s): find "
                f"corroboration from a different publisher, ideally a primary source."
            )
        ctx.board.set("gaps", [gap])
        ctx.board.set("subquestions", [])
        ctx.bus.log(
            f"The evidence does not settle the claim yet ({len(domains)} independent "
            f"outlet(s)), so one more round is run.",
            level="info",
        )
        return NodeResult(
            output=ruling,
            action="needs more evidence",
            artifacts=[artifact],
            next=[Task(node="frame")],
        )

    return NodeResult(
        output=ruling,
        action=f"ruled {verdict}",
        artifacts=[artifact],
        messages=[
            Message(
                sender="FactChecker",
                recipient="Writer",
                topic="ruling",
                content=f"{verdict} ({ruling['confidence']} confidence)",
            )
        ],
        next=[Task(node="explain")],
    )


@graph.node("explain", agent="Writer", label="Explain", description="Write the finding up")
def explain_node(ctx: Any, task: Task) -> NodeResult:
    from ..agents.base import HOUSE_STYLE, evidence_digest, findings_digest

    ruling = ctx.board.mapping("ruling")
    refs = tuple(ruling.get("sources") or ())

    prompt = (
        f"THE CLAIM:\n{ctx.brief}\n\n"
        f"The Fact Checker's ruling: {ruling.get('verdict')} "
        f"({ruling.get('confidence')} confidence, across "
        f"{ruling.get('independent_domains', 0)} independent outlet(s))\n"
        f"Its reasoning: {ruling.get('reasoning')}\n\n"
        f"The findings behind it:\n{findings_digest(ctx.board.findings)}\n\n"
        f"The sources:\n{evidence_digest(ctx, refs or (), chars=1200)}\n\n"
        f"Write the explanation."
    )

    system = f"""You explain a fact check to a reader who has not seen any of the work.

Structure, in markdown:

# <the claim, restated plainly as a heading>

## Verdict
The ruling in one sentence, then one or two sentences on what the evidence
actually shows. Cited.

## What the evidence says
Three to six bullets, every one cited. Include the evidence against, where
there is any: a check that only reports what agrees with its own verdict is
not a check.

## What is not settled
What would resolve the remaining doubt. Write "Nothing significant" only if
the evidence really does settle it.

Do not restate the verdict label more than once. Do not include a sources
list: it is added afterwards.

{HOUSE_STYLE}"""

    markdown = ctx.llm.complete(
        prompt, role="Writer", system=system, temperature=0.3, max_tokens=2000,
        purpose="claim.explain",
    ).strip()

    ctx.board.set("draft", {"markdown": markdown, "revision": 0, "words": len(markdown.split())})
    return NodeResult(
        output={"words": len(markdown.split())},
        action="explained",
        artifacts=[
            writer.artifact(
                ArtifactKind.DRAFT.value, "draft", title="Explanation", body=markdown
            )
        ],
        next=[Task(node="publish")],
    )


@graph.node("publish", agent="Writer", label="Report", description="Attach sources and finish")
def publish_node(ctx: Any, task: Task) -> NodeResult:
    from .render import build_report

    report = build_report(ctx)
    ruling = ctx.board.mapping("ruling")
    report["verdict"] = ruling.get("verdict")
    report["confidence"] = ruling.get("confidence")
    report["title"] = f"Fact check: {truncate(ctx.brief, 70)}"
    ctx.board.set("report", report)

    return NodeResult(
        output={
            "verdict": ruling.get("verdict"),
            "words": report["words"],
            "sources": len(report["sources"]),
        },
        action=f"published: {ruling.get('verdict')}",
        artifacts=[
            writer.artifact(
                ArtifactKind.REPORT.value,
                "report",
                title=report["title"],
                body=report["markdown"],
                content={k: v for k, v in report.items() if k != "markdown"},
            )
        ],
        done=True,
    )


graph.edge("frame", "gather")
graph.edge("gather", "adjudicate")
graph.edge("adjudicate", "frame", "not enough yet")
graph.edge("adjudicate", "explain", "ruled")
graph.edge("explain", "publish")

register(graph)
