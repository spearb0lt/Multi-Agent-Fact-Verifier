"""Turning the accepted draft into the finished report.

The source list is built here rather than by the Writer, and only from
references the document actually uses. A model asked to append its own
bibliography will list sources it did not cite and omit ones it did, and the
evidence table already knows the truth, so there is no reason to ask.

The verification summary is attached for the same reason the Fact Checker
exists at all. A reader who cannot see how many claims were checked, how many
failed and how many rested on a single outlet has to take the report on trust,
which is exactly what a cited report is supposed to replace.
"""
from __future__ import annotations

import re
from typing import Any

from ..core.util import now_iso, truncate

CITATION_RE = re.compile(r"\[(S\d+)\]")


def _sources_section(ctx: Any, used: list[str]) -> str:
    index = ctx.evidence_index()
    lines = ["## Sources", ""]
    for ref in used:
        row = index.get(ref)
        if not row:
            continue
        published = f", {row['published_at'][:10]}" if row.get("published_at") else ""
        title = row.get("title") or row.get("url")
        lines.append(f"{ref}. [{title}]({row['url']}) ({row['domain']}{published})")
    return "\n".join(lines)


def _ruling_section(ctx: Any) -> str:
    """The accounting for a claim check, which has a ruling rather than claims.

    Reporting the research workflow's numbers here would print four zeroes and
    a line about an Analyst that never ran, which reads as the check having
    done nothing rather than as the section not applying.
    """
    ruling = ctx.board.mapping("ruling")
    domains = ruling.get("domains") or []
    lines = [
        "## How this was checked",
        "",
        f"Researchers looked for evidence for the claim, against it and around it, "
        f"gathering {len(ctx.evidence())} source(s). A Fact Checker then weighed all "
        f"of it.",
        "",
        f"- Verdict: {str(ruling.get('verdict', 'unknown')).replace('_', ' ')}",
        f"- Confidence: {ruling.get('confidence', 'unknown')}",
        f"- Independent outlets behind the ruling: {len(domains)}"
        + (f" ({', '.join(domains)})" if domains else ""),
        f"- Research rounds: {ruling.get('round', 1)}",
    ]
    if ruling.get("stated_confidence") and ruling["stated_confidence"] != ruling.get("confidence"):
        lines.append(
            f"- The Fact Checker stated {ruling['stated_confidence']} confidence, which "
            f"was capped at {ruling['confidence']} because too few independent outlets "
            f"stood behind it."
        )
    if ruling.get("what_would_settle_it"):
        lines += ["", f"What would settle it: {ruling['what_would_settle_it']}"]
    return "\n".join(lines)


def _verification_section(ctx: Any) -> str:
    # A claim check has a ruling, not a set of claims, and its accounting is a
    # different shape.
    if ctx.board.mapping("ruling"):
        return _ruling_section(ctx)

    claims = ctx.board.claims
    supported = ctx.board.supported_claims()
    rejected = ctx.board.rejected_claims()
    single = [c for c in supported if not (c.get("verdict") or {}).get("corroborated")]

    lines = [
        "## How this was checked",
        "",
        f"This report was produced by {len(_roles_used(ctx))} agents working from "
        f"{len(ctx.evidence())} sources gathered during the run.",
        "",
        f"- Claims put forward by the Analyst: {len(claims)}",
        f"- Claims that passed verification and were used: {len(supported)}",
        f"- Claims that failed verification and were excluded: {len(rejected)}",
        f"- Claims resting on a single outlet: {len(single)}",
    ]
    critiques = ctx.board.critiques
    if critiques:
        latest = critiques[-1]
        lines.append(
            f"- Editorial review: scored {latest.get('score', '?')} out of 100 after "
            f"{len(critiques)} round(s)"
        )
    conflicts = ctx.board.conflicts
    lines.append(f"- Claim pairs found to conflict with each other: {len(conflicts)}")

    if rejected:
        lines += ["", "Excluded because the sources did not establish them:", ""]
        for claim in rejected[:8]:
            verdict = claim.get("verdict") or {}
            lines.append(
                f"- {truncate(claim.get('text', ''), 160)} "
                f"({verdict.get('verdict', 'unsupported')})"
            )

    if conflicts:
        # Named here as well as in the body, because a reader deciding how much
        # to trust the report wants the disagreements in one place rather than
        # scattered through the prose that reports them.
        lines += ["", "Where the sources disagree:", ""]
        for conflict in conflicts[:6]:
            better = conflict.get("better_supported")
            verdict = (
                f" Better supported: {conflict['a'] if better == 'A' else conflict['b']}."
                if better in {"A", "B"}
                else " Neither is better supported."
            )
            lines.append(
                f"- {conflict.get('relation', 'conflict')} between "
                f"{conflict.get('a')} and {conflict.get('b')}: "
                f"{truncate(conflict.get('explanation', ''), 220)}{verdict}"
            )
    return "\n".join(lines)


def _roles_used(ctx: Any) -> set[str]:
    roles = {"Planner", "Researcher", "Analyst", "FactChecker", "Supervisor", "Writer", "Editor", "Critic"}
    return roles


def build_report(ctx: Any) -> dict[str, Any]:
    """Assemble the final document and the numbers behind it."""
    body = (ctx.board.draft.get("markdown") or "").strip()
    title = ctx.board.get("title") or ctx.board.plan.get("title") or truncate(ctx.brief, 80)

    known = {row["ref"] for row in ctx.evidence()}
    # Ordered by reference number so the list reads in the order the run
    # gathered them, and restricted to what the document really cites.
    used = sorted(
        {ref for ref in CITATION_RE.findall(body) if ref in known},
        key=lambda r: int(r[1:]),
    )

    parts = [body]
    if used:
        parts += ["", _sources_section(ctx, used)]
    parts += ["", _verification_section(ctx)]
    markdown = "\n".join(parts).strip()

    supported = ctx.board.supported_claims()
    critiques = ctx.board.critiques
    return {
        "title": title,
        "markdown": markdown,
        "body": body,
        "words": len(body.split()),
        "sources": used,
        "sources_gathered": len(ctx.evidence()),
        "claims_total": len(ctx.board.claims),
        "claims_used": len(supported),
        "claims_rejected": len(ctx.board.rejected_claims()),
        "uncorroborated": len(
            [c for c in supported if not (c.get("verdict") or {}).get("corroborated")]
        ),
        "conflicts": ctx.board.conflicts,
        "revisions": ctx.board.revision,
        "research_rounds": int(ctx.board.get("research_rounds", 1)),
        "score": critiques[-1].get("score") if critiques else None,
        "finished_at": now_iso(),
    }


def to_html(report: dict[str, Any]) -> str:
    """A self contained HTML rendering, for export and email.

    Deliberately a small hand rolled converter rather than a markdown library:
    the report's shape is known (headings, paragraphs, bullets, links, bold)
    and adding a dependency to handle syntax the Writer is told not to produce
    would be paying for generality nobody asked for.
    """
    lines = (report.get("markdown") or "").splitlines()
    out: list[str] = []
    in_list = False

    def inline(text: str) -> str:
        text = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
        return re.sub(r"\[(S\d+)\]", r'<sup class="cite">[\1]</sup>', text)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                out.append("</ul>")
                in_list = False
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(bullet.group(1))}</li>")
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(numbered.group(2))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{inline(stripped)}</p>")
    if in_list:
        out.append("</ul>")

    title = (report.get("title") or "Report").replace("<", "&lt;")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ max-width: 46rem; margin: 3rem auto; padding: 0 1rem;
         font: 16px/1.65 ui-serif, Georgia, serif; }}
  h1, h2, h3 {{ font-family: ui-sans-serif, system-ui, sans-serif; line-height: 1.25; }}
  h1 {{ font-size: 2rem; }} h2 {{ margin-top: 2.5rem; font-size: 1.3rem; }}
  a {{ color: inherit; }}
  .cite {{ font-size: 0.72em; opacity: 0.7; }}
  ul {{ padding-left: 1.2rem; }}
  li {{ margin: 0.3rem 0; }}
</style></head>
<body>
{chr(10).join(out)}
</body></html>"""
