"""Tools that do not touch the network: recording, recalling and arithmetic.

`record_finding` is the important one. A researcher that only searches and
fetches has done nothing an agent had to do, because the pipeline could have
done it. Making the researcher decide, explicitly, that it now holds a stated
fact backed by named sources is what turns gathering into research, and it puts
that decision in the trace where it can be inspected.

`calculate` exists because models do arithmetic badly and confidently, which is
the worst pair of properties available. A brief involving percentages, growth
rates or currency totals produces plausible wrong numbers unless the model is
given somewhere else to put the sum.
"""
from __future__ import annotations

import ast
import operator
from typing import Any

from ..core.util import truncate
from ..kernel.tool import Tool, ToolError, tool


@tool
class RecordFinding(Tool):
    name = "record_finding"
    description = (
        "Record one specific, checkable fact you have established, with the source "
        "references that support it. Record a finding as soon as you have it. A "
        "finding with no source reference will be thrown away later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "statement": {
                "type": "string",
                "description": (
                    "The fact, as one self contained sentence. Include the figure, "
                    "date or name. Do not write 'the report says X', write X."
                ),
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Source references that support it, like [\"S1\", \"S4\"].",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "How firmly the sources establish it.",
                "default": "medium",
            },
            "note": {
                "type": "string",
                "description": "Anything qualifying it: a disagreement, a caveat, a date range.",
                "default": "",
            },
        },
        "required": ["statement", "sources"],
    }
    metered = False

    def call(
        self,
        ctx: Any,
        *,
        statement: str,
        sources: list[str],
        confidence: str = "medium",
        note: str = "",
    ) -> Any:
        statement = statement.strip()
        if len(statement) < 15:
            raise ToolError(
                "That statement is too short to be a finding.",
                hint="Write the fact as a full sentence including the specifics.",
            )

        known = {row["ref"] for row in ctx.evidence()}
        refs = [str(s).strip().upper() for s in sources if str(s).strip()]
        unknown = [r for r in refs if r not in known]
        if unknown:
            raise ToolError(
                f"No source is stored under {', '.join(unknown)}.",
                hint=(
                    f"Use fetch_page first, then cite the reference it returns. "
                    f"Stored so far: {', '.join(sorted(known)) or 'nothing yet'}."
                ),
            )
        if not refs:
            raise ToolError(
                "A finding needs at least one source reference.",
                hint="Fetch the page that supports it and cite the reference returned.",
            )

        # A step interrupted part way through is re-run on resume, so a
        # researcher can reach this point having already recorded the same
        # fact. Matching on the normalised statement keeps the board clean
        # across a resume, and incidentally stops a researcher that repeats
        # itself within one loop from inflating the finding count.
        from ..core.util import normalise_text

        normalised = normalise_text(statement)
        for existing in ctx.board.findings:
            if normalise_text(str(existing.get("statement", ""))) == normalised:
                merged = sorted(set(existing.get("sources", [])) | set(refs))
                existing["sources"] = merged
                return {
                    "recorded": existing.get("id", "?"),
                    "findings_so_far": len(ctx.board.findings),
                    "message": "You had already recorded that. Its sources were merged.",
                }

        finding = {
            "id": f"F{len(ctx.board.findings) + 1}",
            "statement": statement,
            "sources": refs,
            "confidence": confidence,
            "note": note.strip(),
            "subquestion": ctx.board.get("_current_subquestion", ""),
            "found_by": ctx.bus.agent or "Researcher",
        }
        count = ctx.board.append("findings", finding)
        return {
            "recorded": finding["id"],
            "findings_so_far": count,
            "message": "Recorded. Keep going, or give your final_answer when done.",
        }


@tool
class RecallMemory(Tool):
    name = "recall"
    description = (
        "Look up what previous runs learned: which domains paywall, which "
        "sources proved unreliable, and facts established before."
    )
    parameters = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["fact", "lesson", "source_quality", "any"],
                "description": "Which kind of memory to read.",
                "default": "any",
            },
            "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 40},
        },
    }
    metered = False

    def call(self, ctx: Any, *, kind: str = "any", limit: int = 10) -> Any:
        rows = ctx.recall(kind="" if kind == "any" else kind, limit=limit)
        return {
            "count": len(rows),
            "memories": [
                {
                    "kind": row["kind"],
                    "key": row["mem_key"],
                    "content": truncate(row["content"], 300),
                    "seen": row["hits"],
                }
                for row in rows
            ],
        }


# Only the operators a research calculation needs. An allow list rather than a
# block list, because `eval` on model output with anything less is a way to
# hand a stranger a shell.
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# A power with a large exponent is the one cheap way to hang the process, so
# the exponent is capped rather than the expression being timed.
MAX_EXPONENT = 64


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("Only numbers are allowed in an expression.")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"Exponents above {MAX_EXPONENT} are not allowed.")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ToolError("Division by zero.")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ToolError("That expression uses something this calculator does not allow.")


@tool
class Calculate(Tool):
    name = "calculate"
    description = (
        "Work out an arithmetic expression exactly. Use it for any percentage, "
        "difference, ratio or total that will appear in the report."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Arithmetic only, for example (8.4 - 6.1) / 6.1 * 100",
            }
        },
        "required": ["expression"],
    }
    metered = False

    def call(self, ctx: Any, *, expression: str) -> Any:
        text = expression.strip().replace(",", "").replace("%", "/100")
        if len(text) > 300:
            raise ToolError("That expression is too long.")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"'{expression}' is not a valid expression: {exc.msg}.") from exc
        value = _eval(tree)
        rounded = round(value, 6)
        return {
            "expression": expression,
            "result": int(rounded) if rounded == int(rounded) else rounded,
        }


@tool
class Remember(Tool):
    name = "remember"
    description = (
        "Record a lesson for future runs, not just this one. Use it for things "
        "worth knowing next time: a source that proved authoritative or "
        "unreliable, a search phrasing that worked where others failed, a fact "
        "that will still be true in a month."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "A short stable identifier, like 'source.reliable.who.int'. "
                    "Remembering the same key again strengthens it."
                ),
            },
            "content": {
                "type": "string",
                "description": "The lesson, as one self contained sentence.",
            },
            "kind": {
                "type": "string",
                "enum": ["fact", "lesson", "source_quality"],
                "description": "fact: a durable fact. lesson: how to work. source_quality: about a source.",
                "default": "lesson",
            },
        },
        "required": ["key", "content"],
    }
    metered = False
    # This is the one tool whose effect outlasts the run that called it, which
    # is what `side_effects` is for. With APPROVE_SIDE_EFFECTS on, it needs a
    # person to agree before anything is written.
    side_effects = True

    def call(self, ctx: Any, *, key: str, content: str, kind: str = "lesson") -> Any:
        key = key.strip()[:120]
        content = content.strip()
        if len(content) < 20:
            raise ToolError(
                "That lesson is too short to be useful later.",
                hint="Write it as a full sentence that will still make sense in a month.",
            )
        if not key:
            raise ToolError("A memory needs a short stable key.")

        ctx.remember(kind=kind, key=key, content=content, run=ctx.run_key)
        return {
            "remembered": key,
            "kind": kind,
            "message": "Stored. Future runs will see this when they plan.",
        }
