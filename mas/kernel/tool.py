"""What an agent is allowed to do, and the accounting around each attempt.

A tool here is not a Python function an agent happens to call. It is a
declared capability: a name, a described argument schema, a cost class, and a
record of every invocation with its arguments and its result. That record is
what separates an agent choosing a tool from a pipeline calling a function, and
it is the first thing worth showing anyone who asks whether this system is
really agentic.

Arguments are validated here rather than inside each tool. Models supply the
wrong type constantly: a string "3" where an integer belongs, a bare string
where a list belongs, an invented argument name. Coercing those in one place
costs a few lines and saves every tool from defending itself, and a coercion
that fails comes back to the agent as an observation it can correct on the next
turn rather than as an exception that kills the step.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..core.util import truncate
from .contracts import EventKind, ToolCall, ToolResult, ToolSpec

if TYPE_CHECKING:  # pragma: no cover
    from .context import RunContext


class ToolError(RuntimeError):
    """A tool failed in a way the agent should see and may be able to work around."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} {self.hint}".strip()


class Tool:
    """One capability an agent may invoke.

    Subclasses set the class attributes and implement `call`. The name is a
    class attribute rather than a constructor argument so that a tool is
    identified the same way in the registry, the prompt, the database and the
    UI without anyone having to keep four strings in step.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}
    metered: bool = True
    side_effects: bool = False

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters or {},
            metered=self.metered,
            side_effects=self.side_effects,
        )

    def call(self, ctx: RunContext, **kwargs: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def is_available(self, ctx: RunContext) -> tuple[bool, str]:
        """Whether this tool can run here, and why not when it cannot.

        A tool whose backend has no key reports itself unavailable and is left
        out of the agent's prompt entirely. Offering a capability and then
        failing on use wastes a whole model turn teaching the agent something
        the registry already knew.
        """
        return True, ""


# ------------------------------------------------------------- validation


_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _coerce(value: Any, spec: dict[str, Any], path: str) -> Any:
    """Bend a model's argument into the declared type, or say why it will not bend."""
    kind = spec.get("type", "string")

    if kind == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if value is None:
            return ""
        raise ToolError(f"Argument '{path}' should be a string, got {type(value).__name__}.")

    if kind == "integer":
        if isinstance(value, bool):
            raise ToolError(f"Argument '{path}' should be an integer, got a boolean.")
        if isinstance(value, int):
            return value
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ToolError(f"Argument '{path}' should be an integer, got {value!r}.") from None

    if kind == "number":
        if isinstance(value, bool):
            raise ToolError(f"Argument '{path}' should be a number, got a boolean.")
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            raise ToolError(f"Argument '{path}' should be a number, got {value!r}.") from None

    if kind == "boolean":
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ToolError(f"Argument '{path}' should be true or false, got {value!r}.")

    if kind == "array":
        if value is None:
            return []
        if isinstance(value, str):
            # Models very often send an array argument as its JSON text rather
            # than as an array, especially through a native tool calling API
            # where the whole argument object arrives as a string and only the
            # outer layer gets parsed. Taken literally, '["S4"]' becomes the
            # single item '["S4"]', which then matches nothing. That silently
            # cost a run every one of its findings, so it is unwrapped here.
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                import json

                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        value = parsed
                except ValueError:
                    # Not JSON after all, so fall back to the separator form.
                    value = [part.strip() for part in text[1:-1].split(",") if part.strip()]
            elif "," in text:
                # "S1, S2" is the other shape this arrives in.
                value = [part.strip() for part in text.split(",") if part.strip()]
            else:
                value = [text] if text else []
        if not isinstance(value, list):
            # A list of one, sent as the bare item. Unambiguous, so it is
            # wrapped rather than rejected.
            value = [value]
        item_spec = spec.get("items") or {"type": "string"}
        cleaned = [_coerce(item, item_spec, f"{path}[{i}]") for i, item in enumerate(value)]
        # Strip the quoting that survives a double encoded string list.
        return [
            item.strip().strip('"').strip("'") if isinstance(item, str) else item
            for item in cleaned
        ]

    if kind == "object":
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ToolError(f"Argument '{path}' should be an object, got {type(value).__name__}.")
        return value

    return value


def validate_arguments(spec: ToolSpec, raw: dict[str, Any]) -> dict[str, Any]:
    """Check and coerce what the model supplied against the tool's schema."""
    schema = spec.parameters or {}
    props: dict[str, Any] = schema.get("properties", {}) or {}
    required: list[str] = schema.get("required", []) or []

    if not isinstance(raw, dict):
        raise ToolError(f"Arguments for '{spec.name}' should be an object, got {type(raw).__name__}.")

    missing = [key for key in required if key not in raw or raw[key] in (None, "")]
    if missing:
        raise ToolError(
            f"'{spec.name}' needs {', '.join(missing)}.",
            hint=f"Call it again with every required argument: {', '.join(required)}.",
        )

    unknown = [key for key in raw if key not in props]
    if unknown and props:
        raise ToolError(
            f"'{spec.name}' has no argument named {', '.join(unknown)}.",
            hint=f"Valid arguments are: {', '.join(props) or 'none'}.",
        )

    out: dict[str, Any] = {}
    for key, sub in props.items():
        if key in raw and raw[key] is not None:
            out[key] = _coerce(raw[key], sub, key)
        elif "default" in sub:
            out[key] = sub["default"]

    for key, sub in props.items():
        allowed = sub.get("enum")
        if allowed and key in out and out[key] not in allowed:
            raise ToolError(
                f"Argument '{key}' must be one of {', '.join(map(str, allowed))}, got {out[key]!r}."
            )
        if sub.get("type") in {"integer", "number"} and key in out:
            low, high = sub.get("minimum"), sub.get("maximum")
            if low is not None and out[key] < low:
                out[key] = low
            if high is not None and out[key] > high:
                out[key] = high

    return out


# --------------------------------------------------------------- registry


class ToolRegistry:
    """The tools this deployment has, and the subset a given role may use."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if not tool.name:
            raise ValueError("A tool must have a name.")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def available(self, ctx: RunContext, names: tuple[str, ...] = ()) -> list[Tool]:
        """The tools a role may use here, filtered to those that can actually run.

        An empty `names` means no tools, not every tool. A role that declares
        none is a reasoning role, and handing it the full toolbox would let the
        Writer go back to the web to avoid a judgement the verification stage
        already made for it.
        """
        wanted = [self._tools[n] for n in names if n in self._tools]
        out = []
        for tool in wanted:
            ok, _ = tool.is_available(ctx)
            if ok:
                out.append(tool)
        return out

    def specs(self, ctx: RunContext, names: tuple[str, ...] = ()) -> list[ToolSpec]:
        return [tool.spec for tool in self.available(ctx, names)]


registry = ToolRegistry()


def tool(cls: type[Tool]) -> type[Tool]:
    """Class decorator that puts a tool in the default registry."""
    registry.register(cls())
    return cls


# --------------------------------------------------------------- execution


def execute(
    ctx: RunContext,
    call: ToolCall,
    *,
    agent: str = "",
    allowed: tuple[str, ...] = (),
) -> ToolResult:
    """Run one tool call, accounting for it whatever the outcome.

    Every exit path from here is a ToolResult, including the failures. An agent
    that called a tool wrongly gets told so and can try again next turn, which
    is worth far more than an exception that ends the step, so the only thing
    that escapes this function is a budget breach.
    """
    from .contracts import BudgetExceeded

    started = time.monotonic()
    tool_obj = registry.get(call.name)

    def finish(result: ToolResult) -> ToolResult:
        result.duration_ms = int((time.monotonic() - started) * 1000)
        ctx.store.record_tool_call(
            ctx.run_id,
            step_seq=ctx.seq,
            agent=agent,
            tool=call.name,
            arguments=call.arguments,
            result={"ok": result.ok, "error": result.error, "content": _brief(result.content)},
            ok=result.ok,
            error=result.error,
            duration_ms=result.duration_ms,
        )
        ctx.bus.emit(
            EventKind.TOOL_RESULT if result.ok else EventKind.TOOL_ERROR,
            truncate(result.error or _describe(result.content), 400),
            agent=agent,
            level="info" if result.ok else "warning",
            tool=call.name,
            ok=result.ok,
            duration_ms=result.duration_ms,
            call_id=call.call_id,
        )
        return result

    if tool_obj is None:
        return finish(
            ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=(
                    f"There is no tool called '{call.name}'. "
                    f"Available tools: {', '.join(t.name for t in registry.available(ctx, allowed))}."
                ),
            )
        )

    if allowed and call.name not in allowed:
        return finish(
            ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=f"'{call.name}' is not available to {agent or 'this agent'}.",
            )
        )

    ok, reason = tool_obj.is_available(ctx)
    if not ok:
        return finish(
            ToolResult(call_id=call.call_id, name=call.name, ok=False, error=reason)
        )

    # A tool whose effect outlasts the run needs permission when the operator
    # has asked for it. Refusing here rather than parking the whole run is
    # deliberate: an agent told no can carry on and finish, whereas pausing a
    # parallel fan out to ask about one optional write would stall everything.
    if tool_obj.side_effects and bool(ctx.option("approve_side_effects", False)):
        answer = ctx.store.latest_answer(ctx.run_id, f"tool:{call.name}")
        approved = bool(answer) and not str(answer.get("answer", "")).lower().startswith(
            ("no", "deny", "reject")
        )
        if not approved:
            ctx.store.request_approval(
                ctx.run_id,
                node=f"tool:{call.name}",
                kind="tool",
                question=(
                    f"{agent or 'An agent'} wants to use '{call.name}', which changes "
                    f"something outside this run. Allow it for the rest of the run?"
                ),
                options=["allow", "deny"],
                payload={"tool": call.name, "arguments": call.arguments},
            )
            return finish(
                ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    ok=False,
                    error=(
                        f"'{call.name}' changes something outside this run and has not "
                        f"been approved yet. Carry on without it."
                    ),
                )
            )

    try:
        arguments = validate_arguments(tool_obj.spec, call.arguments)
    except ToolError as exc:
        return finish(ToolResult(call_id=call.call_id, name=call.name, ok=False, error=str(exc)))

    if tool_obj.metered:
        # Charged before the call, so a tool that hangs and is killed still
        # counted against the ceiling it was about to spend.
        ctx.budget.check()
        ctx.budget.charge_tool()

    ctx.bus.emit(
        EventKind.TOOL_CALL,
        f"{agent or 'agent'} calls {call.name}",
        agent=agent,
        tool=call.name,
        arguments=arguments,
        call_id=call.call_id,
    )

    try:
        content = tool_obj.call(ctx, **arguments)
    except BudgetExceeded:
        raise
    except ToolError as exc:
        return finish(ToolResult(call_id=call.call_id, name=call.name, ok=False, error=str(exc)))
    except Exception as exc:
        return finish(
            ToolResult(
                call_id=call.call_id,
                name=call.name,
                ok=False,
                error=f"{type(exc).__name__}: {truncate(str(exc), 300)}",
            )
        )

    return finish(ToolResult(call_id=call.call_id, name=call.name, ok=True, content=content))


def _brief(content: Any) -> Any:
    """Shrink a result for the audit row. The full text lives in `evidence`."""
    if isinstance(content, str):
        return truncate(content, 1200)
    if isinstance(content, list):
        return {"kind": "list", "length": len(content), "head": content[:3]}
    if isinstance(content, dict):
        return {k: _brief(v) for k, v in list(content.items())[:12]}
    return content


def _describe(content: Any) -> str:
    if isinstance(content, list):
        return f"{len(content)} results"
    if isinstance(content, dict):
        return ", ".join(f"{k}={_short(v)}" for k, v in list(content.items())[:4])
    return truncate(str(content), 200)


def _short(value: Any) -> str:
    if isinstance(value, list):
        return f"[{len(value)}]"
    if isinstance(value, dict):
        return "{...}"
    return truncate(str(value), 60)
