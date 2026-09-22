"""The reason and act loop: think, call a tool, read the result, think again.

This is what makes an agent an agent rather than a prompt. The model is not
asked to produce an answer. It is given a goal and a set of capabilities and
allowed to decide, turn by turn, what to do next, until it decides it is done
or the kernel decides it has had enough.

Tool calling is expressed as a JSON protocol in the prompt rather than through
a provider's native function calling API. That choice is deliberate and it is
the reason this works across all nineteen providers. Native function calling is
available on perhaps half of them, is spelled differently on each, and is
absent from exactly the free tier and local models this system is built to run
on. One protocol that every model can follow beats a faster path that only some
can take, especially when the slower path is a few hundred tokens of system
prompt. `native.py` layers the fast path on top where a provider supports it.

What the loop actually has to survive, in the order these were found to matter:

* a model that answers in prose when JSON was demanded,
* a model that calls a tool that does not exist,
* a model that calls the same tool with the same arguments forever,
* a model that never says it is finished,
* a model that says it is finished while producing nothing.

Every one of those ends with the agent returning its best available result, not
with an exception, because a degraded finding is worth more to the run than a
dead step.
"""
from __future__ import annotations

from typing import Any

from ..core.util import dumps, truncate
from .contracts import (
    AgentResult,
    BudgetExceeded,
    EventKind,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from .tool import execute

# How much of one observation the agent is allowed to read back. Search results
# and page bodies are long, and a scratchpad that keeps all of them in full
# overruns the context of exactly the small models this is meant to serve.
OBSERVATION_LIMIT = 4000
# Older turns are squeezed harder than recent ones, because what the agent
# needs from three turns ago is what it learned, not the raw text.
OLD_OBSERVATION_LIMIT = 600
RECENT_TURNS = 2


PROTOCOL = """You work in a loop. On each turn you reply with ONE JSON object and nothing else.

To use a tool:
{"thought": "why this tool and these arguments", "action": "tool_name", "action_input": {"arg": "value"}}

When you have enough to answer:
{"thought": "why you are done", "final_answer": <your answer>}

Rules:
1. Reply with one JSON object. No prose around it, no code fence, no second object.
2. "action" must be exactly one of the tool names listed. Do not invent a tool.
3. Never repeat a tool call you have already made with the same arguments. If a
   result was disappointing, change the arguments or try a different tool.
4. An observation starting with ERROR means that call failed. Read the reason
   and do something different. Do not retry the identical call.
5. Prefer finishing. Every extra turn spends the run's budget, and a good
   answer now beats a slightly better answer that never arrives."""


NATIVE_PROTOCOL = """You have tools. Call them when you need them, one per turn.

When you have everything you need, stop calling tools and reply with your final
answer as a single JSON object and nothing else.

Rules:
1. Never repeat a tool call you have already made with the same arguments. If a
   result was disappointing, change the arguments or try a different tool.
2. An observation starting with ERROR means that call failed. Read the reason
   and do something different. Do not retry the identical call.
3. Prefer finishing. Every extra turn spends the run's budget."""


def _openai_tool_schemas(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """The tool list in the shape every OpenAI compatible gateway expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters or {"type": "object", "properties": {}},
            },
        }
        for spec in specs
    ]


def _tool_block(specs: list[ToolSpec]) -> str:
    if not specs:
        return "You have no tools on this turn. Answer from what you already know."
    lines = [spec.prompt_form() for spec in specs]
    return "Tools available to you:\n" + "\n".join(lines)


def _signature(call: ToolCall) -> str:
    return f"{call.name}:{dumps(call.arguments)}"


def _parse_turn(payload: Any) -> tuple[str, ToolCall | None, Any, bool]:
    """Read one model turn into thought, action, answer, finished.

    Models are inconsistent about this in ways that are all individually minor
    and collectively fatal, so every known shape is accepted here rather than
    being treated as a protocol violation the agent has to be scolded into
    fixing at the cost of a whole turn.
    """
    if isinstance(payload, list):
        # A model that emitted a list of turns meant the first one.
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return "", None, payload, True

    thought = str(payload.get("thought") or payload.get("reasoning") or "").strip()

    # "final_answer" is unambiguous: it is an envelope and its value is the
    # answer, whatever else is alongside it.
    for key in ("final_answer", "final"):
        if key in payload and payload[key] not in (None, ""):
            return thought, None, payload[key], True

    # "answer", "result" and "output" are ambiguous, because they are also
    # ordinary field names in the schemas roles are asked to fill. Unwrapping
    # one of those blindly cost the Researcher its confidence and gaps fields,
    # which is invisible until a report turns out to be missing them. So they
    # count as an envelope only when there is nothing else of substance beside
    # them; otherwise the object as a whole is the answer.
    narrative = {"thought", "reasoning"}
    for key in ("answer", "result", "output"):
        if key in payload and payload[key] not in (None, ""):
            if set(payload) - narrative - {key}:
                return thought, None, payload, True
            return thought, None, payload[key], True

    action = payload.get("action") or payload.get("tool") or payload.get("tool_name")
    if isinstance(action, dict):
        # {"action": {"name": ..., "arguments": {...}}}
        args = action.get("arguments") or action.get("input") or {}
        name = str(action.get("name") or "")
        action = name
    else:
        args = (
            payload.get("action_input")
            or payload.get("arguments")
            or payload.get("tool_input")
            or payload.get("input")
            or {}
        )

    action = str(action or "").strip()
    if not action:
        # No action and no answer. Whatever prose it produced is the answer.
        return thought, None, payload, bool(thought)

    if action.lower() in {"final_answer", "final", "finish", "done", "answer"}:
        return thought, None, args, True

    if isinstance(args, str):
        # A single positional argument. Only unambiguous when the tool takes
        # exactly one, which the validator sorts out from here.
        args = {"__positional__": args}
    if not isinstance(args, dict):
        args = {}

    return thought, ToolCall(name=action, arguments=args), None, False


def _fit_positional(call: ToolCall, specs: list[ToolSpec]) -> ToolCall:
    """Turn a bare string argument into the tool's single required argument."""
    if "__positional__" not in call.arguments:
        return call
    value = call.arguments.pop("__positional__")
    spec = next((s for s in specs if s.name == call.name), None)
    if spec is None:
        return call
    required = (spec.parameters or {}).get("required", [])
    props = list((spec.parameters or {}).get("properties", {}))
    target = required[0] if required else (props[0] if props else "")
    if target:
        call.arguments[target] = value
    return call


class ReactLoop:
    """One agent's turn taking, bounded by iterations and by the run's budget."""

    def __init__(
        self,
        ctx: Any,
        *,
        role: str,
        goal: str,
        system: str,
        tools: tuple[str, ...] = (),
        max_iterations: int = 0,
        final_schema: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self.ctx = ctx
        self.role = role
        self.goal = goal
        self.system = system
        self.allowed = tools
        self.max_iterations = max_iterations or int(ctx.option("max_agent_iterations", 8))
        self.final_schema = final_schema
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.specs = ctx.tools.specs(ctx, tools)
        self.turns: list[dict[str, Any]] = []
        self.seen: dict[str, int] = {}
        # Which protocol this role will use, decided once. Native where the
        # provider has a function calling API, the prompt protocol everywhere
        # else. Models trained for native tools emit one regardless, and a
        # provider that was not told about them rejects the whole request, so
        # this is a correctness choice rather than a performance one.
        self.native = bool(self.specs) and ctx.llm.supports_native_tools(role)
        self.tool_schemas = _openai_tool_schemas(self.specs) if self.native else []

    # ------------------------------------------------------------ prompting

    def _scratchpad(self) -> str:
        if not self.turns:
            return "No steps taken yet."
        lines = []
        total = len(self.turns)
        for index, turn in enumerate(self.turns):
            recent = index >= total - RECENT_TURNS
            limit = OBSERVATION_LIMIT if recent else OLD_OBSERVATION_LIMIT
            lines.append(f"Turn {index + 1}")
            if turn.get("thought"):
                lines.append(f"  Thought: {truncate(turn['thought'], 400)}")
            if turn.get("action"):
                lines.append(f"  Action: {turn['action']} {dumps(turn.get('arguments', {}))}")
                lines.append(f"  Observation: {truncate(turn.get('observation', ''), limit)}")
            if turn.get("note"):
                lines.append(f"  Note: {turn['note']}")
        return "\n".join(lines)

    def _system_prompt(self) -> str:
        if self.native:
            parts = [self.system.strip(), "", NATIVE_PROTOCOL]
            if self.final_schema:
                parts += ["", f"Your final answer must match this shape:\n{self.final_schema}"]
            return "\n".join(parts)
        parts = [self.system.strip(), "", PROTOCOL, "", _tool_block(self.specs)]
        if self.final_schema:
            parts += ["", f"Your final_answer must match this shape:\n{self.final_schema}"]
        return "\n".join(parts)

    def _user_prompt(self, *, forced: bool = False) -> str:
        remaining = self.max_iterations - len(self.turns)
        budget_note = (
            "The run is low on budget. Finish on this turn."
            if self.ctx.budget.critical
            else f"You have {max(0, remaining)} turns left before you must answer."
        )
        if forced:
            budget_note = (
                "You are out of turns. Reply now with your final answer, using only "
                "what the observations below already contain. Do not call another tool."
            )
        return (
            f"Goal:\n{self.goal}\n\n"
            f"What you have done so far:\n{self._scratchpad()}\n\n"
            f"{budget_note}"
        )

    # ------------------------------------------------------------------ turn

    def _turn(self, *, forced: bool = False) -> tuple[str, ToolCall | None, Any, bool]:
        """Take one turn, by whichever protocol this provider supports.

        The forced wrap up always takes the JSON path, whatever the provider
        supports. Withdrawing the tools and asking for an object through the
        provider's own JSON mode is the strongest available way to make a model
        commit to an answer instead of calling one more tool.
        """
        if not self.native or forced:
            payload = self.ctx.llm.complete_json(
                self._user_prompt(forced=forced),
                role=self.role,
                system=self._system_prompt(),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                purpose=f"{self.role.lower()}.react",
                default={},
            )
            return _parse_turn(payload)

        completion = self.ctx.llm.complete_with_tools(
            self._user_prompt(forced=forced),
            role=self.role,
            system=self._system_prompt(),
            tools=self.tool_schemas,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if completion.tool_calls:
            native = completion.tool_calls[0]
            return (
                completion.text.strip(),
                ToolCall(name=native.name, arguments=dict(native.arguments)),
                None,
                False,
            )

        # No tool call means the model is answering.
        from ..core.llm.base import coerce_json

        text = completion.text.strip()
        if not text:
            return "", None, None, False
        try:
            parsed = coerce_json(text)
        except Exception:
            parsed = None
        if parsed is None:
            return "", None, text, True

        if isinstance(parsed, dict):
            # A model handed native tools sometimes still answers in the prompt
            # protocol's shape. Honour an action found that way rather than
            # treating the envelope as the answer.
            if any(key in parsed for key in ("action", "tool", "tool_name")):
                thought, call, _, _ = _parse_turn(parsed)
                if call is not None:
                    return thought, call, None, False
            for key in ("final_answer", "final"):
                if key in parsed and parsed[key] not in (None, ""):
                    return str(parsed.get("thought") or ""), None, parsed[key], True

        # Otherwise the object IS the answer. It must not be unwrapped by key:
        # a role whose schema happens to contain "answer" alongside
        # "confidence" and "gaps" would lose everything but the one field.
        return "", None, parsed, True

    # ------------------------------------------------------------------ loop

    def run(self) -> AgentResult:
        result = AgentResult()
        bus = self.ctx.bus
        bus.emit(
            EventKind.AGENT_START,
            f"{self.role} started",
            agent=self.role,
            goal=truncate(self.goal, 300),
            tools=[s.name for s in self.specs],
            max_iterations=self.max_iterations,
        )

        for iteration in range(1, self.max_iterations + 1):
            # A pause or a cancel propagates out of the agent untouched: the
            # orchestrator owns what stopping means, and swallowing it here
            # would make the button take a whole agent to respond.
            self.ctx.check_control()

            try:
                self.ctx.budget.check()
            except BudgetExceeded:
                # The run cannot afford another turn. Take what has been
                # gathered rather than losing the whole step to the ceiling.
                result.stopped_because = "budget"
                break

            try:
                thought, call, answer, finished = self._turn()
            except BudgetExceeded:
                # The ceiling bound on the way into the call. Keep what the
                # agent already gathered and return: the orchestrator checks
                # the budget again at the top of its own loop and turns this
                # into a pause, so nothing is lost and the run still stops.
                result.stopped_because = "budget"
                break
            result.iterations = iteration

            if thought:
                result.thoughts.append(thought)
                bus.thought(self.role, truncate(thought, 500), iteration=iteration)

            if finished:
                result.output = answer
                result.stopped_because = "finished"
                bus.emit(
                    EventKind.AGENT_FINISH,
                    f"{self.role} finished after {iteration} turn(s)",
                    agent=self.role,
                    iterations=iteration,
                )
                return result

            if call is None:
                self.turns.append(
                    {
                        "thought": thought,
                        "note": "That turn contained neither an action nor a final_answer. "
                        "Reply with one JSON object using exactly one of them.",
                    }
                )
                continue

            call = _fit_positional(call, self.specs)
            signature = _signature(call)
            repeats = self.seen.get(signature, 0)
            self.seen[signature] = repeats + 1

            if repeats >= 1:
                # Answering a repeat from the record costs nothing and breaks
                # the loop the model has fallen into, which spending the call
                # again would only confirm.
                self.turns.append(
                    {
                        "thought": thought,
                        "action": call.name,
                        "arguments": call.arguments,
                        "observation": (
                            "You already made this exact call. Its result is above. "
                            "Either change the arguments, use a different tool, or "
                            "give your final_answer."
                        ),
                    }
                )
                bus.emit(
                    EventKind.LOG,
                    f"{self.role} repeated {call.name} with identical arguments",
                    agent=self.role,
                    level="warning",
                )
                continue

            result.tool_calls.append(call)
            outcome: ToolResult = execute(
                self.ctx, call, agent=self.role, allowed=self.allowed
            )
            self.turns.append(
                {
                    "thought": thought,
                    "action": call.name,
                    "arguments": call.arguments,
                    "observation": outcome.as_observation(OBSERVATION_LIMIT),
                }
            )

        # Out of turns. One last call that is not allowed to ask for a tool,
        # because an agent that has gathered evidence and never concluded has
        # done the expensive half of the work and none of the useful half.
        if result.stopped_because != "budget":
            result.stopped_because = "max_iterations"

        try:
            _, _, answer, _ = self._turn(forced=True)
            result.output = answer
        except BudgetExceeded:
            result.output = None
        except Exception:
            result.output = None

        bus.emit(
            EventKind.AGENT_FINISH,
            f"{self.role} stopped after {result.iterations} turn(s) ({result.stopped_because})",
            agent=self.role,
            level="warning",
            iterations=result.iterations,
            stopped_because=result.stopped_because,
        )
        return result


def think(
    ctx: Any,
    *,
    role: str,
    goal: str,
    system: str,
    tools: tuple[str, ...] = (),
    max_iterations: int = 0,
    final_schema: str = "",
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> AgentResult:
    """Run one reason and act loop. The shape every agent uses."""
    return ReactLoop(
        ctx,
        role=role,
        goal=goal,
        system=system,
        tools=tools,
        max_iterations=max_iterations,
        final_schema=final_schema,
        temperature=temperature,
        max_tokens=max_tokens,
    ).run()
