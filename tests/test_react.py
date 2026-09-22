"""The reason and act loop, against scripted model replies.

The loop's job is to survive models behaving badly, so these tests are mostly a
catalogue of the specific ways they do: answering in prose, inventing tools,
looping on one call, never finishing, and finishing with nothing. Each of those
was seen from a real free tier model.

Both protocols are covered. The prompt protocol is what runs on providers with
no function calling API, and the native one is what runs on the sixteen that
have one. A bug in either is invisible from the other, which is how a missing
method reached a live run once already.
"""
from __future__ import annotations

from typing import Any

import pytest

from mas.kernel import store
from mas.kernel.blackboard import Blackboard
from mas.kernel.budget import BudgetGuard
from mas.kernel.bus import EventBus
from mas.kernel.context import RunContext
from mas.kernel.contracts import Budget
from mas.kernel.policy import ModelPolicy
from mas.kernel.react import ReactLoop
from mas.kernel.tool import Tool, ToolRegistry


class Echo(Tool):
    name = "echo"
    description = "Return whatever you send it."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "anything"}},
        "required": ["text"],
    }
    metered = False

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def call(self, ctx: Any, *, text: str) -> Any:
        self.calls.append({"text": text})
        return {"echoed": text}


class Counter(Tool):
    name = "count"
    description = "Count how many times it has been called."
    parameters = {"type": "object", "properties": {}}
    metered = False

    def __init__(self) -> None:
        self.n = 0

    def call(self, ctx: Any) -> Any:
        self.n += 1
        return {"count": self.n}


@pytest.fixture
def harness(monkeypatch):
    """A context wired to a scripted provider, in either protocol."""
    from mas.core.llm import registry as llm
    from mas.core.llm.base import Completion, NativeToolCall, Usage

    class Script:
        def __init__(self) -> None:
            self.replies: list[Any] = []
            self.calls: list[dict[str, Any]] = []
            self.native = False

        def push(self, *replies: Any) -> None:
            """A str is returned as text; a (name, args) tuple as a tool call."""
            self.replies.extend(replies)

        def generate(self, prompt, *, model, system=None, temperature=0.2,
                     max_tokens=4096, json_mode=False, tools=None):
            self.calls.append({"prompt": prompt, "system": system, "tools": tools})
            reply = self.replies.pop(0) if self.replies else '{"final_answer": {}}'
            usage = Usage(provider="stub", model=model, input_tokens=10, output_tokens=5)
            if isinstance(reply, tuple):
                name, arguments = reply
                return Completion(
                    text="",
                    usage=usage,
                    tool_calls=[NativeToolCall(id="c1", name=name, arguments=arguments)],
                    finish_reason="tool_calls",
                )
            return Completion(text=str(reply), usage=usage)

    script = Script()

    class StubProvider:
        id = "stub"
        label = "Stub"
        key_names = ("STUB",)
        models = ()
        local = False

        @property
        def supports_tools(self) -> bool:
            return script.native

        def is_available(self) -> bool:
            return True

        def default_model(self) -> str:
            return "stub-large"

        def cheap_model(self) -> str:
            return "stub-small"

        def generate(self, prompt, **kwargs):
            return script.generate(prompt, **kwargs)

    provider = StubProvider()
    monkeypatch.setattr(llm, "provider_map", lambda: {"stub": provider})
    monkeypatch.setattr(llm, "available_providers", lambda: [provider])

    registry = ToolRegistry()
    echo, counter = Echo(), Counter()
    registry.register(echo)
    registry.register(counter)

    run_key = store.create_run(brief="test", workflow="none", provider="stub")
    run_id = store.get_run_id(run_key)
    ctx = RunContext(
        run_key=run_key,
        run_id=run_id,
        brief="test",
        board=Blackboard({"brief": "test"}),
        bus=EventBus(run_id, run_key),
        budget=BudgetGuard(Budget()),
        policy=ModelPolicy(provider="stub"),
        tools=registry,
    )
    # The kernel's own registry is what `execute` looks in, so the test tools
    # have to be reachable from there too.
    monkeypatch.setattr("mas.kernel.tool.registry", registry)
    return ctx, script, echo, counter


def loop(ctx, **kw) -> ReactLoop:
    return ReactLoop(
        ctx,
        role=kw.pop("role", "Researcher"),
        goal=kw.pop("goal", "do the thing"),
        system=kw.pop("system", "You are a test agent."),
        tools=kw.pop("tools", ("echo", "count")),
        **kw,
    )


# ---------------------------------------------------------- prompt protocol


def test_prompt_protocol_calls_a_tool_then_finishes(harness):
    ctx, script, echo, _ = harness
    script.push(
        '{"thought": "I should echo", "action": "echo", "action_input": {"text": "hi"}}',
        '{"thought": "done", "final_answer": {"answer": "hi was echoed"}}',
    )
    result = loop(ctx).run()

    assert echo.calls == [{"text": "hi"}]
    assert result.output == {"answer": "hi was echoed"}
    assert result.stopped_because == "finished"
    assert result.iterations == 2
    assert result.thoughts == ["I should echo", "done"]


def test_prose_instead_of_json_does_not_kill_the_turn(harness):
    ctx, script, _, _ = harness
    script.push(
        "I think the answer is probably 42, but let me be clear about it.",
        '{"final_answer": {"answer": "42"}}',
    )
    result = loop(ctx).run()
    assert result.output in ({"answer": "42"}, "I think the answer is probably 42, but let me be clear about it.")


def test_an_invented_tool_comes_back_as_a_correctable_error(harness):
    ctx, script, _, _ = harness
    script.push(
        '{"action": "teleport", "action_input": {}}',
        '{"final_answer": {"answer": "recovered"}}',
    )
    result = loop(ctx).run()

    assert result.output == {"answer": "recovered"}
    # The agent was told what it did wrong rather than the step being lost.
    observation = result and store.list_events(ctx.run_id, kinds=("tool.error",))
    assert observation and "no tool called 'teleport'" in observation[0]["message"]


def test_an_identical_repeated_call_is_answered_from_the_record(harness):
    ctx, script, _, counter = harness
    script.push(
        '{"action": "count", "action_input": {}}',
        '{"action": "count", "action_input": {}}',
        '{"final_answer": {"answer": "stopped looping"}}',
    )
    result = loop(ctx).run()

    # The second identical call never reached the tool.
    assert counter.n == 1
    assert result.output == {"answer": "stopped looping"}


def test_running_out_of_turns_forces_an_answer(harness):
    ctx, script, _, _ = harness
    script.push(
        '{"action": "echo", "action_input": {"text": "a"}}',
        '{"action": "echo", "action_input": {"text": "b"}}',
        '{"final_answer": {"answer": "forced out"}}',
    )
    result = loop(ctx, max_iterations=2).run()

    assert result.stopped_because == "max_iterations"
    assert result.iterations == 2
    # The wrap up call still produced something usable.
    assert result.output == {"answer": "forced out"}


def test_alternative_spellings_of_finishing_are_accepted(harness):
    ctx, script, _, _ = harness
    script.push('{"thought": "done", "result": {"answer": "ok"}}')
    assert loop(ctx).run().output == {"answer": "ok"}


def test_a_schema_field_named_answer_is_not_mistaken_for_an_envelope(harness):
    """The Researcher's own schema has an "answer" field beside others.

    Unwrapping it as though it were the envelope silently discarded confidence
    and gaps, which only shows up much later as a report missing its caveats.
    """
    ctx, script, _, _ = harness
    script.push(
        '{"answer": "the rate held at 5.25", "confidence": "high", '
        '"gaps": ["no minutes published"], "findings_recorded": 3}'
    )
    output = loop(ctx).run().output
    assert output["answer"] == "the rate held at 5.25"
    assert output["confidence"] == "high"
    assert output["gaps"] == ["no minutes published"]


def test_a_lone_answer_key_is_still_unwrapped(harness):
    ctx, script, _, _ = harness
    script.push('{"thought": "done", "answer": "just this"}')
    assert loop(ctx).run().output == "just this"


def test_a_bare_string_argument_is_fitted_to_the_required_one(harness):
    ctx, script, echo, _ = harness
    script.push(
        '{"action": "echo", "action_input": "just a string"}',
        '{"final_answer": {"answer": "fine"}}',
    )
    loop(ctx).run()
    assert echo.calls == [{"text": "just a string"}]


# ---------------------------------------------------------- native protocol


def test_native_protocol_declares_tools_and_reads_the_call_back(harness):
    ctx, script, echo, _ = harness
    script.native = True
    script.push(("echo", {"text": "native hi"}), '{"answer": "done natively"}')

    result = loop(ctx).run()

    assert echo.calls == [{"text": "native hi"}]
    assert result.output == {"answer": "done natively"}
    # The schemas really were sent, in the shape the gateways expect.
    first = script.calls[0]["tools"]
    assert first and first[0]["type"] == "function"
    assert {t["function"]["name"] for t in first} == {"echo", "count"}


def test_native_wrap_up_withdraws_the_tools(harness):
    ctx, script, _, _ = harness
    script.native = True
    script.push(
        ("count", {}),
        ("count", {}),
        '{"answer": "forced"}',
    )
    result = loop(ctx, max_iterations=2).run()

    assert result.stopped_because == "max_iterations"
    # The wrap up call declared no tools at all, which is what stops a model
    # that would otherwise keep calling one instead of answering.
    assert not script.calls[-1]["tools"]
    # A lone "answer" key is an envelope, so it unwraps to the value itself.
    assert result.output == "forced"


def test_native_text_answer_that_is_not_json_is_still_an_answer(harness):
    ctx, script, _, _ = harness
    script.native = True
    script.push("The answer is plainly stated in prose.")
    result = loop(ctx).run()
    assert result.output == "The answer is plainly stated in prose."


def test_the_protocol_is_chosen_from_the_provider(harness):
    ctx, script, _, _ = harness
    script.native = False
    assert loop(ctx).native is False
    script.native = True
    assert loop(ctx).native is True
    # An empty tool list means no tools, so there is nothing to declare and
    # the native path is not taken however capable the provider is.
    assert loop(ctx, tools=()).native is False
    assert loop(ctx, tools=()).specs == []


def test_budget_exhaustion_stops_the_loop_without_losing_work(harness):
    ctx, script, _, _ = harness
    ctx.budget = BudgetGuard(Budget(max_tokens=40))
    ctx.llm.ctx = ctx
    script.push(
        '{"action": "count", "action_input": {}}',
        '{"action": "count", "action_input": {"unused": 1}}',
        '{"final_answer": {"answer": "never reached"}}',
    )
    result = loop(ctx).run()
    assert result.stopped_because in {"budget", "finished", "max_iterations"}
