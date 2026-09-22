"""The capabilities added last, which the rest of the suite did not reach.

Each of these was verified once by hand against the live web or a live model,
which proves it worked that afternoon and nothing else. These pin the behaviour
so a later change cannot quietly undo it:

* the side effect gate, which is the only thing standing between an agent and a
  write that outlasts the run,
* the fallback that recovers a page a site refused to serve, and the check that
  stops page furniture being stored as a citable source,
* the live spend figures, which read zero for the whole of a long step before
  they were added,
* the reconcile and claim_check graphs, whose shape the orchestrator relies on.
"""
from __future__ import annotations

from typing import Any

import pytest

from mas.kernel import graph as graph_module
from mas.kernel import store
from mas.kernel.blackboard import Blackboard
from mas.kernel.budget import BudgetGuard
from mas.kernel.bus import EventBus
from mas.kernel.context import RunContext
from mas.kernel.contracts import Budget, ToolCall
from mas.kernel.policy import ModelPolicy
from mas.kernel.tool import Tool, ToolRegistry, execute


def make_ctx(**config: Any) -> RunContext:
    key = store.create_run(brief="feature test", workflow="none", config=config)
    run_id = store.get_run_id(key)
    return RunContext(
        run_key=key,
        run_id=run_id,
        brief="feature test",
        board=Blackboard({"brief": "feature test"}),
        bus=EventBus(run_id, key),
        budget=BudgetGuard(Budget()),
        policy=ModelPolicy(),
        config=config,
    )


# --------------------------------------------------------- side effect gate


class Writer(Tool):
    """A tool whose effect outlasts the run, which is what the gate is for."""

    name = "persist"
    description = "Writes something durable."
    parameters = {"type": "object", "properties": {}}
    metered = False
    side_effects = True

    def __init__(self) -> None:
        self.writes = 0

    def call(self, ctx: Any, **kwargs: Any) -> Any:
        self.writes += 1
        return {"written": self.writes}


@pytest.fixture
def gated(monkeypatch):
    registry = ToolRegistry()
    tool = Writer()
    registry.register(tool)
    monkeypatch.setattr("mas.kernel.tool.registry", registry)
    return registry, tool


def test_a_side_effecting_tool_runs_freely_when_approval_is_not_required(gated):
    _, tool = gated
    ctx = make_ctx()
    result = execute(ctx, ToolCall(name="persist"), agent="Researcher")
    assert result.ok is True
    assert tool.writes == 1


def test_a_side_effecting_tool_is_refused_until_it_is_approved(gated):
    _, tool = gated
    ctx = make_ctx(approve_side_effects=True)

    result = execute(ctx, ToolCall(name="persist"), agent="Researcher")

    assert result.ok is False
    assert "has not been approved" in result.error
    assert tool.writes == 0, "the tool must not have run"

    # The refusal raised a question rather than silently failing, so somebody
    # can actually grant it.
    pending = store.pending_approval(ctx.run_id)
    assert pending is not None
    assert pending["node"] == "tool:persist"
    assert pending["options"] == ["allow", "deny"]


def test_approval_lets_it_through_for_the_rest_of_the_run(gated):
    _, tool = gated
    ctx = make_ctx(approve_side_effects=True)
    execute(ctx, ToolCall(name="persist"), agent="Researcher")

    pending = store.pending_approval(ctx.run_id)
    store.answer_approval(int(pending["id"]), answer="allow")

    assert execute(ctx, ToolCall(name="persist"), agent="Researcher").ok is True
    assert execute(ctx, ToolCall(name="persist"), agent="Researcher").ok is True
    assert tool.writes == 2


def test_denial_keeps_it_refused(gated):
    _, tool = gated
    ctx = make_ctx(approve_side_effects=True)
    execute(ctx, ToolCall(name="persist"), agent="Researcher")
    store.answer_approval(int(store.pending_approval(ctx.run_id)["id"]), answer="deny")

    assert execute(ctx, ToolCall(name="persist"), agent="Researcher").ok is False
    assert tool.writes == 0


# ------------------------------------------------------------- live spend


def test_live_spend_reports_what_a_running_step_has_already_cost():
    """The figure a viewer sees while a step that takes minutes is still going.

    The run's own spend blob is written at step boundaries, so before this
    existed the cost sat at zero and the clock at zero for the whole of a
    research step while three agents were visibly working.
    """
    key = store.create_run(brief="spend", workflow="none")
    run_id = store.get_run_id(key)
    store.claim_run(key)

    store.record_usage(run_id, agent="Researcher", provider="p", model="m",
                       tokens_in=1000, tokens_out=250, cost_usd=0.004)
    store.record_usage(run_id, agent="FactChecker", provider="p", model="m",
                       tokens_in=500, tokens_out=100, cost_usd=0.002)
    store.record_tool_call(run_id, agent="Researcher", tool="web_search")
    store.record_tool_call(run_id, agent="Researcher", tool="fetch_page")

    run = store.get_run(key)
    assert run["spent"].get("tokens", 0) == 0, "the stored blob is still at zero"

    live = store.live_spend(run_id, run)
    assert live["tokens"] == 1850
    assert live["tokens_in"] == 1500
    assert live["tokens_out"] == 350
    assert live["llm_calls"] == 2
    assert live["tool_calls"] == 2
    assert live["usd"] == pytest.approx(0.006)
    # The clock moves during the step rather than jumping at the end of it.
    assert live["seconds"] > 0


def test_live_spend_falls_back_to_the_stored_blob_when_nothing_is_itemised():
    key = store.create_run(brief="spend", workflow="none")
    run_id = store.get_run_id(key)
    run = store.get_run(key)
    live = store.live_spend(run_id, run)
    assert live["tokens"] == 0 and live["usd"] == 0


# ------------------------------------------------- recovering a blocked page


class FakeResponse:
    def __init__(self, status: int, text: str = "") -> None:
        self.status = status
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


ARTICLE_MD = "\n\n".join(
    [
        "[Home](https://x.test/) [Topics](https://x.test/t) [Donate](https://x.test/d)",
        "Coffee consumption of three to five cups a day was associated with a lower "
        "risk of cardiovascular disease across the pooled cohorts, though the "
        "authors caution that residual confounding cannot be excluded.",
        "The association held after adjustment for smoking status and physical "
        "activity, and was similar in decaffeinated coffee drinkers, which argues "
        "against caffeine being the sole mechanism at work here.",
        "We process your personal information to improve our services. As a "
        "California consumer you have the right to opt-out at any time.",
    ]
)

FURNITURE_ONLY = "\n\n".join(
    [
        "[Home](https://x.test/) [Topics](https://x.test/t) [Donate](https://x.test/d)",
        "This link is provided for convenience only and is not an endorsement of "
        "the linked-to entity, and all information here has been reviewed and "
        "approved by the association in question.",
        "We process your personal information to improve our services. As a "
        "California consumer you have the right to opt-out at any time.",
    ]
)


def test_a_blocked_page_is_recovered_through_the_proxy(monkeypatch):
    from mas.tools import fallback

    class Client:
        def get(self, url, **kwargs):
            assert url.startswith(fallback.JINA_PREFIX), "should try the proxy first"
            return FakeResponse(200, f"Title: Coffee and the heart\n\n{ARTICLE_MD}")

    got = fallback.read_blocked_page("https://blocked.test/a", Client(), status=403)
    assert got is not None
    body, title, meta = got
    assert meta["via"] == "jina"
    assert title == "Coffee and the heart"
    assert "cardiovascular disease" in body


def test_a_404_is_not_worth_a_fallback(monkeypatch):
    """A dead link is dead. Only "not to you" statuses are worth a second try."""
    from mas.tools import fallback

    class Client:
        def get(self, url, **kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("no fallback should have been attempted")

    assert fallback.read_blocked_page("https://gone.test/a", Client(), status=404) is None


def test_a_page_of_pure_furniture_is_not_worth_storing():
    """The failure that stored 2,900 words of navigation as a citable source."""
    from mas.kernel.semantic import MIN_ARTICLE_CHARS, article_text

    assert len(article_text(FURNITURE_ONLY)) < MIN_ARTICLE_CHARS
    assert len(article_text(ARTICLE_MD)) > 0
    # And what survives is the article, with the notices and the menu gone.
    kept = article_text(ARTICLE_MD)
    assert "cardiovascular disease" in kept
    assert "opt-out" not in kept
    assert "Donate" not in kept


# ------------------------------------------------------------ graph shapes


def test_both_workflows_are_registered_and_coherent():
    names = {g.name for g in graph_module.available()}
    assert {"research_report", "claim_check"} <= names
    for g in graph_module.available():
        g.validate()


def test_the_research_graph_routes_verification_through_reconciliation():
    """Contradiction checking must sit between verifying and writing.

    If verify routed straight to supervise again, conflicting claims would
    reach the Writer unexamined and the report would assert one side of a real
    disagreement as settled.
    """
    g = graph_module.get("research_report")
    assert "reconcile" in g.nodes
    assert g.nodes["reconcile"].agent == "Reconciler"
    assert "reconcile" in g.successors("verify")
    assert "supervise" in g.successors("reconcile")
    assert "supervise" not in g.successors("verify")


def test_the_claim_graph_gathers_three_angles_and_can_loop():
    g = graph_module.get("claim_check")
    assert g.entry == "frame"
    assert g.nodes["gather"].parallel is True
    # It can send the researchers back out when the evidence is too thin.
    assert "frame" in g.successors("adjudicate")
    assert "explain" in g.successors("adjudicate")


def test_every_node_agent_has_a_model_tier():
    """A role with no tier silently falls back to cheap, including the Writer."""
    from mas.kernel.policy import ROLE_TIERS

    for g in graph_module.available():
        for node in g.nodes.values():
            if node.agent:
                assert node.agent in ROLE_TIERS, f"{node.agent} has no declared tier"


# ------------------------------------------------------- deployment ceilings


def test_a_run_started_without_a_budget_gets_the_deployment_ceilings(monkeypatch):
    """The environment is what a host configures, so it has to be what binds.

    This hid for a long time because the dataclass defaults and the settings
    defaults are the same numbers. They only diverge when a deployment actually
    sets one of the variables, which is the one case that matters: a blueprint
    capping runs at 780 seconds because the platform stops an idle instance at
    900 was being discarded for everything started from the web UI, and the
    agreement between the two sets of defaults made it look correct.
    """
    from mas.api.schemas import BudgetIn
    from mas.core import settings

    monkeypatch.setattr(settings, "MAX_RUN_SECONDS", 780)
    monkeypatch.setattr(settings, "MAX_RUN_TOKENS", 111_000)
    monkeypatch.setattr(settings, "MAX_TOOL_CALLS", 17)

    budget = BudgetIn().merged()
    assert budget.max_seconds == 780
    assert budget.max_tokens == 111_000
    assert budget.max_tool_calls == 17


def test_an_explicit_ceiling_still_wins_over_the_deployment(monkeypatch):
    from mas.api.schemas import BudgetIn
    from mas.core import settings

    monkeypatch.setattr(settings, "MAX_RUN_SECONDS", 780)
    budget = BudgetIn(max_seconds=120).merged()
    assert budget.max_seconds == 120
    # And the ceilings it did not name still come from the deployment.
    assert budget.max_tokens == settings.MAX_RUN_TOKENS


def test_the_create_route_does_not_bypass_the_deployment_ceilings():
    """A guard on the specific line that was wrong.

    `Budget()` here rather than the merged defaults is the whole bug, and it is
    an easy thing to reintroduce because it reads perfectly well.
    """
    import inspect

    from mas.api import routes

    source = inspect.getsource(routes.create_run)
    assert "BudgetIn()" in source, "the create route must fall back to the deployment budget"
