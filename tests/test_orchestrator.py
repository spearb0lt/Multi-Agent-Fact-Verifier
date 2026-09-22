"""The guarantees the whole system rests on.

These use a synthetic workflow with no model calls, because what is being
tested here is the run loop itself: that it stops when told, resumes where it
stopped, fans out and rejoins, loops without running away, and never loses work
to a ceiling. Those properties have to hold whatever the agents are doing.
"""
from __future__ import annotations

import threading
import time

import pytest

from mas.kernel import graph as graph_module
from mas.kernel import orchestrator, store
from mas.kernel.contracts import (
    Budget,
    NodeResult,
    ProviderExhausted,
    RunPaused,
    RunStatus,
    Task,
)


def build_graph(name: str, *, calls: list[str] | None = None) -> graph_module.Graph:
    """A plan, fan out, join, loop shaped workflow with no model in it."""
    seen = calls if calls is not None else []
    g = graph_module.Graph(name=name, entry="plan", description="synthetic")

    @g.node("plan", agent="Planner")
    def plan(ctx, task):
        seen.append("plan")
        ctx.board.set("brief", ctx.brief)
        group = "fanout"
        return NodeResult(
            output={"n": 3},
            next=[
                Task(node="work", payload={"i": i}, group=group)
                for i in range(3)
            ],
            action="planned",
        )

    @g.node("work", agent="Worker", parallel=True, retries=1)
    def work(ctx, task):
        seen.append(f"work:{task.payload['i']}")
        ctx.board.append("findings", {"i": task.payload["i"]})
        # Only the last branch of the fan out queues the join, so the join runs
        # once rather than three times.
        if task.payload["i"] == 2:
            return NodeResult(output="done", next=[Task(node="join")])
        return NodeResult(output="done")

    @g.node("join", agent="Editor")
    def join(ctx, task):
        seen.append("join")
        rounds = ctx.board.counter("rounds")
        if rounds < 2:
            # A revision loop: back to plan, bounded by the counter.
            return NodeResult(output={"round": rounds}, next=[Task(node="plan")])
        ctx.board.set("report", {"markdown": "final"})
        return NodeResult(output="finished", done=True)

    g.edge("plan", "work")
    g.edge("work", "join")
    g.edge("join", "plan")
    return graph_module.register(g)


def make_run(graph_name: str, **kw) -> str:
    return store.create_run(
        brief="Test brief", workflow=graph_name, provider="stub", model="stub-large", **kw
    )


def test_completes_and_records_every_step():
    calls: list[str] = []
    build_graph("wf_complete", calls=calls)
    key = make_run("wf_complete")

    out = orchestrator.run(key)

    assert out["status"] == RunStatus.COMPLETED.value
    assert out["claimed"] is True
    # The join loops back once, so the whole shape runs twice: plan, a three
    # way fan out, join, and again.
    assert calls.count("plan") == 2
    assert calls.count("join") == 2
    assert sorted(c for c in calls if c.startswith("work")) == sorted(
        ["work:0", "work:1", "work:2"] * 2
    )

    run_id = store.get_run_id(key)
    steps = store.list_steps(run_id)
    # A fan out is one step, not three, so each round costs plan+work+join.
    assert len(steps) == 6
    assert [s["node"] for s in steps[:3]] == ["plan", "work", "join"]
    assert store.get_run(key)["status"] == RunStatus.COMPLETED.value


def test_pause_between_steps_then_resume_exactly_where_it_stopped():
    calls: list[str] = []
    build_graph("wf_pause", calls=calls)
    key = make_run("wf_pause")

    # max_steps is the same stop path the pause button takes.
    first = orchestrator.run(key, max_steps=2)
    assert first["status"] == RunStatus.PAUSED.value
    assert first["steps"] == 2
    after_pause = list(calls)
    assert store.get_run(key)["status"] == RunStatus.PAUSED.value

    second = orchestrator.resume(key)
    assert second["status"] == RunStatus.COMPLETED.value
    assert second["resumed"] is True
    # Resuming continued the sequence rather than restarting it.
    assert calls[: len(after_pause)] == after_pause
    assert second["steps"] > first["steps"]

    run_id = store.get_run_id(key)
    seqs = [s["seq"] for s in store.list_steps(run_id)]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))


def test_blackboard_survives_a_pause():
    build_graph("wf_state")
    key = make_run("wf_state")

    orchestrator.run(key, max_steps=2)
    paused = orchestrator.run(key, max_steps=0)

    assert paused["board"]["counts"]["findings"] == 6
    assert paused["board"]["has_report"] is True


def test_budget_breach_pauses_and_keeps_the_work():
    build_graph("wf_budget")
    key = make_run("wf_budget", budget=Budget(max_steps=3))

    out = orchestrator.run(key)

    assert out["status"] == RunStatus.PAUSED.value
    assert "steps ceiling" in out["reason"]
    assert out["spent"]["steps"] == 3

    # The work already done is still there, and raising the ceiling continues it.
    resumed = orchestrator.resume(key, budget=Budget(max_steps=100))
    assert resumed["status"] == RunStatus.COMPLETED.value
    assert resumed["spent"]["steps"] > 3


def test_cancel_is_terminal_and_not_resumable():
    build_graph("wf_cancel")
    key = make_run("wf_cancel")

    orchestrator.run(key, max_steps=1)
    assert orchestrator.cancel(key) is True
    assert store.get_run(key)["status"] == RunStatus.CANCELLED.value

    out = orchestrator.resume(key)
    assert out["claimed"] is False
    assert out["status"] == RunStatus.CANCELLED.value


def test_control_flag_stops_a_running_orchestrator():
    """The pause button, exercised from another thread while the run executes."""
    g = graph_module.Graph(name="wf_slow", entry="tick")

    @g.node("tick", agent="Ticker")
    def tick(ctx, task):
        time.sleep(0.05)
        n = ctx.board.counter("ticks")
        return NodeResult(output=n, next=[Task(node="tick")] if n < 200 else [], done=n >= 200)

    g.edge("tick", "tick")
    graph_module.register(g)
    key = make_run("wf_slow")

    def stopper():
        time.sleep(0.4)
        orchestrator.pause(key)

    t = threading.Thread(target=stopper)
    t.start()
    out = orchestrator.run(key)
    t.join()

    assert out["status"] == RunStatus.PAUSED.value
    assert out["steps"] < 200
    # Everything it got through is durable and it carries on from there.
    resumed = orchestrator.resume(key)
    assert resumed["steps"] > out["steps"]


def test_a_second_worker_cannot_claim_a_running_run():
    g = graph_module.Graph(name="wf_lease", entry="hold")
    started = threading.Event()
    release = threading.Event()

    @g.node("hold", agent="Holder")
    def hold(ctx, task):
        started.set()
        release.wait(timeout=5)
        return NodeResult(output="ok", done=True)

    g.edge("hold", "hold")
    graph_module.register(g)
    key = make_run("wf_lease")

    result: dict = {}
    t = threading.Thread(target=lambda: result.update(orchestrator.run(key)))
    t.start()
    started.wait(timeout=5)

    second = orchestrator.run(key)
    assert second["claimed"] is False
    assert "Another worker" in second["reason"]

    release.set()
    t.join()
    assert result["status"] == RunStatus.COMPLETED.value


def test_routing_to_an_undeclared_node_is_rejected():
    g = graph_module.Graph(name="wf_bad", entry="start")

    @g.node("start", agent="A")
    def start(ctx, task):
        return NodeResult(next=[Task(node="nowhere")])

    @g.node("other", agent="B")
    def other(ctx, task):
        return NodeResult(done=True)

    g.edge("start", "other")
    graph_module.register(g)
    key = make_run("wf_bad")

    out = orchestrator.run(key)
    assert out["status"] == RunStatus.FAILED.value
    assert "nowhere" in out["error"]


def test_a_failing_branch_is_retried_then_abandoned():
    attempts: list[int] = []
    g = graph_module.Graph(name="wf_retry", entry="start")

    @g.node("start", agent="A")
    def start(ctx, task):
        return NodeResult(
            next=[Task(node="flaky", payload={"i": i}, group="g") for i in range(2)]
        )

    @g.node("flaky", agent="B", parallel=True, retries=1)
    def flaky(ctx, task):
        attempts.append(task.attempt)
        if task.payload["i"] == 0:
            raise RuntimeError("this branch always fails")
        return NodeResult(output="ok", next=[Task(node="end")])

    @g.node("end", agent="C")
    def end(ctx, task):
        return NodeResult(output="done", done=True)

    g.edge("start", "flaky")
    g.edge("flaky", "end")
    graph_module.register(g)
    key = make_run("wf_retry")

    out = orchestrator.run(key)

    # The good branch carried the run to completion; the bad one was tried
    # twice and then left behind rather than failing everything.
    assert out["status"] == RunStatus.COMPLETED.value
    assert attempts.count(1) == 2 and attempts.count(2) == 1


def test_graph_validation_catches_an_unreachable_node():
    g = graph_module.Graph(name="wf_orphan", entry="a")

    @g.node("a")
    def a(ctx, task):
        return NodeResult(done=True)

    @g.node("b")
    def b(ctx, task):
        return NodeResult(done=True)

    with pytest.raises(ValueError, match="nothing can reach"):
        graph_module.register(g)


def test_approval_gate_parks_the_run_until_answered():
    g = graph_module.Graph(name="wf_gate", entry="gate")
    passes: list[str] = []

    @g.node("gate", agent="Supervisor")
    def gate(ctx, task):
        answer = store.latest_answer(ctx.run_id, "gate")
        if answer is None:
            passes.append("asked")
            return NodeResult(await_approval={"question": "Proceed?", "options": ["yes", "no"]})
        passes.append(f"answered:{answer['answer']}")
        return NodeResult(output=answer["answer"], done=True)

    g.edge("gate", "gate")
    graph_module.register(g)
    key = make_run("wf_gate")

    out = orchestrator.run(key)
    assert out["status"] == RunStatus.WAITING.value
    assert passes == ["asked"]

    assert orchestrator.answer(key, reply="yes") is True
    out2 = orchestrator.resume(key)
    assert out2["status"] == RunStatus.COMPLETED.value
    assert passes == ["asked", "answered:yes"]


def test_work_in_flight_is_requeued_when_a_step_is_interrupted():
    """The bug that made a paused run come back and report itself complete.

    A task is popped off the queue before it runs, so a stop arriving mid step
    has to put it back. Without that the checkpoint records a queue with a hole
    in it, and the resume walks straight past the work that never happened.
    """
    g = graph_module.Graph(name="wf_inflight", entry="start")
    ran: list[str] = []

    @g.node("start", agent="A")
    def start(ctx, task):
        ran.append("start")
        return NodeResult(next=[Task(node="slow")])

    @g.node("slow", agent="B")
    def slow(ctx, task):
        ran.append("slow")
        if len(ran) == 2:
            # Stop from inside the node, the way a pause or a spent quota does.
            raise RunPaused("stopped mid step")
        return NodeResult(output="finished", next=[Task(node="end")])

    @g.node("end", agent="C")
    def end(ctx, task):
        ran.append("end")
        return NodeResult(output="done", done=True)

    g.edge("start", "slow")
    g.edge("slow", "end")
    graph_module.register(g)
    key = make_run("wf_inflight")

    first = orchestrator.run(key)
    assert first["status"] == RunStatus.PAUSED.value
    # The interrupted node is back on the queue rather than lost.
    assert first["queued"] == ["slow"]
    assert ran == ["start", "slow"]

    second = orchestrator.resume(key)
    assert second["status"] == RunStatus.COMPLETED.value
    # It really did re-run the interrupted node and then carry on.
    assert ran == ["start", "slow", "slow", "end"]


def test_a_partly_finished_fan_out_keeps_the_branches_that_completed():
    """One branch running out of quota must not discard its siblings' work."""
    g = graph_module.Graph(name="wf_partial", entry="start")

    @g.node("start", agent="A")
    def start(ctx, task):
        return NodeResult(
            next=[Task(node="branch", payload={"i": i}, group="g") for i in range(3)]
        )

    @g.node("branch", agent="B", parallel=True, retries=0)
    def branch(ctx, task):
        index = task.payload["i"]
        if index == 1 and not ctx.board.get("second_pass"):
            raise ProviderExhausted("stub", "out of quota")
        ctx.board.append("done_branches", index)
        return NodeResult(output=index)

    @g.node("end", agent="C")
    def end(ctx, task):
        return NodeResult(done=True)

    g.edge("start", "branch")
    g.edge("branch", "end")
    graph_module.register(g)
    key = make_run("wf_partial")

    first = orchestrator.run(key)
    assert first["status"] == RunStatus.PAUSED.value
    assert "quota" in first["reason"]
    # Two branches finished and are recorded; only the failed one is requeued.
    assert first["queued"] == ["branch"]

    run_id = store.get_run_id(key)
    state = store.latest_checkpoint(run_id)["state"]
    assert sorted(state["done_branches"]) == [0, 2]

    # Letting it through on the second pass completes the fan out.
    state["second_pass"] = True
    store.save_checkpoint(
        run_id, seq=int(store.latest_checkpoint(run_id)["seq"]) + 1,
        node="branch", queue=[Task(node="branch", payload={"i": 1})], state=state,
    )
    second = orchestrator.resume(key)
    assert second["status"] in {RunStatus.COMPLETED.value, RunStatus.PAUSED.value}
    final = store.latest_checkpoint(run_id)["state"]
    assert sorted(final["done_branches"]) == [0, 1, 2]


def test_a_live_run_in_another_process_is_not_reclaimed():
    """Starting the API must not stop a run the CLI is happily executing.

    `reclaim_orphaned_runs` exists to rescue runs whose worker died. Written
    without the staleness check it also paused a run that was executing fine in
    another process, which looked exactly like the run mysteriously stopping
    for no reason.
    """
    from mas.api.app import reclaim_orphaned_runs

    build_graph("wf_reclaim")
    key = make_run("wf_reclaim")

    # A live worker: claimed, and heartbeating now.
    assert store.claim_run(key) is True
    store.heartbeat(key, phase="plan")

    assert reclaim_orphaned_runs() == 0
    assert store.get_run(key)["status"] == RunStatus.RUNNING.value


def test_a_run_whose_worker_died_is_reclaimed():
    from mas.api.app import reclaim_orphaned_runs
    from mas.kernel import store as store_module

    build_graph("wf_orphan")
    key = make_run("wf_orphan")
    store.claim_run(key)

    # Backdate the heartbeat past the lease, which is what a killed process
    # leaves behind.
    from datetime import timedelta

    from mas.core.util import to_iso, utcnow

    store.update_run(
        key,
        heartbeat_at=to_iso(utcnow() - timedelta(seconds=store_module.STALE_LEASE_SECONDS + 30)),
    )

    assert reclaim_orphaned_runs() == 1
    run = store.get_run(key)
    assert run["status"] == RunStatus.PAUSED.value
    assert "process executing this run stopped" in run["pause_reason"]
    # And it is resumable, with nothing lost.
    assert RunStatus(run["status"]).resumable
