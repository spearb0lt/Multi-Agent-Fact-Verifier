"""The run loop: schedule a step, record it, decide whether to carry on.

The whole design turns on one invariant. Between two steps, everything that
describes the run is in the database: the queue of work not yet done, the
blackboard of work already done, the spend so far, and the operator's wishes.
Nothing important lives only in this process. That is what makes stopping free
and resuming exact, and it is why the loop looks like a state machine rather
than a call chain.

Stopping is not a special case. A pause, a cancel, a budget breach and a crash
all arrive here as exceptions, and all four take the same path: write the
checkpoint, release the lease, record why. The only difference between them is
the status the run is left in and whether it can be resumed.

The one cost of this design is worth naming. A step that is interrupted part
way through is abandoned and re-run on resume, so nodes are executed at least
once, not exactly once. Everything expensive a node does is deduplicated
underneath it (evidence by normalised URL, artifacts by version), so a re-run
costs model calls rather than correctness. Making it exactly once would mean
checkpointing inside a node, which would mean every agent understanding
persistence, which is a far worse trade.
"""
from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..core.util import truncate
from . import graph as graph_module
from . import store
from .blackboard import Blackboard
from .budget import BudgetGuard
from .bus import EventBus
from .context import RunContext
from .contracts import (
    Budget,
    BudgetExceeded,
    Control,
    EventKind,
    NodeResult,
    ProviderExhausted,
    RunCancelled,
    RunPaused,
    RunStatus,
    Spend,
    Task,
)
from .policy import ModelPolicy

log = logging.getLogger(__name__)


class Outcome(dict):
    """What a call to `run` reports back, as a plain mapping for the API."""


def _pop_batch(queue: list[Task]) -> list[Task]:
    """Take the next unit of work: one task, or a whole parallel group.

    A fan-out is drained as one step rather than one step per branch. That
    keeps the step count meaningful (it counts decisions, not threads) and it
    keeps the checkpoint honest, because half a fan-out is not a state anything
    downstream knows how to resume from.
    """
    head = queue.pop(0)
    if not head.group:
        return [head]
    batch = [head]
    remaining = []
    for task in queue:
        if task.group == head.group:
            batch.append(task)
        else:
            remaining.append(task)
    queue[:] = remaining
    return batch


def _run_node(ctx: RunContext, node: graph_module.Node, task: Task) -> NodeResult:
    ctx.node = task.node
    ctx.bus.bind(node=task.node, agent=node.agent)
    ctx.bus.emit(
        EventKind.NODE_ENTER,
        f"{node.label} started",
        agent=node.agent,
        node=task.node,
        attempt=task.attempt,
        payload_keys=sorted(task.payload)[:10],
    )
    result = node.handler(ctx, task)
    if result is None:
        result = NodeResult()
    return result


class BatchOutcome:
    """What one step's tasks produced, including what they did not finish.

    The unfinished list is the important part. A task is removed from the queue
    before it runs, so a stop that arrives mid step would otherwise lose it:
    the checkpoint would record a queue that no longer contains the work in
    flight, and resuming would skip straight past it. That is exactly how a
    paused run came back and reported itself complete having written nothing.
    """

    def __init__(self) -> None:
        self.results: list[NodeResult] = []
        self.failures: list[tuple[Task, Exception]] = []
        self.unfinished: list[Task] = []
        self.control_error: Exception | None = None


STOP_SIGNALS = (RunPaused, RunCancelled, BudgetExceeded, ProviderExhausted)


def _execute_batch(
    ctx: RunContext,
    graph: graph_module.Graph,
    batch: list[Task],
) -> BatchOutcome:
    """Run one step's tasks, in parallel when the node allows it.

    A stop signal is returned rather than raised, so the caller can still apply
    whatever the finished branches produced before it unwinds. Half a fan out
    of research is worth keeping.
    """
    node = graph.nodes[batch[0].node]
    outcome = BatchOutcome()

    if len(batch) == 1 or not node.parallel:
        for index, task in enumerate(batch):
            try:
                outcome.results.append(_run_node(ctx, graph.nodes[task.node], task))
            except STOP_SIGNALS as exc:
                outcome.control_error = exc
                outcome.unfinished.extend(batch[index:])
                return outcome
            except Exception as exc:
                outcome.failures.append((task, exc))
        return outcome

    workers = max(1, min(len(batch), int(ctx.option("agent_concurrency", 3))))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mas") as pool:
        futures = {pool.submit(_run_node, ctx, node, task): task for task in batch}
        for future, task in futures.items():
            try:
                outcome.results.append(future.result())
            except STOP_SIGNALS as exc:
                # Remembered rather than re-raised here, so that every branch
                # finishes before the loop unwinds. Unwinding early would leave
                # threads writing to the database after the run was marked
                # stopped, which is the inconsistency the checkpoint discipline
                # exists to prevent.
                outcome.control_error = outcome.control_error or exc
                outcome.unfinished.append(task)
            except Exception as exc:
                outcome.failures.append((task, exc))
    return outcome


def run(run_key: str, *, max_steps: int = 0) -> Outcome:
    """Execute or continue a run until it finishes, stops or is told to stop.

    Safe to call on a run that is already finished, already running elsewhere,
    or half done. It reports what it found rather than assuming a fresh start,
    which is what lets the CLI, the API and a scheduler all call it without
    coordinating.
    """
    row = store.get_run(run_key)
    if row is None:
        raise LookupError(f"No run named {run_key}.")

    status = RunStatus(str(row["status"]))
    if status.terminal:
        return Outcome(run_key=run_key, status=status.value, claimed=False,
                       reason="This run has already finished.")

    if not store.claim_run(run_key):
        return Outcome(run_key=run_key, status=row["status"], claimed=False,
                       reason="Another worker holds this run.")

    run_id = int(row["id"])
    config = dict(row["config"] or {})
    graph = graph_module.get(str(row["workflow"]))

    guard = BudgetGuard(Budget.from_dict(row["budget"]), Spend.from_dict(row["spent"]))
    bus = EventBus(run_id, run_key)
    policy = ModelPolicy(
        provider=str(row["provider"] or ""),
        model=str(row["model"] or ""),
        cheap_model=str(row["cheap_model"] or ""),
        role_tiers=config.get("role_tiers"),
        allow_degrade=bool(config.get("allow_degrade", True)),
    )

    checkpoint = store.latest_checkpoint(run_id)
    if checkpoint:
        board = Blackboard.from_dict(checkpoint["state"])
        queue: list[Task] = list(checkpoint["queue"])
        seq = int(checkpoint["seq"])
        resumed = True
    else:
        board = Blackboard({"brief": row["brief"], "title": row["title"] or ""})
        queue = [Task(node=graph.entry, payload={"brief": row["brief"]})]
        seq = 0
        resumed = False

    ctx = RunContext(
        run_key=run_key,
        run_id=run_id,
        brief=str(row["brief"]),
        board=board,
        bus=bus,
        budget=guard,
        policy=policy,
        config=config,
        seq=seq,
    )

    bus.emit(
        EventKind.RUN_RESUMED if resumed else EventKind.RUN_STARTED,
        f"Run {'resumed' if resumed else 'started'} on workflow '{graph.name}'",
        workflow=graph.name,
        from_step=seq,
        budget=guard.budget.to_dict(),
        models=policy.describe(),
    )

    steps_this_call = 0
    final_status = RunStatus.COMPLETED
    reason = ""
    error_text = ""

    try:
        while queue:
            ctx.check_control(force=True)
            guard.check()

            if max_steps and steps_this_call >= max_steps:
                raise RunPaused(f"Stopped after {max_steps} step(s) as asked.")

            batch = _pop_batch(queue)
            seq += 1
            steps_this_call += 1
            ctx.seq = seq
            guard.charge_step()

            started = time.monotonic()
            before_in, before_out = guard.spent.tokens_in, guard.spent.tokens_out
            before_usd = guard.spent.usd

            node = graph.nodes.get(batch[0].node)
            if node is None:
                raise LookupError(
                    f"Workflow '{graph.name}' has no node '{batch[0].node}'."
                )

            outcome = _execute_batch(ctx, graph, batch)
            results, failures = outcome.results, outcome.failures

            # A branch that failed is retried on its own budget of attempts,
            # then abandoned. One lost researcher is a thinner report; one lost
            # writer is a failed run, and the difference is whether anything
            # downstream still has work queued.
            for task, exc in failures:
                bus.emit(
                    EventKind.NODE_ERROR,
                    f"{task.node} failed: {truncate(str(exc), 300)}",
                    node=task.node,
                    agent=node.agent,
                    level="error",
                    attempt=task.attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
                log.debug("node %s failed\n%s", task.node, traceback.format_exc())
                if task.attempt < node.retries + 1:
                    retry = Task(
                        node=task.node,
                        payload=task.payload,
                        attempt=task.attempt + 1,
                        group=task.group,
                    )
                    queue.insert(0, retry)
                    bus.emit(
                        EventKind.NODE_RETRY,
                        f"Retrying {task.node} (attempt {retry.attempt})",
                        node=task.node,
                        level="warning",
                    )

            exhausted = [
                (task, exc) for task, exc in failures if task.attempt >= node.retries + 1
            ]
            if exhausted and not results and not queue:
                first_exc = exhausted[0][1]
                raise RuntimeError(
                    f"Node '{batch[0].node}' failed after {node.retries + 1} attempts: {first_exc}"
                ) from first_exc

            done = False
            awaiting: dict[str, Any] | None = None
            next_tasks: list[Task] = []
            outputs: list[Any] = []

            for result in results:
                outputs.append(result.output)
                for message in result.messages:
                    ctx.send(message)
                for artifact in result.artifacts:
                    ctx.emit_artifact(artifact)
                for task in result.next:
                    graph.check_hop(batch[0].node, task.node)
                    next_tasks.append(task)
                done = done or result.done
                awaiting = awaiting or result.await_approval

            duration_ms = int((time.monotonic() - started) * 1000)
            store.record_step(
                run_id,
                seq=seq,
                node=batch[0].node,
                agent=node.agent,
                action=(results[0].action if results else ""),
                status="ok" if not failures else ("partial" if results else "error"),
                attempt=batch[0].attempt,
                payload_in={"tasks": [t.to_dict() for t in batch]},
                payload_out={
                    "outputs": _summarise(outputs),
                    "next": [t.node for t in next_tasks],
                    "failures": len(failures),
                },
                tokens_in=guard.spent.tokens_in - before_in,
                tokens_out=guard.spent.tokens_out - before_out,
                cost_usd=round(guard.spent.usd - before_usd, 6),
                duration_ms=duration_ms,
                error="; ".join(f"{type(e).__name__}: {e}" for _, e in failures)[:800],
            )

            bus.emit(
                EventKind.NODE_EXIT,
                f"{node.label} finished in {duration_ms} ms",
                node=batch[0].node,
                agent=node.agent,
                next=[t.node for t in next_tasks],
                duration_ms=duration_ms,
            )

            if outcome.control_error is not None:
                # Put back exactly what did not finish, ahead of anything the
                # branches that did finish asked for. Without this the work in
                # flight is lost with the popped batch, and the run comes back
                # from a pause believing it has nothing left to do.
                queue[:0] = outcome.unfinished + next_tasks
                bus.emit(
                    EventKind.LOG,
                    f"Stopping with {len(outcome.unfinished)} task(s) put back on the queue.",
                    node=batch[0].node,
                    level="warning",
                    requeued=[t.node for t in outcome.unfinished],
                )
                raise outcome.control_error

            if awaiting:
                # Park the run in front of the node that asked, so the answer
                # is read by the same node when the run is resumed.
                queue.insert(0, Task(node=batch[0].node, payload=batch[0].payload))
                store.request_approval(
                    run_id,
                    node=batch[0].node,
                    kind=str(awaiting.get("kind", "gate")),
                    question=str(awaiting.get("question", "")),
                    options=list(awaiting.get("options", [])),
                    payload=dict(awaiting.get("payload", {})),
                )
                bus.emit(
                    EventKind.APPROVAL_REQUESTED,
                    str(awaiting.get("question", "A decision is needed.")),
                    node=batch[0].node,
                    agent=node.agent,
                    level="warning",
                )
                store.save_checkpoint(
                    run_id, seq=seq, node=batch[0].node, queue=queue, state=board.to_dict()
                )
                final_status = RunStatus.WAITING
                reason = str(awaiting.get("question", "Waiting for a person to answer."))
                break

            queue.extend(next_tasks)

            if bool(ctx.option("checkpoint_every_step", True)):
                store.save_checkpoint(
                    run_id, seq=seq, node=batch[0].node, queue=queue, state=board.to_dict()
                )
                store.prune_checkpoints(run_id, keep=5)

            guard.sync_seconds()
            store.heartbeat(run_key, phase=batch[0].node, spent=guard.spent)

            if done:
                queue.clear()
                break

    except RunPaused as exc:
        final_status = RunStatus.PAUSED
        reason = str(exc)
        _checkpoint_now(run_id, seq, queue, board, ctx)
        bus.emit(EventKind.RUN_PAUSED, reason, level="warning", at_step=seq)

    except RunCancelled as exc:
        final_status = RunStatus.CANCELLED
        reason = str(exc)
        _checkpoint_now(run_id, seq, queue, board, ctx)
        bus.emit(EventKind.RUN_CANCELLED, reason, level="warning", at_step=seq)

    except BudgetExceeded as exc:
        # A ceiling is not a failure. The run keeps everything it has and can
        # be continued the moment the operator decides it is worth more.
        final_status = RunStatus.PAUSED
        reason = exc.detail
        _checkpoint_now(run_id, seq, queue, board, ctx)
        bus.emit(
            EventKind.BUDGET_BREACH,
            exc.detail,
            level="warning",
            limit=exc.limit,
            at_step=seq,
            spent=guard.spent.to_dict(),
        )
        bus.emit(EventKind.RUN_PAUSED, exc.detail, level="warning", at_step=seq)

    except ProviderExhausted as exc:
        # Out of quota is not a broken run. A free tier window reopens on its
        # own, so this pauses exactly like a budget breach and everything the
        # team gathered is still on the board when it is resumed.
        final_status = RunStatus.PAUSED
        reason = f"{exc.detail} {exc.hint}".strip()
        _checkpoint_now(run_id, seq, queue, board, ctx)
        bus.emit(
            EventKind.BUDGET_BREACH,
            exc.detail,
            level="warning",
            limit="provider_quota",
            provider=exc.provider,
            at_step=seq,
            hint=exc.hint,
        )
        bus.emit(EventKind.RUN_PAUSED, reason, level="warning", at_step=seq)

    except Exception as exc:
        final_status = RunStatus.FAILED
        error_text = f"{type(exc).__name__}: {exc}"
        reason = truncate(error_text, 400)
        log.error("run %s failed\n%s", run_key, traceback.format_exc())
        _checkpoint_now(run_id, seq, queue, board, ctx)
        bus.emit(
            EventKind.RUN_FAILED,
            reason,
            level="error",
            at_step=seq,
            traceback=truncate(traceback.format_exc(), 4000),
        )

    else:
        if final_status is not RunStatus.WAITING:
            final_status = RunStatus.COMPLETED
            bus.emit(
                EventKind.RUN_COMPLETED,
                f"Run finished after {seq} step(s)",
                at_step=seq,
                spent=guard.spent.to_dict(),
                board=board.summary(),
            )

    guard.sync_seconds()
    store.release_run(
        run_key,
        status=final_status,
        reason=reason if final_status is not RunStatus.FAILED else "",
        error=error_text,
        spent=guard.spent,
        phase=ctx.node,
    )
    if final_status is RunStatus.WAITING:
        store.update_run(run_key, pause_reason=reason)

    return Outcome(
        run_key=run_key,
        status=final_status.value,
        claimed=True,
        resumed=resumed,
        steps=seq,
        steps_this_call=steps_this_call,
        reason=reason,
        error=error_text,
        spent=guard.spent.to_dict(),
        pressure=round(guard.pressure(), 4),
        board=board.summary(),
        queued=[t.node for t in queue],
    )


def _checkpoint_now(
    run_id: int, seq: int, queue: list[Task], board: Blackboard, ctx: RunContext
) -> None:
    """Persist the run's state on the way out of any stop.

    Wrapped because a failure to checkpoint must not replace the error that
    caused the stop. Losing the checkpoint costs a resume; losing the original
    exception costs the ability to understand what went wrong at all.
    """
    try:
        store.save_checkpoint(
            run_id, seq=seq, node=ctx.node, queue=queue, state=board.to_dict()
        )
    except Exception:
        log.error("could not checkpoint run %s\n%s", ctx.run_key, traceback.format_exc())


def _summarise(outputs: list[Any]) -> Any:
    """Keep the step row readable. The full products are in `artifacts`."""
    out = []
    for value in outputs:
        if isinstance(value, str):
            out.append(truncate(value, 400))
        elif isinstance(value, dict):
            out.append({k: truncate(str(v), 200) for k, v in list(value.items())[:10]})
        elif isinstance(value, list):
            out.append({"length": len(value)})
        else:
            out.append(value)
    return out


# ------------------------------------------------------------------ control


def pause(run_key: str) -> bool:
    """Ask a running orchestrator to stop at the next safe point."""
    row = store.get_run(run_key)
    if row is None:
        raise LookupError(f"No run named {run_key}.")
    if RunStatus(str(row["status"])).terminal:
        return False
    store.set_control(run_key, Control.PAUSE)
    return True


def cancel(run_key: str) -> bool:
    row = store.get_run(run_key)
    if row is None:
        raise LookupError(f"No run named {run_key}.")
    status = RunStatus(str(row["status"]))
    if status.terminal:
        return False
    if status in {RunStatus.PENDING, RunStatus.PAUSED, RunStatus.WAITING}:
        # Nothing is executing, so there is nobody to notice the control flag.
        store.release_run(run_key, status=RunStatus.CANCELLED, reason="Cancelled while stopped.")
        return True
    store.set_control(run_key, Control.CANCEL)
    return True


def resume(run_key: str, *, budget: Budget | None = None, max_steps: int = 0) -> Outcome:
    """Continue a stopped run, optionally with a raised ceiling."""
    row = store.get_run(run_key)
    if row is None:
        raise LookupError(f"No run named {run_key}.")
    status = RunStatus(str(row["status"]))
    if status.terminal:
        return Outcome(run_key=run_key, status=status.value, claimed=False,
                       reason="This run has already finished.")
    if budget is not None:
        store.update_run(run_key, budget=budget.to_dict())
    store.set_control(run_key, Control.NONE)
    return run(run_key, max_steps=max_steps)


def answer(run_key: str, *, reply: str, note: str = "") -> bool:
    """Answer the approval a waiting run is parked on."""
    row = store.get_run(run_key)
    if row is None:
        raise LookupError(f"No run named {run_key}.")
    pending = store.pending_approval(int(row["id"]))
    if pending is None:
        return False
    store.answer_approval(int(pending["id"]), answer=reply, note=note)
    store.append_event(
        int(row["id"]),
        EventKind.APPROVAL_ANSWERED,
        node=str(pending["node"]),
        message=f"Answered '{reply}'",
        payload={"answer": reply, "note": note},
    )
    return True
