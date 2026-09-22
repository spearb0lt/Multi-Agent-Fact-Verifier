"""The command line. Start a run, stop it, resume it, read what it produced.

Everything here goes through the same store and orchestrator the web API uses,
so a run started in this terminal can be paused from the browser and resumed by
a scheduled job. There is no CLI specific state, which is the whole reason the
control channel is a database column.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from typing import Any

# A report about anything outside western Europe contains characters cp1252
# cannot encode, and a Windows console defaults to cp1252. Without this the run
# finishes, the report is correct in the database, and printing it raises
# UnicodeEncodeError, which reads as a crash at the very last step.
for stream in (sys.stdout, sys.stderr):
    with_reconfigure = getattr(stream, "reconfigure", None)
    if with_reconfigure is not None:
        with_reconfigure(encoding="utf-8", errors="replace")


COLOURS = {
    "agent": "\033[36m",
    "dim": "\033[2m",
    "warn": "\033[33m",
    "error": "\033[31m",
    "ok": "\033[32m",
    "bold": "\033[1m",
    "off": "\033[0m",
}


def _supports_colour() -> bool:
    return sys.stdout.isatty()


def paint(text: str, colour: str) -> str:
    if not _supports_colour():
        return text
    return f"{COLOURS.get(colour, '')}{text}{COLOURS['off']}"


# Which events are worth a line in a terminal. The trace holds everything; a
# person watching wants the shape of what is happening, not every field.
TRACE_STYLE: dict[str, tuple[str, str]] = {
    "run.started": ("ok", "RUN"),
    "run.resumed": ("ok", "RUN"),
    "run.paused": ("warn", "RUN"),
    "run.completed": ("ok", "RUN"),
    "run.failed": ("error", "RUN"),
    "run.cancelled": ("warn", "RUN"),
    "node.enter": ("bold", ">>"),
    "node.exit": ("dim", "<<"),
    "node.error": ("error", "!!"),
    "node.retry": ("warn", "~~"),
    "agent.start": ("agent", "AGENT"),
    "agent.thought": ("dim", "think"),
    "agent.finish": ("agent", "AGENT"),
    "agent.message": ("agent", "msg"),
    "tool.call": ("dim", "tool"),
    "tool.result": ("dim", "  ->"),
    "tool.error": ("warn", "  !!"),
    "evidence.added": ("ok", "src"),
    "artifact.created": ("ok", "made"),
    "budget.warn": ("warn", "$$"),
    "budget.breach": ("warn", "$$"),
    "model.downgrade": ("warn", "$$"),
    "approval.requested": ("warn", "ASK"),
    "log": ("dim", "..."),
}


def _print_event(event: dict[str, Any], *, verbose: bool) -> None:
    kind = event.get("kind", "")
    if kind not in TRACE_STYLE:
        return
    colour, tag = TRACE_STYLE[kind]
    if not verbose and kind in {"agent.thought", "tool.result", "node.exit", "log"}:
        return
    agent = event.get("agent") or ""
    message = event.get("message") or ""
    prefix = paint(f"{tag:>6}", colour)
    who = paint(f"{agent:<12}", "agent") if agent else " " * 12
    print(f"{prefix} {who} {message}"[:300], flush=True)


def _watch(run_id: int, stop: threading.Event, *, verbose: bool) -> None:
    from .kernel.bus import hub

    channel = hub.subscribe(run_id)
    try:
        while not stop.is_set():
            try:
                event = channel.get(timeout=0.25)
            except queue.Empty:
                continue
            _print_event(event, verbose=verbose)
    finally:
        hub.unsubscribe(run_id, channel)


# ---------------------------------------------------------------- commands


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check that everything this deployment needs is actually reachable."""
    from . import tools  # noqa: F401 - registers the tool set
    from .core import settings
    from .core.db import get_db
    from .core.embeddings import registry as embeddings
    from .core.llm import registry as llm
    from .core.runtime import current as runtime
    from .kernel.tool import registry as tool_registry
    from .tools.web import search_status

    ok = True
    print(paint(f"{settings.APP_NAME} doctor", "bold"))
    print()

    tier = runtime()
    print(f"runtime tier   {tier.tier}  (writable: {tier.writable_dir})")

    try:
        db = get_db()
        db.ensure_schema()
        count = db.scalar("SELECT COUNT(*) FROM runs")
        print(f"database       {paint('ok', 'ok')}  {db.dialect}, {count} run(s) stored")
    except Exception as exc:
        ok = False
        print(f"database       {paint('FAILED', 'error')}  {exc}")

    providers = [p for p in llm.provider_map().values() if p.is_available()]
    if providers:
        print(f"llm providers  {paint('ok', 'ok')}  {len(providers)} available")
        for provider in providers:
            print(f"               - {provider.id:<12} {provider.default_model()}")
    else:
        ok = False
        print(f"llm providers  {paint('NONE', 'error')}  set one key, for example GOOGLE_API_KEY")

    search = search_status()
    if search["usable"]:
        keyed = ", ".join(search["keyed"]) or "none"
        free = ", ".join(search["keyless"]) or "none"
        print(f"search         {paint('ok', 'ok')}  keyed: {keyed} | keyless: {free}")
    else:
        ok = False
        print(f"search         {paint('NONE', 'error')}  agents cannot research without one")

    try:
        embedder = embeddings.resolve()
        vec = embedder.embed(["probe"], operation="doctor").vectors
        print(f"embeddings     {paint('ok', 'ok')}  {embedder.id}, {len(vec[0])} dimensions")
    except Exception as exc:
        print(f"embeddings     {paint('degraded', 'warn')}  {exc}")
        print("               ranking falls back to lexical overlap, which is worse but works")

    print(f"tools          {paint('ok', 'ok')}  {', '.join(tool_registry.names())}")

    if args.model:
        print()
        print("calling the model to check the key really works...")
        try:
            reply = llm.generate_json(
                "Reply with the JSON object {\"ok\": true} and nothing else.",
                provider=args.provider or None,
                cheap=True,
                max_tokens=100,
                operation="doctor",
            )
            print(f"model call     {paint('ok', 'ok')}  {reply}")
        except Exception as exc:
            ok = False
            print(f"model call     {paint('FAILED', 'error')}  {exc}")

    print()
    print(paint("ready" if ok else "not ready", "ok" if ok else "error"))
    return 0 if ok else 1


def _budget_from(args: argparse.Namespace):
    from .core import settings
    from .kernel.contracts import Budget

    if getattr(args, "no_limit", False):
        return Budget(max_steps=0, max_tokens=0, max_usd=0, max_seconds=0, max_tool_calls=0)
    return Budget(
        max_steps=args.max_steps if args.max_steps is not None else settings.MAX_RUN_STEPS,
        max_tokens=args.max_tokens if args.max_tokens is not None else settings.MAX_RUN_TOKENS,
        max_usd=args.max_usd if args.max_usd is not None else settings.MAX_RUN_USD,
        max_seconds=args.max_seconds if args.max_seconds is not None else settings.MAX_RUN_SECONDS,
        max_tool_calls=(
            args.max_tool_calls if args.max_tool_calls is not None else settings.MAX_TOOL_CALLS
        ),
    )


def _execute(run_key: str, args: argparse.Namespace, *, resume: bool = False) -> int:
    from .kernel import orchestrator, store

    run_id = store.get_run_id(run_key)
    stop = threading.Event()
    watcher = None
    if not args.quiet:
        watcher = threading.Thread(
            target=_watch, args=(run_id, stop), kwargs={"verbose": args.verbose}, daemon=True
        )
        watcher.start()

    print(paint(f"run {run_key}", "bold"))
    print()
    try:
        if resume:
            outcome = orchestrator.resume(run_key, max_steps=args.max_run_steps or 0)
        else:
            outcome = orchestrator.run(run_key, max_steps=args.max_run_steps or 0)
    except KeyboardInterrupt:
        # Ctrl+C means stop, not abandon. The control flag makes the next safe
        # point pause the run, so nothing gathered is lost.
        orchestrator.pause(run_key)
        print()
        print(paint("interrupted, pausing at the next step...", "warn"))
        outcome = orchestrator.run(run_key, max_steps=1)
    finally:
        stop.set()
        if watcher:
            watcher.join(timeout=2)

    print()
    _print_outcome(outcome)
    return 0 if outcome["status"] in {"completed", "paused", "waiting"} else 1


def _print_outcome(outcome: dict[str, Any]) -> None:
    status = outcome["status"]
    colour = {"completed": "ok", "failed": "error"}.get(status, "warn")
    print(paint(f"status: {status}", colour))
    if outcome.get("reason"):
        print(f"reason: {outcome['reason']}")
    if outcome.get("error"):
        print(paint(f"error:  {outcome['error']}", "error"))

    spent = outcome.get("spent") or {}
    if spent:
        print(
            f"spent:  {spent.get('steps', 0)} steps, "
            f"{spent.get('tokens', 0):,} tokens, "
            f"{spent.get('llm_calls', 0)} model calls, "
            f"{spent.get('tool_calls', 0)} tool calls, "
            f"${spent.get('usd', 0):.4f}, "
            f"{spent.get('seconds', 0):.0f}s"
        )
    board = outcome.get("board") or {}
    counts = board.get("counts") or {}
    if counts:
        print(
            f"board:  {counts.get('findings', 0)} findings, "
            f"{counts.get('claims', 0)} claims, "
            f"{counts.get('supported', 0)} verified, "
            f"{counts.get('rejected', 0)} rejected"
        )
    if outcome.get("queued"):
        print(f"queued: {', '.join(outcome['queued'])}")
    if status in {"paused", "waiting"}:
        print()
        print(paint(f"resume with:  python -m mas.cli resume {outcome['run_key']}", "dim"))


def cmd_run(args: argparse.Namespace) -> int:
    from . import (
        tools,  # noqa: F401
        workflows,  # noqa: F401
    )
    from .core.llm import registry as llm
    from .kernel import store

    provider, model = args.provider or "", args.model or ""
    if not provider:
        provider, model = llm.default_selection()
        if not provider:
            print(paint("No AI provider is available. Run 'doctor' to see why.", "error"))
            return 1
    if args.model:
        model = args.model

    # Resolve the defaults now rather than at first use, so that what is printed
    # and what is stored on the run are the models actually about to be spent.
    cheap = ""
    try:
        chosen = llm.provider_map()[provider]
        model = model or chosen.default_model()
        cheap = chosen.cheap_model()
    except KeyError:
        print(paint(f"No provider called '{provider}'. Run 'doctor' to list them.", "error"))
        return 1

    config: dict[str, Any] = {"depth": args.depth}
    if args.max_revisions is not None:
        config["max_revisions"] = args.max_revisions
    if args.max_research_rounds is not None:
        config["max_research_rounds"] = args.max_research_rounds
    if args.concurrency is not None:
        config["agent_concurrency"] = args.concurrency
    if args.all_strong:
        config["role_tiers"] = dict.fromkeys(
            ["Planner", "Researcher", "Analyst", "FactChecker", "Supervisor",
             "Writer", "Editor", "Critic"], "strong"
        )
    if args.all_cheap:
        config["role_tiers"] = dict.fromkeys(
            ["Planner", "Researcher", "Analyst", "FactChecker", "Supervisor",
             "Writer", "Editor", "Critic"], "cheap"
        )

    run_key = store.create_run(
        brief=args.brief,
        workflow=args.workflow,
        provider=provider,
        model=model,
        cheap_model=cheap,
        config=config,
        budget=_budget_from(args),
    )
    print(f"brief:    {args.brief}")
    print(f"models:   {provider}/{model}   cheap: {cheap or 'same'}")
    print()
    return _execute(run_key, args)


def cmd_resume(args: argparse.Namespace) -> int:
    from . import tools, workflows  # noqa: F401
    from .core.llm import registry as llm
    from .kernel import store

    if store.get_run(args.run_key) is None:
        print("no such run")
        return 1

    if args.max_steps is not None or args.max_usd is not None or args.max_tokens is not None:
        store.update_run(args.run_key, budget=_budget_from(args).to_dict())

    # Switching provider on resume is the direct answer to the most common
    # reason a run stops: the free tier it was using ran out. The blackboard
    # does not care which model produced what is already on it.
    if args.provider or args.model:
        run = store.get_run(args.run_key)
        provider = args.provider or str(run["provider"])
        try:
            chosen = llm.provider_map()[provider]
        except KeyError:
            print(paint(f"No provider called '{provider}'. Run 'doctor' to list them.", "error"))
            return 1
        if not chosen.is_available():
            print(paint(f"{provider} is not usable here: {chosen.unavailable_reason()}", "error"))
            return 1
        model = args.model or chosen.default_model()
        store.update_run(
            args.run_key, provider=provider, model=model, cheap_model=chosen.cheap_model()
        )
        print(f"switched to {provider}/{model}")

    return _execute(args.run_key, args, resume=True)


def cmd_pause(args: argparse.Namespace) -> int:
    from .kernel import orchestrator

    changed = orchestrator.pause(args.run_key)
    print("pause requested" if changed else "that run is not running")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    from .kernel import orchestrator

    changed = orchestrator.cancel(args.run_key)
    print("cancelled" if changed else "that run has already finished")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    from .kernel import orchestrator

    if not orchestrator.answer(args.run_key, reply=args.reply, note=args.note or ""):
        print("that run is not waiting for an answer")
        return 1
    print(f"answered: {args.reply}")
    if args.then_resume:
        return _execute(args.run_key, args, resume=True)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    from .kernel import store

    runs = store.list_runs(limit=args.limit, status=args.status or "")
    if not runs:
        print("no runs yet")
        return 0
    print(f"{'RUN':<20} {'STATUS':<10} {'STEPS':>5} {'TOKENS':>9} {'USD':>8}  BRIEF")
    print("-" * 100)
    for run in runs:
        spent = run.get("spent") or {}
        print(
            f"{run['run_key']:<20} {run['status']:<10} "
            f"{spent.get('steps', 0):>5} {spent.get('tokens', 0):>9,} "
            f"{spent.get('usd', 0):>8.4f}  {(run['brief'] or '')[:46]}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .kernel import store

    run = store.get_run(args.run_key)
    if run is None:
        print("no such run")
        return 1
    run_id = int(run["id"])
    spent = run.get("spent") or {}
    budget = run.get("budget") or {}

    print(paint(run["run_key"], "bold"))
    print(f"brief      {run['brief']}")
    print(f"workflow   {run['workflow']}")
    print(f"status     {run['status']}   phase: {run['phase'] or '-'}")
    if run.get("pause_reason"):
        print(f"reason     {run['pause_reason']}")
    if run.get("error"):
        print(paint(f"error      {run['error']}", "error"))
    print(f"models     {run['provider']}/{run['model']}")
    print(
        f"spent      {spent.get('steps', 0)}/{budget.get('max_steps', 0)} steps, "
        f"{spent.get('tokens', 0):,}/{budget.get('max_tokens', 0):,} tokens, "
        f"${spent.get('usd', 0):.4f}/${budget.get('max_usd', 0)}"
    )
    print(f"sources    {store.evidence_count(run_id)}")
    usage = store.usage_by_agent(run_id)
    if usage:
        print()
        print(f"{'AGENT':<14} {'CALLS':>6} {'TOKENS':>9} {'USD':>9}")
        for row in usage:
            total = int(row["tokens_in"] or 0) + int(row["tokens_out"] or 0)
            print(
                f"{row['agent'] or '-':<14} {row['calls']:>6} {total:>9,} "
                f"{float(row['cost_usd'] or 0):>9.4f}"
            )
    tools_used = store.tool_usage(run_id)
    if tools_used:
        print()
        print(f"{'TOOL':<16} {'AGENT':<14} {'CALLS':>6} {'FAILED':>7}")
        for row in tools_used:
            print(
                f"{row['tool']:<16} {row['agent'] or '-':<14} "
                f"{row['calls']:>6} {row['failures'] or 0:>7}"
            )
    pending = store.pending_approval(run_id)
    if pending:
        print()
        print(paint(f"waiting for an answer: {pending['question']}", "warn"))
        print(f"options: {', '.join(pending['options']) or 'free text'}")
        print(paint(f"answer with:  python -m mas.cli answer {run['run_key']} <reply>", "dim"))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    from .kernel import store
    from .workflows.render import to_html

    run = store.get_run(args.run_key)
    if run is None:
        print("no such run")
        return 1
    report = store.get_artifact(int(run["id"]), "report", "report")
    if report is None:
        draft = store.get_artifact(int(run["id"]), "draft", "edited") or store.get_artifact(
            int(run["id"]), "draft", "draft"
        )
        if draft is None:
            print("that run has not produced a report yet")
            return 1
        print(paint("no final report, showing the latest draft", "warn"))
        print()
        report = draft

    if args.json:
        print(json.dumps({"title": report.title, **report.content, "markdown": report.body}, indent=2))
        return 0
    if args.html:
        html = to_html({"title": report.title, "markdown": report.body, **report.content})
        if args.out:
            from pathlib import Path

            Path(args.out).write_text(html, encoding="utf-8")
            print(f"written to {args.out}")
        else:
            print(html)
        return 0
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(report.body, encoding="utf-8")
        print(f"written to {args.out}")
        return 0
    print(report.body)
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    from .kernel import store

    run = store.get_run(args.run_key)
    if run is None:
        print("no such run")
        return 1
    events = store.list_events(int(run["id"]), limit=args.limit)
    for event in events:
        _print_event(
            {
                "kind": event["kind"],
                "agent": event["agent"],
                "message": event["message"],
            },
            verbose=True,
        )
    print()
    print(f"{len(events)} event(s)")
    return 0


def cmd_messages(args: argparse.Namespace) -> int:
    """The agent to agent traffic on its own, which is the collaboration record."""
    from .kernel import store

    run = store.get_run(args.run_key)
    if run is None:
        print("no such run")
        return 1
    for row in store.list_messages(int(run["id"])):
        body = row["content"]
        if isinstance(body, list):
            body = "; ".join(str(b) for b in body)
        sender = paint(f"{row['sender']:>13}", "agent")
        topic = f"[{row['topic']}] " if row["topic"] else ""
        print(f"{sender} -> {row['recipient']:<13} {topic}{str(body)[:120]}")
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from . import workflows  # noqa: F401
    from .kernel import graph as graph_module

    for graph in graph_module.available():
        described = graph.describe()
        print(paint(described["name"], "bold"))
        print(f"  {described['description']}")
        print()
        print(f"  {'NODE':<11} {'AGENT':<13} {'PARALLEL':<9} DESCRIPTION")
        for node in described["nodes"]:
            print(
                f"  {node['name']:<11} {node['agent']:<13} "
                f"{'yes' if node['parallel'] else '':<9} {node['description']}"
            )
        print()
        for edge in described["edges"]:
            condition = f"  ({edge['condition']})" if edge["condition"] else ""
            print(f"  {edge['source']:<11} -> {edge['target']}{condition}")
        print()
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    from .kernel import store

    print("deleted" if store.delete_run(args.run_key) else "no such run")
    return 0


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mas",
        description="A supervised team of agents that researches, verifies and writes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_watch_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("-v", "--verbose", action="store_true", help="show thoughts and tool results")
        p.add_argument("-q", "--quiet", action="store_true", help="no live trace")
        p.add_argument(
            "--max-run-steps", type=int, default=0,
            help="stop after this many steps in THIS invocation, then pause",
        )

    def add_budget_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--max-steps", type=int, default=None)
        p.add_argument("--max-tokens", type=int, default=None)
        p.add_argument("--max-usd", type=float, default=None)
        p.add_argument("--max-seconds", type=int, default=None)
        p.add_argument("--max-tool-calls", type=int, default=None)
        p.add_argument(
            "--no-limit", action="store_true",
            help="remove every ceiling. Only sensible with a free tier provider.",
        )

    p = sub.add_parser("doctor", help="check providers, search, embeddings and storage")
    p.add_argument("--model", action="store_true", help="also make one real model call")
    p.add_argument("--provider", default="")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("run", help="start a new run")
    p.add_argument("brief", help="what to research")
    p.add_argument("--workflow", default="research_report")
    p.add_argument("--provider", default="")
    p.add_argument("--model", default="")
    p.add_argument("--depth", choices=["quick", "standard", "deep"], default="standard")
    p.add_argument("--max-revisions", type=int, default=None)
    p.add_argument("--max-research-rounds", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--all-strong", action="store_true", help="use the strong model for every role")
    p.add_argument("--all-cheap", action="store_true", help="use the cheap model for every role")
    add_budget_flags(p)
    add_watch_flags(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue a paused or waiting run")
    p.add_argument("run_key")
    p.add_argument(
        "--provider", default="",
        help="continue on a different provider, for when the last one ran out of quota",
    )
    p.add_argument("--model", default="")
    add_budget_flags(p)
    add_watch_flags(p)
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("pause", help="ask a running run to stop at the next step")
    p.add_argument("run_key")
    p.set_defaults(func=cmd_pause)

    p = sub.add_parser("cancel", help="stop a run for good")
    p.add_argument("run_key")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("answer", help="answer a run waiting on an approval")
    p.add_argument("run_key")
    p.add_argument("reply")
    p.add_argument("--note", default="")
    p.add_argument("--then-resume", action="store_true")
    add_budget_flags(p)
    add_watch_flags(p)
    p.set_defaults(func=cmd_answer)

    p = sub.add_parser("list", help="every run")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--status", default="")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("status", help="one run in detail, with its cost breakdown")
    p.add_argument("run_key")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("show", help="print the report")
    p.add_argument("run_key")
    p.add_argument("--json", action="store_true")
    p.add_argument("--html", action="store_true")
    p.add_argument("--out", default="")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("trace", help="the full event trace")
    p.add_argument("run_key")
    p.add_argument("--limit", type=int, default=500)
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("messages", help="agent to agent traffic only")
    p.add_argument("run_key")
    p.set_defaults(func=cmd_messages)

    p = sub.add_parser("graph", help="show the workflow topology")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("delete", help="remove a run and everything it produced")
    p.add_argument("run_key")
    p.set_defaults(func=cmd_delete)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
