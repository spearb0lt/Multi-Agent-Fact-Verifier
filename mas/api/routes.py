"""Every HTTP endpoint.

The API is deliberately thin. It creates runs, asks the worker to execute them,
signals the control channel, and reads back what the database already holds.
No agent logic lives here, which is what keeps the CLI and the web UI honestly
equivalent rather than two implementations that drift.
"""
from __future__ import annotations

import asyncio
import json
import queue
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

from ..core import settings
from ..core.embeddings import registry as embeddings
from ..core.llm import keyring
from ..core.llm import registry as llm
from ..core.runtime import current as runtime
from ..kernel import graph as graph_module
from ..kernel import orchestrator, store
from ..kernel.bus import hub
from ..kernel.contracts import Budget, RunStatus
from ..kernel.pacer import pacer
from ..kernel.tool import registry as tool_registry
from . import worker
from .schemas import AnswerIn, KeysIn, ResumeIn, RunIn

router = APIRouter(prefix="/api")


def _bind_request_keys(request: Request) -> None:
    """Attach any keys this request carried, for the duration of the request."""
    if not settings.ALLOW_CLIENT_KEYS:
        return
    raw = request.headers.get("X-Provider-Keys")
    if not raw:
        return
    try:
        payload = json.loads(raw)
    except ValueError:
        return
    keyring.bind(
        payload.get("keys") or {},
        payload.get("base_urls") or {},
        payload.get("accounts") or {},
    )


def _run_or_404(run_key: str) -> dict[str, Any]:
    run = store.get_run(run_key)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No run named {run_key}.")
    return run


# ------------------------------------------------------------------ system


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "app": settings.APP_NAME, "version": "1.0.0"}


@router.get("/config")
def config(request: Request) -> dict[str, Any]:
    """Everything the UI needs to render its controls, in one call."""
    from ..tools.web import search_status

    _bind_request_keys(request)
    tier = runtime()
    try:
        embedder = embeddings.resolve()
        embedding = {"id": embedder.id, "available": True}
    except Exception as exc:
        embedding = {"id": "", "available": False, "reason": str(exc)}

    return {
        "app": settings.APP_NAME,
        "tagline": settings.APP_TAGLINE,
        "runtime": {
            "tier": tier.tier.value,
            "persistent_disk": tier.persistent_disk,
            "max_job_seconds": tier.max_job_seconds,
        },
        "allow_client_keys": settings.ALLOW_CLIENT_KEYS,
        "providers": llm.all_status(),
        "default": dict(zip(("provider", "model"), llm.default_selection(), strict=False)),
        "embedding": embedding,
        "search": search_status(),
        "tools": [t.spec.to_dict() for t in tool_registry.all()],
        "workflows": [g.describe() for g in graph_module.available()],
        "defaults": {
            "max_steps": settings.MAX_RUN_STEPS,
            "max_tokens": settings.MAX_RUN_TOKENS,
            "max_usd": settings.MAX_RUN_USD,
            "max_seconds": settings.MAX_RUN_SECONDS,
            "max_tool_calls": settings.MAX_TOOL_CALLS,
            "max_revisions": settings.MAX_REVISIONS,
            "max_research_rounds": settings.MAX_RESEARCH_ROUNDS,
            "agent_concurrency": settings.AGENT_CONCURRENCY,
            "quality_bar": settings.QUALITY_BAR,
        },
        "pacing": pacer.snapshot(),
        "worker": {"active": worker.queue_depth(), "capacity": worker.MAX_CONCURRENT_RUNS},
    }


@router.get("/workflows")
def workflows() -> dict[str, Any]:
    return {"workflows": [g.describe() for g in graph_module.available()]}


@router.post("/keys/verify")
def verify_keys(payload: KeysIn) -> dict[str, Any]:
    """Check a pasted key actually works, without starting a run for it."""
    keyring.bind(
        payload.credentials.keys,
        payload.credentials.base_urls,
        payload.credentials.accounts,
    )
    try:
        return {"providers": llm.all_status(), "supplied": sorted(keyring.supplied())}
    finally:
        keyring.reset()


# --------------------------------------------------------------------- runs


@router.post("/runs", status_code=201)
def create_run(payload: RunIn, request: Request) -> dict[str, Any]:
    _bind_request_keys(request)
    if payload.credentials:
        keyring.bind(
            payload.credentials.keys,
            payload.credentials.base_urls,
            payload.credentials.accounts,
        )

    try:
        graph_module.get(payload.workflow)
    except LookupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    provider, model = payload.provider, payload.model
    if not provider:
        provider, model = llm.default_selection()
    if not provider:
        raise HTTPException(
            status_code=400,
            detail=(
                "No AI provider is available. Set a server key, or paste your own "
                "in Settings."
            ),
        )
    try:
        chosen = llm.provider_map()[provider]
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"No provider called '{provider}'.") from exc

    model = payload.model or model or chosen.default_model()

    config: dict[str, Any] = {"depth": payload.depth, "allow_degrade": payload.allow_degrade}
    for name in ("max_revisions", "max_research_rounds", "agent_concurrency"):
        value = getattr(payload, name)
        if value is not None:
            config[name] = value
    if payload.role_tiers:
        config["role_tiers"] = payload.role_tiers
    if payload.approve_plan:
        config["approve_plan"] = True

    run_key = store.create_run(
        brief=payload.brief.strip(),
        workflow=payload.workflow,
        provider=provider,
        model=model,
        cheap_model=chosen.cheap_model(),
        config=config,
        budget=(payload.budget or None).merged() if payload.budget else Budget(),
    )

    if payload.credentials:
        worker.remember_credentials(run_key, payload.credentials.model_dump())

    started = worker.start(run_key) if payload.start else {"started": False}
    return {"run_key": run_key, **store.get_run(run_key), **{"worker": started}}


@router.get("/runs")
def list_runs(limit: int = Query(default=30, ge=1, le=200), status: str = "") -> dict[str, Any]:
    runs = store.list_runs(limit=limit, status=status)
    for run in runs:
        run["executing_here"] = worker.is_running(str(run["run_key"]))
        if run["status"] == "running":
            run["spent"] = store.live_spend(int(run["id"]), run)
    return {"runs": runs, "worker": {"active": worker.queue_depth()}}


@router.get("/runs/{run_key}")
def get_run(run_key: str) -> dict[str, Any]:
    run = _run_or_404(run_key)
    run_id = int(run["id"])
    latest = store.latest_checkpoint(run_id)
    return {
        **run,
        # The itemised tables, not the step boundary snapshot, so a viewer sees
        # the cost climbing while a long step is still running.
        "spent": store.live_spend(run_id, run),
        "executing_here": worker.is_running(run_key),
        "has_client_keys": worker.has_credentials(run_key),
        "evidence_count": store.evidence_count(run_id),
        "usage_by_agent": store.usage_by_agent(run_id),
        "tool_usage": store.tool_usage(run_id),
        "pending_approval": store.pending_approval(run_id),
        "board": (latest or {}).get("state", {}),
        "queued": [t.node for t in (latest or {}).get("queue", [])],
        "graph": graph_module.get(str(run["workflow"])).describe(),
    }


@router.post("/runs/{run_key}/start")
def start_run(run_key: str) -> dict[str, Any]:
    _run_or_404(run_key)
    return worker.start(run_key)


@router.post("/runs/{run_key}/pause")
def pause_run(run_key: str) -> dict[str, Any]:
    _run_or_404(run_key)
    return {"paused": orchestrator.pause(run_key), "run_key": run_key}


@router.post("/runs/{run_key}/cancel")
def cancel_run(run_key: str) -> dict[str, Any]:
    _run_or_404(run_key)
    changed = orchestrator.cancel(run_key)
    worker.forget_credentials(run_key)
    return {"cancelled": changed, "run_key": run_key}


@router.post("/runs/{run_key}/resume")
def resume_run(run_key: str, payload: ResumeIn | None = None) -> dict[str, Any]:
    run = _run_or_404(run_key)
    if RunStatus(str(run["status"])).terminal:
        raise HTTPException(status_code=409, detail="That run has already finished.")
    if payload and payload.credentials:
        worker.remember_credentials(run_key, payload.credentials.model_dump())

    if payload and (payload.provider or payload.model):
        provider_id = payload.provider or str(run["provider"])
        try:
            chosen = llm.provider_map()[provider_id]
        except KeyError as exc:
            raise HTTPException(
                status_code=400, detail=f"No provider called '{provider_id}'."
            ) from exc
        if not chosen.is_available():
            raise HTTPException(status_code=400, detail=chosen.unavailable_reason())
        store.update_run(
            run_key,
            provider=provider_id,
            model=payload.model or chosen.default_model(),
            cheap_model=chosen.cheap_model(),
        )

    budget = payload.budget.merged() if payload and payload.budget else None
    return worker.resume_with_budget(run_key, budget)


@router.post("/runs/{run_key}/answer")
def answer_run(run_key: str, payload: AnswerIn) -> dict[str, Any]:
    _run_or_404(run_key)
    if not orchestrator.answer(run_key, reply=payload.reply, note=payload.note):
        raise HTTPException(status_code=409, detail="That run is not waiting for an answer.")
    started = worker.start(run_key, resume=True) if payload.resume else {"started": False}
    return {"answered": True, "worker": started}


@router.delete("/runs/{run_key}")
def delete_run(run_key: str) -> dict[str, Any]:
    if worker.is_running(run_key):
        raise HTTPException(status_code=409, detail="Stop the run before deleting it.")
    worker.forget_credentials(run_key)
    return {"deleted": store.delete_run(run_key)}


# ------------------------------------------------------------------ reading


@router.get("/runs/{run_key}/events")
def run_events(
    run_key: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=300, ge=1, le=2000),
) -> dict[str, Any]:
    run = _run_or_404(run_key)
    events = store.list_events(int(run["id"]), after_id=after, limit=limit)
    return {
        "events": events,
        "cursor": events[-1]["id"] if events else after,
        "status": run["status"],
    }


@router.get("/runs/{run_key}/stream")
async def stream_events(run_key: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
    """Server sent events for the live trace.

    The backlog is replayed from the database first and only then does the
    stream switch to the in-process feed, so a browser that connects late, or
    reconnects, sees the whole run rather than whatever happened to arrive
    after it opened the socket.
    """
    run = _run_or_404(run_key)
    run_id = int(run["id"])

    async def generator():
        cursor = after
        backlog = store.list_events(run_id, after_id=cursor, limit=2000)
        for event in backlog:
            cursor = int(event["id"])
            yield f"id: {cursor}\ndata: {json.dumps(event, default=str)}\n\n"

        channel = hub.subscribe(run_id)
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    event = await loop.run_in_executor(None, channel.get, True, 15.0)
                except queue.Empty:
                    fresh = store.get_run(run_key)
                    if fresh and RunStatus(str(fresh["status"])) is not RunStatus.RUNNING:
                        yield f"event: done\ndata: {json.dumps({'status': fresh['status']})}\n\n"
                        return
                    # Comment frames keep proxies from closing an idle stream.
                    yield ": keepalive\n\n"
                    continue
                if int(event.get("id", 0)) <= cursor:
                    continue
                cursor = int(event.get("id", cursor))
                yield f"id: {cursor}\ndata: {json.dumps(event, default=str)}\n\n"
                # A terminal run event closes the stream. "started" and
                # "resumed" are the two run events that mean the opposite.
                kind = str(event.get("kind", ""))
                if kind.startswith("run.") and kind not in {"run.started", "run.resumed"}:
                    yield f"event: done\ndata: {json.dumps({'status': kind})}\n\n"
                    return
        finally:
            hub.unsubscribe(run_id, channel)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/runs/{run_key}/steps")
def run_steps(run_key: str) -> dict[str, Any]:
    run = _run_or_404(run_key)
    return {"steps": store.list_steps(int(run["id"]))}


@router.get("/runs/{run_key}/messages")
def run_messages(run_key: str) -> dict[str, Any]:
    run = _run_or_404(run_key)
    return {"messages": store.list_messages(int(run["id"]))}


@router.get("/runs/{run_key}/artifacts")
def run_artifacts(run_key: str, kind: str = "") -> dict[str, Any]:
    run = _run_or_404(run_key)
    artifacts = store.list_artifacts(int(run["id"]), kind=kind)
    return {"artifacts": [a.to_dict() for a in artifacts]}


@router.get("/runs/{run_key}/evidence")
def run_evidence(run_key: str) -> dict[str, Any]:
    run = _run_or_404(run_key)
    rows = store.list_evidence(int(run["id"]))
    for row in rows:
        # The body is thousands of characters and the UI shows a snippet, so
        # sending it to every browser on every poll is pure waste.
        row.pop("embedding", None)
        row["body"] = (row.get("body") or "")[:600]
    return {"evidence": rows}


@router.get("/runs/{run_key}/report")
def run_report(run_key: str, format: str = Query(default="json", pattern="^(json|markdown|html)$")):
    from ..workflows.render import to_html

    run = _run_or_404(run_key)
    run_id = int(run["id"])
    artifact = store.get_artifact(run_id, "report", "report")
    if artifact is None:
        artifact = store.get_artifact(run_id, "draft", "edited") or store.get_artifact(
            run_id, "draft", "draft"
        )
    if artifact is None:
        raise HTTPException(status_code=404, detail="That run has no report yet.")

    payload = {"title": artifact.title, "markdown": artifact.body, **artifact.content}
    if format == "markdown":
        return PlainTextResponse(artifact.body, media_type="text/markdown; charset=utf-8")
    if format == "html":
        return PlainTextResponse(to_html(payload), media_type="text/html; charset=utf-8")
    return {"report": payload, "final": artifact.kind == "report"}
