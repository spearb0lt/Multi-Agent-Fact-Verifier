"""Running an agent team in the background of a web process.

A run takes minutes, so an HTTP request cannot wait for it. The API starts a
worker thread and returns immediately, and everything the browser needs after
that comes from the database: status, trace, spend, artifacts. The worker does
not talk to the request that started it, which is exactly why closing the tab,
restarting the browser or moving to another machine loses nothing.

There are two things worth knowing about the boundary.

Credentials do not cross it by themselves. The provider keyring is a context
variable, and a new thread gets a fresh context, so a visitor's own key has to
be handed over explicitly and rebound inside the worker. Those keys are kept in
memory only, never written to the database, which means a run resumed after a
process restart falls back to the server's own keys. That is the right trade,
but it has to be stated rather than discovered.

And the pool is small on purpose. Every run here spends a rate limited free
tier, so running six at once mostly produces six runs being throttled.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from ..core import settings
from ..core.llm import keyring
from ..kernel import orchestrator, store
from ..kernel.contracts import Budget

log = logging.getLogger(__name__)

MAX_CONCURRENT_RUNS = settings.env_int("MAX_CONCURRENT_RUNS", 2)

_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RUNS, thread_name_prefix="mas-run")
_lock = threading.RLock()
_active: dict[str, Future] = {}
# Visitor supplied credentials, in memory only and deliberately never persisted.
_credentials: dict[str, dict[str, Any]] = {}


def remember_credentials(run_key: str, payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    with _lock:
        _credentials[run_key] = {
            "keys": dict(payload.get("keys") or {}),
            "base_urls": dict(payload.get("base_urls") or {}),
            "accounts": dict(payload.get("accounts") or {}),
        }


def forget_credentials(run_key: str) -> None:
    with _lock:
        _credentials.pop(run_key, None)


def has_credentials(run_key: str) -> bool:
    with _lock:
        return run_key in _credentials


def _execute(run_key: str, resume: bool) -> dict[str, Any]:
    with _lock:
        credentials = _credentials.get(run_key)
    if credentials:
        keyring.bind(
            credentials.get("keys"),
            credentials.get("base_urls"),
            credentials.get("accounts"),
        )
    try:
        if resume:
            return orchestrator.resume(run_key)
        return orchestrator.run(run_key)
    except Exception:
        log.exception("worker crashed on run %s", run_key)
        raise
    finally:
        keyring.reset()
        with _lock:
            _active.pop(run_key, None)


def start(run_key: str, *, resume: bool = False) -> dict[str, Any]:
    """Queue a run. Returns what the caller should show, not what it produced."""
    with _lock:
        if run_key in _active and not _active[run_key].done():
            return {"started": False, "reason": "That run is already executing here."}

    run = store.get_run(run_key)
    if run is None:
        raise LookupError(f"No run named {run_key}.")

    from ..kernel.contracts import RunStatus

    status = RunStatus(str(run["status"]))
    if status.terminal:
        return {"started": False, "reason": f"That run has already {status.value}."}

    with _lock:
        future = _pool.submit(_execute, run_key, resume)
        _active[run_key] = future

    return {"started": True, "run_key": run_key, "queued": queue_depth()}


def is_running(run_key: str) -> bool:
    with _lock:
        future = _active.get(run_key)
        return bool(future and not future.done())


def queue_depth() -> int:
    with _lock:
        return sum(1 for f in _active.values() if not f.done())


def active_runs() -> list[str]:
    with _lock:
        return [key for key, future in _active.items() if not future.done()]


def resume_with_budget(run_key: str, budget: Budget | None) -> dict[str, Any]:
    if budget is not None:
        store.update_run(run_key, budget=budget.to_dict())
    return start(run_key, resume=True)


def shutdown(wait: bool = False) -> None:
    """Stop accepting work. Running runs are asked to pause so nothing is lost."""
    for run_key in active_runs():
        try:
            orchestrator.pause(run_key)
        except Exception:
            log.warning("could not signal pause to %s during shutdown", run_key)
    _pool.shutdown(wait=wait, cancel_futures=not wait)
