"""The FastAPI application.

Two behaviours here are worth more than they look.

Orphaned runs are reclaimed, on start up and then periodically. A process
killed mid run leaves its lease behind, and without this the run sits there
apparently executing with nobody executing it. The sweep has to keep running
rather than happen once, because a hosted free tier suspends an idle service
without caring that a run is in progress and wakes it again the moment somebody
visits, at which point the dead worker's heartbeat is still fresh enough to
look alive.

On shutdown, every executing run is asked to pause. The checkpoint discipline
means an abrupt exit loses at most the step in flight, but asking first usually
loses nothing at all.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core import settings
from ..core.llm import keyring
from . import worker
from .routes import router
from .worker import shutdown as shutdown_worker

log = logging.getLogger(__name__)

# How often to look for runs whose worker died. Shorter than the lease
# itself, so an orphan is picked up soon after it becomes one.
SWEEP_SECONDS = settings.env_int("ORPHAN_SWEEP_SECONDS", 30)

DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]


def reclaim_orphaned_runs() -> int:
    """Turn runs abandoned by a dead process back into resumable ones.

    Only runs whose lease has gone stale. A run marked running is not
    necessarily orphaned: the CLI in another terminal holds a perfectly live
    lease on one, and pausing it from here because this process happened to
    start would stop work that was going fine. The heartbeat is what tells the
    two apart, and it is why `claim_run` writes one.
    """
    from ..kernel import store
    from ..kernel.contracts import RunStatus

    reclaimed = 0
    for run in store.list_runs(limit=200, status=RunStatus.RUNNING.value):
        key = str(run["run_key"])
        if worker.is_running(key):
            continue
        if not store.lease_is_stale(run):
            log.info("run %s has a live lease elsewhere, leaving it alone", key)
            continue
        store.release_run(
            key,
            status=RunStatus.PAUSED,
            reason=(
                "The process executing this run stopped. Everything it had gathered "
                "was checkpointed, so resuming continues from the last completed step."
            ),
        )
        reclaimed += 1
    if reclaimed:
        log.warning("reclaimed %s run(s) left running by a dead process", reclaimed)
    return reclaimed


def _sweep_orphans(stop: threading.Event) -> None:
    """Keep looking for orphaned runs, not just at start up.

    Reclaiming only on start up leaves a hole exactly where a hosted free tier
    puts one. A platform that suspends an idle service does not care that a run
    is executing, because a background run generates no inbound traffic; and it
    wakes the service again the moment somebody visits, which can be seconds
    later. At that point the dead worker's heartbeat is still fresh, the lease
    reads as live, start up correctly declines to touch it, and the run sits
    marked running with nobody running it until the next restart.

    A periodic sweep closes that: whenever the lease does go stale, whichever
    process is up notices within a sweep interval and turns the run back into
    something resumable.
    """
    while not stop.wait(SWEEP_SECONDS):
        try:
            reclaim_orphaned_runs()
        except Exception:
            log.exception("orphan sweep failed")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    from ..core.db import get_db

    get_db().ensure_schema()
    # Importing these registers the tools and the workflows, so a request never
    # races an import and finds half a registry.
    from .. import tools, workflows  # noqa: F401

    reclaim_orphaned_runs()

    stop = threading.Event()
    sweeper = threading.Thread(
        target=_sweep_orphans, args=(stop,), name="mas-orphan-sweep", daemon=True
    )
    sweeper.start()

    yield

    stop.set()
    shutdown_worker(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_TAGLINE,
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS or DEV_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def gate_and_scope(request: Request, call_next):
        """Check the optional password, and scope any pasted keys to this request."""
        path = request.url.path
        if (
            settings.APP_PASSWORD
            and path.startswith("/api")
            and path not in {"/api/health"}
            and request.headers.get("X-App-Password") != settings.APP_PASSWORD
            and request.method != "OPTIONS"
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "This deployment is password protected."},
            )
        try:
            return await call_next(request)
        finally:
            # Credentials must not survive into the next request handled by
            # this worker, which is the whole hazard of a context variable.
            keyring.reset()

    @app.exception_handler(LookupError)
    async def not_found(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def bad_request(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    app.include_router(router)
    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the exported Next.js pages from this same process, when built.

    One process and one origin, which is what removes the whole class of
    problems that came with proxying: no CORS on the event stream, no API
    address baked into the bundle at build time, and one start command that is
    identical on a laptop, in Docker and on Render.

    When the frontend has not been built the API still runs, and the root path
    says how to build it rather than returning a bare 404.
    """
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    out = Path(__file__).resolve().parent.parent.parent / "out"
    index = out / "index.html"

    if not index.exists():
        @app.get("/")
        def root() -> dict[str, Any]:
            return {
                "app": settings.APP_NAME,
                "tagline": settings.APP_TAGLINE,
                "api": "/api/config",
                "docs": "/docs",
                "frontend": "Not built. Run 'npm install && npm run build' to serve the UI here.",
            }
        return

    # Next's own assets are content hashed, so they are safe to cache hard.
    app.mount("/_next", StaticFiles(directory=out / "_next"), name="next-assets")

    # HEAD as well as GET: Next's client router prefetches links with HEAD, and
    # a route registered for GET alone answers those with 405, which fills the
    # browser console with errors on a page that is working perfectly.
    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    def frontend(path: str) -> Any:
        """Serve an exported page, falling back to the index for unknown paths."""
        if path.startswith(("api/", "docs", "openapi.json", "redoc")):
            # Handled by the router; reaching here means it genuinely is not a
            # route, so a 404 is the honest answer rather than the app shell.
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Not found.")

        candidate = (out / path).resolve()
        # Refuse anything that climbs out of the export directory.
        if out.resolve() not in candidate.parents and candidate != out.resolve():
            return FileResponse(index)

        if candidate.is_dir() and (candidate / "index.html").exists():
            return FileResponse(candidate / "index.html")
        if candidate.is_file():
            return FileResponse(candidate)
        # trailingSlash puts every page at <name>/index.html.
        page = out / path / "index.html"
        if page.exists():
            return FileResponse(page)
        return FileResponse(index)


app = create_app()
