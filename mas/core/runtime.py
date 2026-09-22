"""What this process is actually allowed to do.

The same source tree runs in two very different places. On Vercel it is a
serverless function: the filesystem is read only apart from /tmp, there is no
way to keep a thread alive after the response is written, and the platform
kills the invocation at a fixed deadline. On a laptop, a Docker container or an
Oracle VM none of that is true and every feature is available.

Rather than fork the code, every module asks this one for permission. A feature
that cannot run here reports itself as unavailable with a reason, the API
surfaces that reason, and the UI greys the control out instead of failing at
call time. Adding a new hosting target means teaching `detect()` about it, not
editing the pipeline.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import util as importlib_util
from pathlib import Path
from typing import Any


class Tier(StrEnum):
    """How much control this process has over its own lifetime."""

    SERVERLESS = "serverless"
    SERVER = "server"


def _has_module(name: str) -> bool:
    """True when an import would succeed, without paying for the import.

    `find_spec` raises rather than returning None when a parent package is
    missing, which is exactly the case for an optional extra that was never
    installed, so the exception is part of the answer.
    """
    try:
        return importlib_util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw and raw.strip() else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Capability:
    """One thing the process may or may not be able to do."""

    id: str
    label: str
    available: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Runtime:
    tier: Tier
    platform: str
    python_version: str

    # Wall clock the platform allows for one invocation. The pipeline treats
    # this as a hard deadline and checkpoints before it is reached, because
    # being killed mid stage loses the work rather than pausing it.
    max_job_seconds: int

    # The only directory guaranteed writable. On Vercel this is /tmp, which is
    # scratch space shared by invocations that happen to land on the same
    # instance and is gone when that instance is archived.
    writable_dir: Path

    # Whether anything written to `writable_dir` will still be there on the
    # next run. When false, every cache is best effort and the database is the
    # only real storage.
    persistent_disk: bool

    capabilities: tuple[Capability, ...] = field(default_factory=tuple)

    def can(self, capability_id: str) -> bool:
        for cap in self.capabilities:
            if cap.id == capability_id:
                return cap.available
        return False

    def reason(self, capability_id: str) -> str:
        for cap in self.capabilities:
            if cap.id == capability_id:
                return "" if cap.available else (cap.reason or "Not available here.")
        return f"Unknown capability '{capability_id}'."

    def require(self, capability_id: str) -> None:
        """Raise a message worth showing a user, rather than an ImportError."""
        if not self.can(capability_id):
            raise CapabilityError(capability_id, self.reason(capability_id))

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "platform": self.platform,
            "python_version": self.python_version,
            "max_job_seconds": self.max_job_seconds,
            "writable_dir": str(self.writable_dir),
            "persistent_disk": self.persistent_disk,
            "capabilities": [c.as_dict() for c in self.capabilities],
        }


class CapabilityError(RuntimeError):
    """A feature was asked for that this deployment cannot provide."""

    def __init__(self, capability_id: str, reason: str) -> None:
        super().__init__(reason)
        self.capability_id = capability_id
        self.reason = reason


# Platform fingerprints, checked in order. The first environment variable that
# is present decides, so a more specific host is listed before a generic one.
_PLATFORM_MARKERS: tuple[tuple[str, str, Tier], ...] = (
    ("VERCEL", "vercel", Tier.SERVERLESS),
    ("AWS_LAMBDA_FUNCTION_NAME", "aws-lambda", Tier.SERVERLESS),
    ("FUNCTION_TARGET", "gcp-functions", Tier.SERVERLESS),
    ("K_SERVICE", "cloud-run", Tier.SERVER),
    ("NETLIFY", "netlify", Tier.SERVERLESS),
    ("RENDER", "render", Tier.SERVER),
    ("FLY_APP_NAME", "fly", Tier.SERVER),
    ("RAILWAY_ENVIRONMENT", "railway", Tier.SERVER),
    ("WEBSITE_INSTANCE_ID", "azure-appservice", Tier.SERVER),
    ("DYNO", "heroku", Tier.SERVER),
)

# Vercel kills a Hobby invocation at 300s and a Pro one at 800s. Stopping a
# little early leaves room to write the checkpoint and the response, so the
# next invocation resumes instead of repeating the stage.
_SERVERLESS_SAFETY_MARGIN = 25


def _detect_platform() -> tuple[str, Tier]:
    forced = os.environ.get("RUNTIME_TIER", "").strip().lower()
    for marker, name, tier in _PLATFORM_MARKERS:
        if os.environ.get(marker):
            if forced in {"serverless", "server"}:
                return name, Tier(forced)
            return name, tier
    if forced in {"serverless", "server"}:
        return "custom", Tier(forced)
    if _in_container():
        return "container", Tier.SERVER
    return "local", Tier.SERVER


def _in_container() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        # cgroup v1 names the container runtime on every line of this file.
        text = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return any(token in text for token in ("docker", "kubepods", "containerd", "podman"))


def _writable_dir(tier: Tier) -> Path:
    override = os.environ.get("WRITABLE_DIR", "").strip()
    if override:
        return Path(override)
    if tier is Tier.SERVERLESS:
        return Path("/tmp")
    # mas/core/runtime.py, so three parents up is the repository root.
    return Path(__file__).resolve().parent.parent.parent / "data"


def _max_job_seconds(tier: Tier) -> int:
    explicit = _env_int("MAX_JOB_SECONDS", 0)
    if explicit > 0:
        return explicit
    if tier is Tier.SERVERLESS:
        # Hobby's ceiling is the safe assumption. Set MAX_JOB_SECONDS=775 on a
        # Pro project to use the longer window.
        return 300 - _SERVERLESS_SAFETY_MARGIN
    return _env_int("SERVER_JOB_SECONDS", 3600)


def detect() -> Runtime:
    platform, tier = _detect_platform()
    writable = _writable_dir(tier)
    persistent = tier is Tier.SERVER and not _env_flag("EPHEMERAL_DISK", False)

    serverless = tier is Tier.SERVERLESS
    no_bg = (
        "This deployment is a serverless function. Work cannot continue after "
        "the response is sent, so long tasks are split across scheduled runs "
        "instead."
    )

    caps: list[Capability] = [
        Capability(
            id="background_threads",
            label="Background work after a response",
            available=not serverless,
            reason=no_bg if serverless else "",
        ),
        Capability(
            id="in_process_scheduler",
            label="Built in scheduler",
            available=not serverless,
            reason=(
                "Use Vercel Cron or an external trigger against /api/cron/tick."
                if serverless
                else ""
            ),
        ),
        Capability(
            id="headless_browser",
            label="Headless browser rendering",
            available=_headless_available(serverless),
            reason=_headless_reason(serverless),
        ),
        Capability(
            id="torch_models",
            label="PyTorch and sentence-transformers",
            available=(not serverless) and _has_module("sentence_transformers"),
            reason=_torch_reason(serverless),
        ),
        Capability(
            id="onnx_models",
            label="Local ONNX embedding model",
            available=_has_module("onnxruntime") and _has_module("tokenizers"),
            reason=(
                ""
                if _has_module("onnxruntime") and _has_module("tokenizers")
                else "onnxruntime and tokenizers are not installed in this environment."
            ),
        ),
        Capability(
            id="persistent_disk",
            label="Disk that survives a restart",
            available=persistent,
            reason=(
                "Only /tmp is writable here and it is cleared when the instance "
                "is recycled, so caches are rebuilt on demand."
                if not persistent
                else ""
            ),
        ),
        Capability(
            id="long_crawl",
            label="Unbounded crawling",
            available=not serverless,
            reason=(
                "Crawls here are capped to fit inside the function's time limit "
                "and continue on the next scheduled run."
                if serverless
                else ""
            ),
        ),
        Capability(
            id="local_llm",
            label="Local LLM server (Ollama, LM Studio, llama.cpp, vLLM)",
            available=not serverless,
            reason=(
                "A serverless function cannot reach a model server running on "
                "your own machine."
                if serverless
                else ""
            ),
        ),
        Capability(
            id="outbound_http",
            label="Outbound HTTP",
            available=True,
        ),
    ]

    return Runtime(
        tier=tier,
        platform=platform,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        max_job_seconds=_max_job_seconds(tier),
        writable_dir=writable,
        persistent_disk=persistent,
        capabilities=tuple(caps),
    )


def _headless_available(serverless: bool) -> bool:
    if serverless:
        return False
    if not _has_module("playwright"):
        return False
    # The Python package installs without the browsers, and that is the common
    # half configured state, so the binary itself is what is checked.
    return _playwright_browser_present()


def _playwright_browser_present() -> bool:
    explicit = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    roots: list[Path] = []
    if explicit and explicit != "0":
        roots.append(Path(explicit))
    else:
        home = Path.home()
        roots.extend(
            [
                home / ".cache" / "ms-playwright",
                home / "AppData" / "Local" / "ms-playwright",
                home / "Library" / "Caches" / "ms-playwright",
            ]
        )
    for root in roots:
        try:
            if root.is_dir() and any(root.glob("chromium*")):
                return True
        except OSError:
            continue
    return bool(shutil.which("chromium") or shutil.which("chromium-browser"))


def _headless_reason(serverless: bool) -> str:
    if serverless:
        return (
            "A browser engine does not fit in a serverless bundle. Run the "
            "Docker image or a local server to use this source type."
        )
    if not _has_module("playwright"):
        return "Install the server extras: pip install -r requirements-server.txt"
    if not _playwright_browser_present():
        return "Playwright is installed but its browsers are not. Run: playwright install chromium"
    return ""


def _torch_reason(serverless: bool) -> str:
    if serverless:
        return (
            "PyTorch is about 1 GB unpacked and exceeds the serverless bundle "
            "limit. The bundled ONNX model provides local embeddings instead."
        )
    if not _has_module("sentence_transformers"):
        return "Install the server extras: pip install -r requirements-server.txt"
    return ""


class Deadline:
    """A clock the pipeline checks so it stops rather than gets killed.

    A serverless invocation that overruns is terminated with no chance to save
    progress, which turns a partial success into a total loss. Every stage
    checks `remaining()` between units of work and checkpoints when the budget
    runs low.
    """

    def __init__(self, seconds: float | None = None, *, reserve: float = 5.0) -> None:
        budget = seconds if seconds is not None else float(current().max_job_seconds)
        self.started = time.monotonic()
        self.budget = max(0.0, budget - reserve)

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def remaining(self) -> float:
        return max(0.0, self.budget - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def allows(self, estimated_seconds: float) -> bool:
        """Whether there is room for a unit of work of roughly this size."""
        return self.remaining() > estimated_seconds


_cached: Runtime | None = None


def current() -> Runtime:
    global _cached
    if _cached is None:
        _cached = detect()
    return _cached


def reset() -> None:
    """Re-detect, used by tests that change the environment."""
    global _cached
    _cached = None
