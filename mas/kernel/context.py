"""Everything one run carries, in a single object the agents are handed.

Agents receive a context rather than eight collaborators. That is not just
convenience: it means an agent cannot quietly acquire a dependency the
orchestrator does not know about, because the only things reachable from here
are the things a run is allowed to touch. A role that wanted its own database
handle or its own provider client would have to take it as an argument, and
that shows up in review.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core import settings
from ..core.util import truncate
from . import store as store_module
from .blackboard import Blackboard
from .budget import BudgetGuard
from .bus import EventBus
from .contracts import Artifact, Control, EventKind, Message, RunCancelled, RunPaused
from .llm import RunLLM
from .policy import ModelPolicy
from .tool import ToolRegistry
from .tool import registry as default_registry


@dataclass
class RunContext:
    """The world as one run sees it."""

    run_key: str
    run_id: int
    brief: str
    board: Blackboard
    bus: EventBus
    budget: BudgetGuard
    policy: ModelPolicy
    config: dict[str, Any] = field(default_factory=dict)
    tools: ToolRegistry = field(default=default_registry)
    store: Any = field(default=store_module)
    # The step currently executing. Written by the orchestrator so that usage
    # rows and tool calls can be attributed without threading it through every
    # signature between here and the call site.
    seq: int = 0
    node: str = ""
    llm: RunLLM = field(init=False)

    def __post_init__(self) -> None:
        self.llm = RunLLM(self)
        self._control_checked = 0.0
        self._control = Control.NONE

    # ---------------------------------------------------------------- control

    def check_control(self, *, force: bool = False) -> None:
        """Raise if the operator has asked this run to stop.

        Called from inside the reason and act loop as well as between steps, so
        that pressing pause during a researcher's eighth tool call takes effect
        on the next turn instead of after the whole agent finishes. The read is
        rate limited because it is a database round trip and a tight agent loop
        would otherwise spend more time asking whether to stop than working.
        """
        import time

        now = time.monotonic()
        interval = float(self.option("control_poll_seconds", 1.0))
        if not force and (now - self._control_checked) < interval:
            control = self._control
        else:
            control = self.store.read_control(self.run_key)
            self._control = control
            self._control_checked = now

        if control is Control.CANCEL:
            raise RunCancelled("The operator cancelled this run.")
        if control is Control.PAUSE:
            raise RunPaused("The operator paused this run.")

    # ---------------------------------------------------------------- options

    def option(self, name: str, default: Any = None) -> Any:
        """A run level setting, falling back to the deployment's default."""
        if name in self.config:
            return self.config[name]
        return getattr(settings, name.upper(), default)

    @property
    def depth(self) -> str:
        """How thorough this run should be. Set once, read by several agents."""
        value = str(self.config.get("depth", "standard")).lower()
        return value if value in {"quick", "standard", "deep"} else "standard"

    @property
    def breadth(self) -> int:
        """How many subquestions the Planner should produce.

        Scaled down under budget pressure, because the cheapest way to finish a
        run that is running out of room is to research fewer things properly
        rather than more things badly.
        """
        base = {"quick": 3, "standard": 5, "deep": 8}[self.depth]
        if self.budget.critical:
            return max(2, base // 2)
        if self.budget.degraded:
            return max(2, int(base * 0.7))
        return base

    # -------------------------------------------------------------- artifacts

    def emit_artifact(self, artifact: Artifact) -> Artifact:
        self.store.put_artifact(self.run_id, artifact)
        self.bus.emit(
            EventKind.ARTIFACT,
            f"{artifact.produced_by or 'run'} produced {artifact.kind} '{artifact.title or artifact.key}'",
            agent=artifact.produced_by,
            kind=artifact.kind,
            key=artifact.key,
            version=artifact.version,
            title=artifact.title,
        )
        return artifact

    def send(self, message: Message) -> None:
        self.bus.post(message)

    # --------------------------------------------------------------- evidence

    def add_evidence(self, item: dict[str, Any]) -> tuple[str, bool]:
        ref, is_new = self.store.add_evidence(self.run_id, item)
        if is_new:
            self.bus.emit(
                EventKind.EVIDENCE,
                f"[{ref}] {truncate(str(item.get('title') or item.get('url', '')), 120)}",
                agent=item.get("found_by", ""),
                ref=ref,
                url=item.get("url", ""),
                domain=item.get("domain", ""),
            )
        return ref, is_new

    def evidence(self, refs: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        return self.store.list_evidence(self.run_id, refs=refs)

    def evidence_index(self) -> dict[str, dict[str, Any]]:
        return {row["ref"]: row for row in self.evidence()}

    # ----------------------------------------------------------------- memory

    def recall(self, kind: str = "", limit: int = 12) -> list[dict[str, Any]]:
        return self.store.recall(scope="global", kind=kind, limit=limit)

    def remember(self, *, kind: str, key: str, content: str, **meta: Any) -> None:
        self.store.remember(scope="global", kind=kind, key=key, content=content, meta=meta)

    # ----------------------------------------------------------------- status

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "node": self.node,
            "seq": self.seq,
            "board": self.board.summary(),
            "budget": self.budget.snapshot(),
        }
