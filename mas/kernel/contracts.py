"""The vocabulary every layer of the system shares.

Everything here is a plain dataclass that round trips through JSON. That is not
a style preference: a run has to survive the process that started it, so any
value that can sit on the blackboard between two steps must be serialisable
without custom logic. A dataclass with `to_dict` and `from_dict` is the whole
contract, and `checkpoint` relies on it holding for every type below.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from ..core.util import now_iso


def new_id(prefix: str) -> str:
    """A short, readable, collision resistant id.

    Twelve hex characters is 48 bits, which is far more than enough inside one
    run and still short enough to read aloud when debugging a trace.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    # The run is blocked on a person answering an approval gate.
    WAITING = "waiting"

    @property
    def terminal(self) -> bool:
        return self in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}

    @property
    def resumable(self) -> bool:
        return self in {RunStatus.PAUSED, RunStatus.WAITING, RunStatus.PENDING}


class Control(StrEnum):
    """What the operator has asked a running orchestrator to do.

    This is a database column rather than a thread event because the process
    that starts a run is often not the process that stops it: the web UI, the
    CLI and a scheduled job all have to be able to pause the same run.
    """

    NONE = "none"
    PAUSE = "pause"
    CANCEL = "cancel"


class EventKind(StrEnum):
    """Every kind of thing the trace can record.

    The UI renders from this enum alone, so adding a kind means teaching the
    timeline about it rather than inventing a string at the call site.
    """

    RUN_STARTED = "run.started"
    RUN_RESUMED = "run.resumed"
    RUN_PAUSED = "run.paused"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"

    NODE_ENTER = "node.enter"
    NODE_EXIT = "node.exit"
    NODE_ERROR = "node.error"
    NODE_RETRY = "node.retry"

    AGENT_START = "agent.start"
    AGENT_THOUGHT = "agent.thought"
    AGENT_ACTION = "agent.action"
    AGENT_OBSERVATION = "agent.observation"
    AGENT_FINISH = "agent.finish"
    AGENT_MESSAGE = "agent.message"

    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    TOOL_ERROR = "tool.error"

    ARTIFACT = "artifact.created"
    EVIDENCE = "evidence.added"

    BUDGET_WARN = "budget.warn"
    BUDGET_BREACH = "budget.breach"
    MODEL_DOWNGRADE = "model.downgrade"

    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_ANSWERED = "approval.answered"

    LOG = "log"


class ArtifactKind(StrEnum):
    """The typed products a run accumulates, in roughly the order they appear."""

    PLAN = "plan"
    SUBQUESTION = "subquestion"
    FINDING = "finding"
    CLAIM = "claim"
    VERDICT = "verdict"
    DRAFT = "draft"
    CRITIQUE = "critique"
    REPORT = "report"


@dataclass
class Budget:
    """Hard ceilings on one run.

    A breach pauses the run, it does not kill it. That distinction is the whole
    point: work already paid for stays on the blackboard, and raising the
    ceiling and resuming costs nothing extra. A ceiling of 0 means unlimited,
    which is what an explicit `--no-limit` sets rather than a huge number.
    """

    max_steps: int = 120
    max_tokens: int = 400_000
    max_usd: float = 0.50
    max_seconds: int = 1800
    max_tool_calls: int = 80

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Budget:
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Spend:
    """What the run has consumed so far, mirrored into the `runs` row."""

    steps: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    tool_calls: int = 0
    llm_calls: int = 0

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["tokens"] = self.tokens
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Spend:
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ToolSpec:
    """What an agent is told about a tool it may call.

    `parameters` is a JSON Schema object. It is rendered into the prompt for
    the universal protocol and handed over verbatim when a provider supports
    native function calling, so one definition serves both paths.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    # Tools that cost money or hit the network are metered against the run's
    # tool call ceiling. A pure local computation is not worth counting.
    metered: bool = True
    # A tool that changes something outside the run needs an approval gate.
    side_effects: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt_form(self) -> str:
        """How this tool is described inside the ReAct system prompt."""
        props = (self.parameters or {}).get("properties", {})
        required = set((self.parameters or {}).get("required", []))
        if not props:
            return f"- {self.name}: {self.description} (no arguments)"
        args = []
        for key, spec in props.items():
            kind = spec.get("type", "string")
            note = spec.get("description", "")
            flag = "required" if key in required else "optional"
            args.append(f"    - {key} ({kind}, {flag}): {note}")
        return f"- {self.name}: {self.description}\n  arguments:\n" + "\n".join(args)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: new_id("call"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    call_id: str
    name: str
    ok: bool
    content: Any = None
    error: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_observation(self, limit: int = 6000) -> str:
        """The string the agent actually reads back on the next turn."""
        if not self.ok:
            return f"ERROR from {self.name}: {self.error}"
        from ..core.util import dumps, truncate

        text = self.content if isinstance(self.content, str) else dumps(self.content)
        return truncate(text, limit)


@dataclass
class Message:
    """One agent addressing another, or the whole team.

    Agents do not call each other. They post here and the orchestrator decides
    who runs next, which is what keeps the graph in charge of control flow and
    stops the roles growing knowledge of one another.
    """

    sender: str
    content: Any
    recipient: str = "*"
    topic: str = ""
    in_reply_to: str = ""
    msg_id: str = field(default_factory=lambda: new_id("msg"))
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Artifact:
    """A typed product of the run, versioned rather than overwritten."""

    kind: str
    key: str
    title: str = ""
    body: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    parent_key: str = ""
    produced_by: str = ""
    version: int = 1
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Task:
    """One unit of work on the orchestrator's queue.

    The queue plus the blackboard is the entire resumable state of a run. A
    node that fans out returns several of these; a node that loops back returns
    one pointing at an earlier node. Neither case needs the orchestrator to
    understand the graph's shape, which is why loops and retries cost nothing
    extra here.
    """

    node: str
    payload: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    task_id: str = field(default_factory=lambda: new_id("task"))
    # Tasks with the same group id were fanned out together and may run in
    # parallel. The orchestrator drains one group per step.
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class NodeResult:
    """What a graph node hands back to the orchestrator."""

    output: Any = None
    # Tasks to enqueue next. Empty means this branch is finished, which is not
    # the same as the run being finished: other tasks may still be queued.
    next: list[Task] = field(default_factory=list)
    # Set by a terminal node to stop the run even if the queue is not empty.
    done: bool = False
    # Set to park the run until a person answers. The orchestrator persists the
    # question and stops without losing the queue.
    await_approval: dict[str, Any] | None = None
    messages: list[Message] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    action: str = ""


@dataclass
class AgentResult:
    """What one agent's reason and act loop produced."""

    output: Any = None
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    stopped_because: str = "finished"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "thoughts": self.thoughts,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "iterations": self.iterations,
            "stopped_because": self.stopped_because,
        }


class BudgetExceeded(RuntimeError):
    """Raised when a ceiling binds. Caught by the orchestrator, which pauses."""

    def __init__(self, limit: str, detail: str) -> None:
        super().__init__(detail)
        self.limit = limit
        self.detail = detail


class ProviderExhausted(RuntimeError):
    """The provider is out of quota, or is rate limiting past the point of retrying.

    Treated exactly like a budget breach: the run pauses and keeps everything.
    A free tier quota comes back on its own, so failing the run would throw
    away an hour of research over a limit that resets by itself. This is the
    single most common way a run on a free tier stops, which is why it gets its
    own type rather than being reported as an unexpected error.
    """

    def __init__(self, provider: str, detail: str, *, hint: str = "") -> None:
        super().__init__(detail)
        self.provider = provider
        self.detail = detail
        self.hint = hint


class RunCancelled(RuntimeError):
    """Raised when the operator asked for a cancel rather than a pause."""


class RunPaused(RuntimeError):
    """Raised when the operator asked for a pause mid step."""
