"""The run's nervous system: what happened, and who told whom.

Everything an agent does is announced here, and the announcement goes to the
database first. That ordering is the point. The database is the source of
truth, so a browser that reconnects halfway through a run replays the trace by
id cursor and misses nothing, and a run executed by the CLI is just as
watchable from the web UI as one the web UI started.

On top of that there is an in-process fan-out to any subscriber in the same
interpreter. It is an accelerator, not a channel: dropping it entirely would
cost latency and nothing else, which is why a slow or broken subscriber is
discarded rather than allowed to block the run.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Any

from . import store
from .contracts import EventKind, Message

log = logging.getLogger(__name__)

# Bounded so that a subscriber which stops reading (a closed browser tab whose
# generator was never finalised) costs a fixed amount of memory and is then
# dropped, rather than growing until the process dies.
SUBSCRIBER_QUEUE_SIZE = 512


class _Hub:
    """Process wide fan-out, keyed by run id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[int, list[queue.Queue]] = {}

    def subscribe(self, run_id: int) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            self._subs.setdefault(run_id, []).append(q)
        return q

    def unsubscribe(self, run_id: int, q: queue.Queue) -> None:
        with self._lock:
            listeners = self._subs.get(run_id)
            if not listeners:
                return
            if q in listeners:
                listeners.remove(q)
            if not listeners:
                self._subs.pop(run_id, None)

    def publish(self, run_id: int, event: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._subs.get(run_id, ()))
        for q in listeners:
            try:
                q.put_nowait(event)
            except queue.Full:
                # The reader has stopped draining. Its events are already
                # durable, so it can catch up from the database by cursor.
                self.unsubscribe(run_id, q)

    def subscriber_count(self, run_id: int) -> int:
        with self._lock:
            return len(self._subs.get(run_id, ()))


hub = _Hub()


class EventBus:
    """One run's view of the bus, with the run id and current node bound in."""

    def __init__(self, run_id: int, run_key: str) -> None:
        self.run_id = run_id
        self.run_key = run_key
        self.node = ""
        self.agent = ""
        self._seq = 0

    def bind(self, *, node: str = "", agent: str = "") -> None:
        """Set the context later events are attributed to."""
        if node:
            self.node = node
        self.agent = agent

    def emit(
        self,
        event_kind: EventKind | str,
        message: str = "",
        *,
        agent: str = "",
        node: str = "",
        level: str = "info",
        **payload: Any,
    ) -> int:
        """Record one thing that happened.

        The first parameter is `event_kind` rather than `kind` because "kind" is
        exactly what a payload wants to call an artifact's kind or a memory's
        kind, and a collision there is a TypeError raised from inside a run.
        """
        event_id = store.append_event(
            self.run_id,
            event_kind,
            agent=agent or self.agent,
            node=node or self.node,
            level=level,
            message=message,
            payload=payload,
        )
        kind_value = event_kind.value if isinstance(event_kind, EventKind) else str(event_kind)
        hub.publish(
            self.run_id,
            {
                "id": event_id,
                "run_key": self.run_key,
                "kind": kind_value,
                "agent": agent or self.agent,
                "node": node or self.node,
                "level": level,
                "message": message,
                "payload": payload,
            },
        )
        if level in {"warning", "error"}:
            log.log(
                logging.ERROR if level == "error" else logging.WARNING,
                "[%s] %s %s", self.run_key, kind_value, message,
            )
        return event_id

    def post(self, message: Message) -> None:
        """Record one agent addressing another.

        Kept separate from a plain event because this is the collaboration
        record: the UI draws the team from these, and a reviewer asking whether
        the agents actually talk to one another reads this table, not the log.
        """
        self._seq += 1
        store.post_message(self.run_id, message, seq=self._seq)
        self.emit(
            EventKind.AGENT_MESSAGE,
            f"{message.sender} to {message.recipient}: {message.topic}",
            agent=message.sender,
            recipient=message.recipient,
            topic=message.topic,
            body=message.content,
            msg_id=message.msg_id,
        )

    # Convenience wrappers. These exist so that call sites read as the thing
    # that happened rather than as a logging call with a kind argument.

    def thought(self, agent: str, text: str, *, iteration: int = 0) -> None:
        self.emit(EventKind.AGENT_THOUGHT, text, agent=agent, iteration=iteration)

    def action(self, agent: str, tool: str, arguments: Any, *, iteration: int = 0) -> None:
        self.emit(
            EventKind.AGENT_ACTION,
            f"{agent} calls {tool}",
            agent=agent,
            tool=tool,
            arguments=arguments,
            iteration=iteration,
        )

    def observation(self, agent: str, tool: str, summary: str, *, ok: bool = True) -> None:
        self.emit(
            EventKind.AGENT_OBSERVATION,
            summary,
            agent=agent,
            level="info" if ok else "warning",
            tool=tool,
            ok=ok,
        )

    def log(self, text: str, *, level: str = "info", **payload: Any) -> None:
        self.emit(EventKind.LOG, text, level=level, **payload)
