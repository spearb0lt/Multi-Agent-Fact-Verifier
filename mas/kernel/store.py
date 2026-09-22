"""Every read and write a run needs, in one place.

The rest of the kernel never touches SQL. That matters more here than in a
normal application because the database is not a side effect of running an
agent, it *is* the run: the control channel that stops it, the checkpoint that
resumes it and the trace the UI renders all live in these tables.

JSON columns are dumped on write and read back through `util.loads`, following
the convention the storage engine was built for.
"""
from __future__ import annotations

import os
import socket
from typing import Any

from ..core.db import get_db
from ..core.db.engine import Row
from ..core.util import dumps, loads, now_iso, utcnow
from ..core.util import url_key as make_url_key
from .contracts import (
    Artifact,
    Budget,
    Control,
    EventKind,
    Message,
    RunStatus,
    Spend,
    Task,
    new_id,
)

# A run claimed by a process that then died would stay locked forever, so a
# claim is a lease: no heartbeat for this long and another process may take it.
STALE_LEASE_SECONDS = 90


def owner_tag() -> str:
    """Who holds a run's lease. Host and pid is enough to tell two workers apart."""
    return f"{socket.gethostname()}:{os.getpid()}"


def _json(value: Any) -> str:
    return dumps(value if value is not None else {})


# ------------------------------------------------------------------- runs


def create_run(
    *,
    brief: str,
    workflow: str = "research_report",
    title: str = "",
    provider: str = "",
    model: str = "",
    cheap_model: str = "",
    config: dict[str, Any] | None = None,
    budget: Budget | None = None,
) -> str:
    db = get_db()
    run_key = new_id("run")
    stamp = now_iso()
    db.insert(
        """
        INSERT INTO runs (
            run_key, brief, title, workflow, status, control, phase,
            provider, model, cheap_model, config, budget, spent, next_seq,
            created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_key,
            brief,
            title,
            workflow,
            RunStatus.PENDING.value,
            Control.NONE.value,
            "",
            provider,
            model,
            cheap_model,
            _json(config or {}),
            _json((budget or Budget()).to_dict()),
            _json(Spend().to_dict()),
            0,
            stamp,
            stamp,
        ),
    )
    return run_key


def _hydrate(row: Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    out["config"] = loads(row["config"], {}) or {}
    out["budget"] = loads(row["budget"], {}) or {}
    out["spent"] = loads(row["spent"], {}) or {}
    return out


def get_run(run_key: str) -> dict[str, Any] | None:
    return _hydrate(get_db().query_one("SELECT * FROM runs WHERE run_key = ?", (run_key,)))


def get_run_id(run_key: str) -> int:
    value = get_db().scalar("SELECT id FROM runs WHERE run_key = ?", (run_key,))
    if not value:
        raise LookupError(f"No run named {run_key}.")
    return int(value)


def list_runs(*, limit: int = 50, status: str = "") -> list[dict[str, Any]]:
    sql = "SELECT * FROM runs"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return [_hydrate(row) for row in get_db().query(sql, params)]  # type: ignore[misc]


def update_run(run_key: str, **fields: Any) -> None:
    if not fields:
        return
    json_fields = {"config", "budget", "spent"}
    sets, values = [], []
    for key, value in fields.items():
        sets.append(f"{key} = ?")
        values.append(_json(value) if key in json_fields else value)
    sets.append("updated_at = ?")
    values.append(now_iso())
    values.append(run_key)
    get_db().execute(f"UPDATE runs SET {', '.join(sets)} WHERE run_key = ?", values)


def set_control(run_key: str, control: Control | str) -> None:
    """Ask a running orchestrator to stop. Safe from any process."""
    value = control.value if isinstance(control, Control) else str(control)
    get_db().execute(
        "UPDATE runs SET control = ?, updated_at = ? WHERE run_key = ?",
        (value, now_iso(), run_key),
    )


def read_control(run_key: str) -> Control:
    raw = get_db().scalar("SELECT control FROM runs WHERE run_key = ?", (run_key,))
    try:
        return Control(str(raw or "none"))
    except ValueError:
        return Control.NONE


def heartbeat(run_key: str, *, phase: str = "", spent: Spend | None = None) -> None:
    fields: dict[str, Any] = {"heartbeat_at": now_iso()}
    if phase:
        fields["phase"] = phase
    if spent is not None:
        fields["spent"] = spent.to_dict()
    update_run(run_key, **fields)


def _lease_is_stale(row: dict[str, Any]) -> bool:
    from ..core.util import parse_datetime

    beat = parse_datetime(row.get("heartbeat_at"))
    if beat is None:
        return True
    return (utcnow() - beat).total_seconds() > STALE_LEASE_SECONDS


def claim_run(run_key: str) -> bool:
    """Take the lease on a run, or report that somebody else holds it.

    The UPDATE's WHERE clause is the lock. Two workers racing both issue it,
    the database serialises them, and the loser's clause no longer matches
    because the winner already moved the row out of a claimable status.
    """
    run = get_run(run_key)
    if run is None:
        raise LookupError(f"No run named {run_key}.")

    status = RunStatus(str(run["status"]))
    if status.terminal:
        return False
    if status is RunStatus.RUNNING and not _lease_is_stale(run):
        return False

    me = owner_tag()
    stamp = now_iso()
    get_db().execute(
        """
        UPDATE runs
           SET status = ?, owner = ?, control = ?, heartbeat_at = ?,
               updated_at = ?, pause_reason = '', error = '',
               started_at = COALESCE(started_at, ?)
         WHERE run_key = ?
           AND status <> ?
        """,
        (
            RunStatus.RUNNING.value,
            me,
            Control.NONE.value,
            stamp,
            stamp,
            stamp,
            run_key,
            RunStatus.RUNNING.value,
        ),
    )
    # A stale lease has to be broken separately, because the guard above
    # deliberately refuses to touch a row that is already marked running.
    if status is RunStatus.RUNNING:
        get_db().execute(
            """
            UPDATE runs
               SET owner = ?, control = ?, heartbeat_at = ?, updated_at = ?
             WHERE run_key = ? AND owner = ?
            """,
            (me, Control.NONE.value, stamp, stamp, run_key, run["owner"]),
        )

    fresh = get_run(run_key)
    return bool(fresh and fresh["owner"] == me and fresh["status"] == RunStatus.RUNNING.value)


def release_run(
    run_key: str,
    *,
    status: RunStatus,
    reason: str = "",
    error: str = "",
    spent: Spend | None = None,
    phase: str = "",
) -> None:
    fields: dict[str, Any] = {
        "status": status.value,
        "control": Control.NONE.value,
        "pause_reason": reason,
        "error": error,
        "owner": "",
    }
    if phase:
        fields["phase"] = phase
    if spent is not None:
        fields["spent"] = spent.to_dict()
    if status.terminal:
        fields["finished_at"] = now_iso()
    update_run(run_key, **fields)


# ------------------------------------------------------------------ events


def append_event(
    run_id: int,
    kind: EventKind | str,
    *,
    agent: str = "",
    node: str = "",
    level: str = "info",
    message: str = "",
    payload: dict[str, Any] | None = None,
) -> int:
    value = kind.value if isinstance(kind, EventKind) else str(kind)
    return get_db().insert(
        """
        INSERT INTO events (run_id, kind, agent, node, level, message, payload, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (run_id, value, agent, node, level, message, _json(payload or {}), now_iso()),
    )


def list_events(
    run_id: int, *, after_id: int = 0, limit: int = 500, kinds: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM events WHERE run_id = ? AND id > ?"
    params: list[Any] = [run_id, after_id]
    if kinds:
        sql += f" AND kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(limit)
    rows = get_db().query(sql, params)
    out = []
    for row in rows:
        item = dict(row)
        item["payload"] = loads(row["payload"], {}) or {}
        out.append(item)
    return out


# ------------------------------------------------------------------- steps


def record_step(
    run_id: int,
    *,
    seq: int,
    node: str,
    agent: str = "",
    action: str = "",
    status: str = "ok",
    attempt: int = 1,
    payload_in: Any = None,
    payload_out: Any = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    error: str = "",
) -> None:
    get_db().execute(
        """
        INSERT INTO steps (
            run_id, seq, node, agent, action, status, attempt,
            input, output, tokens_in, tokens_out, cost_usd, duration_ms,
            error, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, seq, node, agent, action, status, attempt,
            _json(payload_in), _json(payload_out), tokens_in, tokens_out,
            cost_usd, duration_ms, error, now_iso(),
        ),
    )


def list_steps(run_id: int, *, limit: int = 500) -> list[dict[str, Any]]:
    rows = get_db().query(
        "SELECT * FROM steps WHERE run_id = ? ORDER BY seq ASC LIMIT ?", (run_id, limit)
    )
    out = []
    for row in rows:
        item = dict(row)
        item["input"] = loads(row["input"], {})
        item["output"] = loads(row["output"], {})
        out.append(item)
    return out


# ------------------------------------------------------------- checkpoints


def save_checkpoint(
    run_id: int, *, seq: int, node: str, queue: list[Task], state: dict[str, Any]
) -> None:
    """Persist everything needed to carry on from here.

    The queue and the blackboard together are the run. Writing them after every
    completed step is what makes stopping free: there is never work in flight
    that a resume would have to reconstruct.
    """
    cursor = {"queue": [task.to_dict() for task in queue]}
    get_db().execute(
        """
        INSERT INTO checkpoints (run_id, seq, node, cursor, state, created_at)
        VALUES (?,?,?,?,?,?)
        """,
        (run_id, seq, node, _json(cursor), _json(state), now_iso()),
    )


def latest_checkpoint(run_id: int) -> dict[str, Any] | None:
    row = get_db().query_one(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY seq DESC LIMIT 1", (run_id,)
    )
    if row is None:
        return None
    cursor = loads(row["cursor"], {}) or {}
    return {
        "seq": int(row["seq"]),
        "node": row["node"],
        "queue": [Task.from_dict(item) for item in cursor.get("queue", [])],
        "state": loads(row["state"], {}) or {},
    }


def prune_checkpoints(run_id: int, *, keep: int = 5) -> None:
    """Keep the tail only. Older snapshots are superseded and just cost disk."""
    get_db().execute(
        """
        DELETE FROM checkpoints
         WHERE run_id = ?
           AND seq <= COALESCE((
                SELECT MIN(seq) FROM (
                    SELECT seq FROM checkpoints WHERE run_id = ?
                    ORDER BY seq DESC LIMIT ?
                ) AS recent
           ), 0) - 1
        """,
        (run_id, run_id, keep),
    )


# --------------------------------------------------------------- artifacts


def put_artifact(run_id: int, artifact: Artifact) -> int:
    """Store a new version of an artifact, superseding any earlier one."""
    db = get_db()
    current = db.scalar(
        "SELECT MAX(version) FROM artifacts WHERE run_id = ? AND kind = ? AND art_key = ?",
        (run_id, artifact.kind, artifact.key),
    )
    version = int(current or 0) + 1
    if current:
        db.execute(
            "UPDATE artifacts SET superseded = TRUE WHERE run_id = ? AND kind = ? AND art_key = ?",
            (run_id, artifact.kind, artifact.key),
        )
    artifact.version = version
    return db.insert(
        """
        INSERT INTO artifacts (
            run_id, kind, art_key, parent_key, title, body, content,
            produced_by, version, superseded, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,FALSE,?)
        """,
        (
            run_id, artifact.kind, artifact.key, artifact.parent_key,
            artifact.title, artifact.body, _json(artifact.content),
            artifact.produced_by, version, artifact.created_at,
        ),
    )


def _artifact_row(row: Row) -> Artifact:
    return Artifact(
        kind=row["kind"],
        key=row["art_key"],
        title=row["title"] or "",
        body=row["body"] or "",
        content=loads(row["content"], {}) or {},
        parent_key=row["parent_key"] or "",
        produced_by=row["produced_by"] or "",
        version=int(row["version"]),
        created_at=row["created_at"],
    )


def list_artifacts(
    run_id: int, *, kind: str = "", current_only: bool = True
) -> list[Artifact]:
    sql = "SELECT * FROM artifacts WHERE run_id = ?"
    params: list[Any] = [run_id]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    if current_only:
        sql += " AND superseded = FALSE"
    sql += " ORDER BY id ASC"
    return [_artifact_row(row) for row in get_db().query(sql, params)]


def get_artifact(run_id: int, kind: str, key: str) -> Artifact | None:
    row = get_db().query_one(
        """
        SELECT * FROM artifacts
         WHERE run_id = ? AND kind = ? AND art_key = ?
         ORDER BY version DESC LIMIT 1
        """,
        (run_id, kind, key),
    )
    return _artifact_row(row) if row else None


# ---------------------------------------------------------------- evidence


def add_evidence(run_id: int, item: dict[str, Any]) -> tuple[str, bool]:
    """Store one source. Returns its reference and whether it was new.

    Two researchers working in parallel routinely land on the same page. The
    unique constraint on url_key means the second one costs a rejected insert
    rather than a second fetch and a duplicate citation.
    """
    db = get_db()
    url = str(item.get("url") or "").strip()
    key = make_url_key(url)
    existing = db.query_one(
        "SELECT ref FROM evidence WHERE run_id = ? AND url_key = ?", (run_id, key)
    )
    if existing:
        return str(existing["ref"]), False

    count = int(db.scalar("SELECT COUNT(*) FROM evidence WHERE run_id = ?", (run_id,)) or 0)
    ref = f"S{count + 1}"
    db.execute(
        """
        INSERT INTO evidence (
            run_id, ref, url, url_key, domain, title, author, snippet, body,
            published_at, backend, found_by, query, relevance, credibility,
            simhash, embedding, fetched_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, ref, url, key,
            item.get("domain", ""), item.get("title", ""), item.get("author", ""),
            item.get("snippet", ""), item.get("body", ""), item.get("published_at"),
            item.get("backend", ""), item.get("found_by", ""), item.get("query", ""),
            float(item.get("relevance", 0) or 0), float(item.get("credibility", 0) or 0),
            str(item.get("simhash", "")), item.get("embedding"), now_iso(),
        ),
    )
    return ref, True


def list_evidence(run_id: int, *, refs: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    sql = "SELECT * FROM evidence WHERE run_id = ?"
    params: list[Any] = [run_id]
    if refs:
        sql += f" AND ref IN ({','.join('?' * len(refs))})"
        params.extend(refs)
    sql += " ORDER BY id ASC"
    return [dict(row) for row in get_db().query(sql, params)]


def evidence_count(run_id: int) -> int:
    return int(get_db().scalar("SELECT COUNT(*) FROM evidence WHERE run_id = ?", (run_id,)) or 0)


# ------------------------------------------------------------------- usage


def record_usage(
    run_id: int,
    *,
    step_seq: int = 0,
    agent: str = "",
    purpose: str = "",
    provider: str = "",
    model: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    priced: bool = False,
) -> None:
    get_db().execute(
        """
        INSERT INTO usage (
            run_id, step_seq, agent, purpose, provider, model,
            tokens_in, tokens_out, cost_usd, priced, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, step_seq, agent, purpose, provider, model,
            tokens_in, tokens_out, cost_usd, bool(priced), now_iso(),
        ),
    )


def usage_by_agent(run_id: int) -> list[dict[str, Any]]:
    rows = get_db().query(
        """
        SELECT agent,
               COUNT(*)          AS calls,
               SUM(tokens_in)    AS tokens_in,
               SUM(tokens_out)   AS tokens_out,
               SUM(cost_usd)     AS cost_usd
          FROM usage
         WHERE run_id = ?
         GROUP BY agent
         ORDER BY SUM(tokens_in) + SUM(tokens_out) DESC
        """,
        (run_id,),
    )
    return [dict(row) for row in rows]


# -------------------------------------------------------------- tool calls


def record_tool_call(
    run_id: int,
    *,
    step_seq: int = 0,
    agent: str = "",
    tool: str = "",
    arguments: Any = None,
    result: Any = None,
    ok: bool = True,
    error: str = "",
    duration_ms: int = 0,
) -> None:
    get_db().execute(
        """
        INSERT INTO tool_calls (
            run_id, step_seq, agent, tool, arguments, result, ok, error,
            duration_ms, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, step_seq, agent, tool, _json(arguments), _json(result),
            bool(ok), error, duration_ms, now_iso(),
        ),
    )


def tool_usage(run_id: int) -> list[dict[str, Any]]:
    rows = get_db().query(
        """
        SELECT tool, agent, COUNT(*) AS calls,
               SUM(CASE WHEN ok THEN 0 ELSE 1 END) AS failures
          FROM tool_calls WHERE run_id = ?
         GROUP BY tool, agent ORDER BY COUNT(*) DESC
        """,
        (run_id,),
    )
    return [dict(row) for row in rows]


# ---------------------------------------------------------------- messages


def post_message(run_id: int, message: Message, *, seq: int = 0) -> None:
    get_db().execute(
        """
        INSERT INTO messages (run_id, seq, sender, recipient, topic, in_reply_to, content, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            run_id, seq, message.sender, message.recipient, message.topic,
            message.in_reply_to, _json({"body": message.content, "id": message.msg_id}),
            message.created_at,
        ),
    )


def list_messages(run_id: int, *, limit: int = 300) -> list[dict[str, Any]]:
    rows = get_db().query(
        "SELECT * FROM messages WHERE run_id = ? ORDER BY id ASC LIMIT ?", (run_id, limit)
    )
    out = []
    for row in rows:
        item = dict(row)
        payload = loads(row["content"], {}) or {}
        item["content"] = payload.get("body")
        item["msg_id"] = payload.get("id", "")
        out.append(item)
    return out


# ---------------------------------------------------------------- memories


def remember(
    *,
    scope: str = "global",
    kind: str = "fact",
    key: str,
    content: str,
    meta: dict[str, Any] | None = None,
    weight: float = 1.0,
    embedding: bytes | None = None,
) -> None:
    """Write a durable lesson. Re-remembering the same key strengthens it."""
    db = get_db()
    stamp = now_iso()
    existing = db.query_one(
        "SELECT id, weight, hits FROM memories WHERE scope = ? AND kind = ? AND mem_key = ?",
        (scope, kind, key),
    )
    if existing:
        db.execute(
            """
            UPDATE memories
               SET content = ?, meta = ?, weight = ?, hits = ?, embedding = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                content, _json(meta or {}), float(existing["weight"]) + weight,
                int(existing["hits"]) + 1, embedding, stamp, existing["id"],
            ),
        )
        return
    db.execute(
        """
        INSERT INTO memories (scope, kind, mem_key, content, meta, weight, hits, embedding, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (scope, kind, key, content, _json(meta or {}), weight, 1, embedding, stamp, stamp),
    )


def recall(*, scope: str = "global", kind: str = "", limit: int = 20) -> list[dict[str, Any]]:
    sql = "SELECT * FROM memories WHERE scope = ?"
    params: list[Any] = [scope]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY weight DESC, updated_at DESC LIMIT ?"
    params.append(limit)
    rows = get_db().query(sql, params)
    out = []
    for row in rows:
        item = dict(row)
        item["meta"] = loads(row["meta"], {}) or {}
        out.append(item)
    return out


# --------------------------------------------------------------- approvals


def request_approval(
    run_id: int,
    *,
    node: str = "",
    kind: str = "gate",
    question: str = "",
    options: list[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> int:
    return get_db().insert(
        """
        INSERT INTO approvals (run_id, node, kind, question, options, payload, status, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            run_id, node, kind, question, _json(options or []),
            _json(payload or {}), "pending", now_iso(),
        ),
    )


def pending_approval(run_id: int) -> dict[str, Any] | None:
    row = get_db().query_one(
        "SELECT * FROM approvals WHERE run_id = ? AND status = 'pending' ORDER BY id DESC LIMIT 1",
        (run_id,),
    )
    if row is None:
        return None
    item = dict(row)
    item["options"] = loads(row["options"], []) or []
    item["payload"] = loads(row["payload"], {}) or {}
    return item


def answer_approval(approval_id: int, *, answer: str, note: str = "") -> None:
    get_db().execute(
        "UPDATE approvals SET status = 'answered', answer = ?, note = ?, answered_at = ? WHERE id = ?",
        (answer, note, now_iso(), approval_id),
    )


def latest_answer(run_id: int, node: str) -> dict[str, Any] | None:
    row = get_db().query_one(
        """
        SELECT * FROM approvals
         WHERE run_id = ? AND node = ? AND status = 'answered'
         ORDER BY id DESC LIMIT 1
        """,
        (run_id, node),
    )
    if row is None:
        return None
    item = dict(row)
    item["options"] = loads(row["options"], []) or []
    item["payload"] = loads(row["payload"], {}) or {}
    return item


def delete_run(run_key: str) -> bool:
    """Remove a run and everything hanging off it.

    SQLite does not enforce ON DELETE CASCADE unless foreign keys are switched
    on for the connection, so the children are removed explicitly rather than
    trusted to the declaration.
    """
    db = get_db()
    row = db.query_one("SELECT id FROM runs WHERE run_key = ?", (run_key,))
    if row is None:
        return False
    run_id = int(row["id"])
    for table in (
        "events", "steps", "messages", "artifacts", "checkpoints",
        "evidence", "usage", "tool_calls", "approvals",
    ):
        db.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
    db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    return True
