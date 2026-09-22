"""The full schema, written once in the portable dialect from `engine`.

`create_all` is idempotent, so it runs on every cold start and costs nothing
after the first.

Timestamps are ISO 8601 UTC text throughout. See `engine` for why.

Boolean defaults are written as TRUE and FALSE rather than 1 and 0. SQLite maps
{BOOL} to INTEGER and would take either, but Postgres maps it to BOOLEAN and
rejects an integer default outright with a DatatypeMismatch.

The shape of this schema is what makes a run resumable. Every table below is
append only except `runs`, so stopping the process at any instant leaves a
consistent record, and `checkpoints` holds enough to rebuild the blackboard and
carry on from the last completed step.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .engine import render_ddl

if TYPE_CHECKING:  # pragma: no cover
    from .engine import Database


TABLES: tuple[str, ...] = (
    # One row per agent run. This is the only mutable table: `status`,
    # `control` and the spend counters change as the run proceeds. `control` is
    # how another process asks a running orchestrator to stop, which is why it
    # lives in the database rather than in a thread event.
    """
    CREATE TABLE IF NOT EXISTS runs (
        id                 {PK},
        run_key            {TEXT} NOT NULL UNIQUE,
        brief              {TEXT} NOT NULL,
        title              {TEXT} DEFAULT '',
        workflow           {TEXT} NOT NULL DEFAULT 'research_report',
        status             {TEXT} NOT NULL DEFAULT 'pending',
        control            {TEXT} NOT NULL DEFAULT 'none',
        phase              {TEXT} DEFAULT '',
        provider           {TEXT} DEFAULT '',
        model              {TEXT} DEFAULT '',
        cheap_model        {TEXT} DEFAULT '',
        config             {JSON},
        budget             {JSON},
        spent              {JSON},
        next_seq           {INT} DEFAULT 0,
        error              {TEXT} DEFAULT '',
        pause_reason       {TEXT} DEFAULT '',
        owner              {TEXT} DEFAULT '',
        heartbeat_at       {TS},
        created_at         {TS} NOT NULL,
        started_at         {TS},
        updated_at         {TS} NOT NULL,
        finished_at        {TS}
    )
    """,
    # Every unit of work the orchestrator scheduled, in order. A step is
    # recorded only once it has finished, so the highest seq present is exactly
    # the point a resume should start after.
    """
    CREATE TABLE IF NOT EXISTS steps (
        id            {PK},
        run_id        {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        seq           {INT} NOT NULL,
        node          {TEXT} NOT NULL DEFAULT '',
        agent         {TEXT} NOT NULL DEFAULT '',
        action        {TEXT} DEFAULT '',
        status        {TEXT} NOT NULL DEFAULT 'ok',
        attempt       {INT} DEFAULT 1,
        input         {JSON},
        output        {JSON},
        tokens_in     {INT} DEFAULT 0,
        tokens_out    {INT} DEFAULT 0,
        cost_usd      {REAL} DEFAULT 0,
        duration_ms   {INT} DEFAULT 0,
        error         {TEXT} DEFAULT '',
        created_at    {TS} NOT NULL,
        UNIQUE (run_id, seq)
    )
    """,
    # The live trace. Append only and never read back by the engine, so it can
    # be pruned freely. The UI streams from here by id cursor, which is what
    # lets a browser reconnect mid run and miss nothing.
    """
    CREATE TABLE IF NOT EXISTS events (
        id            {PK},
        run_id        {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        kind          {TEXT} NOT NULL,
        agent         {TEXT} DEFAULT '',
        node          {TEXT} DEFAULT '',
        level         {TEXT} DEFAULT 'info',
        message       {TEXT} DEFAULT '',
        payload       {JSON},
        created_at    {TS} NOT NULL
    )
    """,
    # Agent to agent traffic. Kept separate from `events` because this is the
    # actual collaboration record: who told whom what, and on whose request.
    """
    CREATE TABLE IF NOT EXISTS messages (
        id            {PK},
        run_id        {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        seq           {INT} NOT NULL DEFAULT 0,
        sender        {TEXT} NOT NULL,
        recipient     {TEXT} NOT NULL DEFAULT '*',
        topic         {TEXT} DEFAULT '',
        in_reply_to   {TEXT} DEFAULT '',
        content       {JSON},
        created_at    {TS} NOT NULL
    )
    """,
    # Typed outputs. `key` is stable within a run so a revision supersedes its
    # predecessor by bumping `version` rather than overwriting it, which keeps
    # the whole revision history of a draft inspectable after the fact.
    """
    CREATE TABLE IF NOT EXISTS artifacts (
        id            {PK},
        run_id        {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        kind          {TEXT} NOT NULL,
        art_key       {TEXT} NOT NULL,
        parent_key    {TEXT} DEFAULT '',
        title         {TEXT} DEFAULT '',
        body          {TEXT} DEFAULT '',
        content       {JSON},
        produced_by   {TEXT} DEFAULT '',
        version       {INT} NOT NULL DEFAULT 1,
        superseded    {BOOL} DEFAULT FALSE,
        created_at    {TS} NOT NULL,
        UNIQUE (run_id, kind, art_key, version)
    )
    """,
    # The blackboard, snapshotted after each step. Resuming reads the newest
    # row and replays nothing: the state is the state.
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        id            {PK},
        run_id        {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        seq           {INT} NOT NULL,
        node          {TEXT} DEFAULT '',
        cursor        {JSON},
        state         {JSON},
        created_at    {TS} NOT NULL,
        UNIQUE (run_id, seq)
    )
    """,
    # Source material the researchers gathered. url_key is a normalised hash of
    # the canonical URL, so the unique constraint stops two agents paying twice
    # for the same page without either having to ask the other.
    """
    CREATE TABLE IF NOT EXISTS evidence (
        id             {PK},
        run_id         {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        ref            {TEXT} NOT NULL DEFAULT '',
        url            {TEXT} NOT NULL,
        url_key        {TEXT} NOT NULL,
        domain         {TEXT} DEFAULT '',
        title          {TEXT} DEFAULT '',
        author         {TEXT} DEFAULT '',
        snippet        {TEXT} DEFAULT '',
        body           {TEXT} DEFAULT '',
        published_at   {TS},
        backend        {TEXT} DEFAULT '',
        found_by       {TEXT} DEFAULT '',
        query          {TEXT} DEFAULT '',
        relevance      {REAL} DEFAULT 0,
        credibility    {REAL} DEFAULT 0,
        simhash        {TEXT} DEFAULT '',
        embedding      {BLOB},
        fetched_at     {TS} NOT NULL,
        UNIQUE (run_id, url_key)
    )
    """,
    # Per call token accounting. The run's `spent` blob is the running total,
    # this is the itemised bill behind it.
    """
    CREATE TABLE IF NOT EXISTS usage (
        id            {PK},
        run_id        {BIGINT} REFERENCES runs(id) ON DELETE CASCADE,
        step_seq      {INT} DEFAULT 0,
        agent         {TEXT} DEFAULT '',
        purpose       {TEXT} DEFAULT '',
        provider      {TEXT} DEFAULT '',
        model         {TEXT} DEFAULT '',
        tokens_in     {INT} DEFAULT 0,
        tokens_out    {INT} DEFAULT 0,
        cost_usd      {REAL} DEFAULT 0,
        priced        {BOOL} DEFAULT FALSE,
        created_at    {TS} NOT NULL
    )
    """,
    # Every tool invocation, with its arguments and result. This is the record
    # that distinguishes an agent choosing a tool from a pipeline calling one.
    """
    CREATE TABLE IF NOT EXISTS tool_calls (
        id            {PK},
        run_id        {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        step_seq      {INT} DEFAULT 0,
        agent         {TEXT} DEFAULT '',
        tool          {TEXT} NOT NULL,
        arguments     {JSON},
        result        {JSON},
        ok            {BOOL} DEFAULT TRUE,
        error         {TEXT} DEFAULT '',
        duration_ms   {INT} DEFAULT 0,
        created_at    {TS} NOT NULL
    )
    """,
    # Cross run memory. Scoped to 'global' it survives the run that wrote it,
    # which is what lets the system carry a lesson (this domain paywalls, that
    # backend returns nothing for finance queries) into the next brief.
    """
    CREATE TABLE IF NOT EXISTS memories (
        id            {PK},
        scope         {TEXT} NOT NULL DEFAULT 'global',
        kind          {TEXT} NOT NULL DEFAULT 'fact',
        mem_key       {TEXT} NOT NULL,
        content       {TEXT} NOT NULL,
        meta          {JSON},
        weight        {REAL} DEFAULT 1.0,
        hits          {INT} DEFAULT 0,
        embedding     {BLOB},
        created_at    {TS} NOT NULL,
        updated_at    {TS} NOT NULL,
        UNIQUE (scope, kind, mem_key)
    )
    """,
    # Human in the loop. A gate blocks the run until someone answers, and the
    # answer is durable, so the approval survives a restart like everything else.
    """
    CREATE TABLE IF NOT EXISTS approvals (
        id            {PK},
        run_id        {BIGINT} NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
        node          {TEXT} DEFAULT '',
        kind          {TEXT} NOT NULL DEFAULT 'gate',
        question      {TEXT} DEFAULT '',
        options       {JSON},
        payload       {JSON},
        status        {TEXT} NOT NULL DEFAULT 'pending',
        answer        {TEXT} DEFAULT '',
        note          {TEXT} DEFAULT '',
        created_at    {TS} NOT NULL,
        answered_at   {TS}
    )
    """,
)


INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status)",
    "CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_steps_run ON steps (run_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_events_run ON events (run_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_events_kind ON events (run_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_messages_run ON messages (run_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts (run_id, kind)",
    "CREATE INDEX IF NOT EXISTS idx_artifacts_key ON artifacts (run_id, art_key)",
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints (run_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_evidence_domain ON evidence (run_id, domain)",
    "CREATE INDEX IF NOT EXISTS idx_usage_run ON usage (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_tool_calls_run ON tool_calls (run_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories (scope, kind)",
    "CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals (run_id, status)",
)


def create_all(db: Database) -> None:
    with db.connect() as conn:
        for statement in TABLES:
            conn.execute(render_ddl(statement.strip(), db.dialect))
        for statement in INDEXES:
            conn.execute(statement)
