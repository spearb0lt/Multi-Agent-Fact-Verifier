"""One database API over SQLite and Postgres.

The project has to run from a laptop with no services installed and from a
serverless function whose only storage is a remote Postgres. Rather than write
two data layers, queries are written once in a small common dialect and this
module adapts them.

Two conventions keep the adaptation trivial:

* Placeholders are always `?`. Postgres wants `%s`, so they are rewritten on
  the way out, taking care not to touch a `?` inside a string literal.
* Timestamps are always ISO 8601 UTC strings, never native date types. They
  sort correctly as text, they survive both drivers unchanged, and they remove
  an entire category of timezone bug at the driver boundary.
"""
from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .. import settings


class Dialect:
    SQLITE = "sqlite"
    POSTGRES = "postgres"


# A `?` that is not inside a single quoted string literal. The alternation puts
# the literal first so the regex consumes it whole and only the trailing branch
# can match a real placeholder.
_PLACEHOLDER_RE = re.compile(r"'(?:[^']|'')*'|(\?)")


def to_pg_params(sql: str) -> str:
    """Rewrite `?` placeholders to `%s`, leaving string literals alone.

    Query text in this project must not contain a literal `%`. Postgres reads
    one as the start of a placeholder and would need it doubled, which SQLite
    would then take literally. LIKE wildcards therefore belong in the bound
    parameter (`LIKE ?` with `f"%{term}%"`), never in the SQL, which keeps both
    dialects reading the same string.
    """

    def repl(match: re.Match[str]) -> str:
        return "%s" if match.group(1) else match.group(0)

    return _PLACEHOLDER_RE.sub(repl, sql)


# Portable type names. The schema is written with these and each dialect
# substitutes its own spelling.
_TYPES = {
    Dialect.SQLITE: {
        "PK": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "JSON": "TEXT",
        "BLOB": "BLOB",
        "BOOL": "INTEGER",
        "TS": "TEXT",
        "TEXT": "TEXT",
        "INT": "INTEGER",
        "REAL": "REAL",
        "BIGINT": "INTEGER",
    },
    Dialect.POSTGRES: {
        "PK": "BIGSERIAL PRIMARY KEY",
        "JSON": "JSONB",
        "BLOB": "BYTEA",
        "BOOL": "BOOLEAN",
        "TS": "TEXT",
        "TEXT": "TEXT",
        "INT": "INTEGER",
        "REAL": "DOUBLE PRECISION",
        "BIGINT": "BIGINT",
    },
}

_TYPE_TOKEN_RE = re.compile(r"\{(\w+)\}")


def render_ddl(sql: str, dialect: str) -> str:
    types = _TYPES[dialect]

    def repl(match: re.Match[str]) -> str:
        token = match.group(1)
        if token not in types:
            raise KeyError(f"Unknown portable type '{token}' in DDL.")
        return types[token]

    return _TYPE_TOKEN_RE.sub(repl, sql)


class Row(dict):
    """A result row usable as a mapping or by attribute.

    Attribute access reads much better in template and summary code, where a
    row is threaded through several layers, and the mapping behaviour keeps it
    JSON serialisable without conversion.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class Database:
    def __init__(self, url: str | None = None) -> None:
        self.url = (url or settings.DATABASE_URL or "").strip()
        self.dialect = Dialect.POSTGRES if self._is_postgres(self.url) else Dialect.SQLITE
        self._local = threading.local()
        self._initialised = False
        self._init_lock = threading.Lock()

    @staticmethod
    def _is_postgres(url: str) -> bool:
        return url.startswith(("postgres://", "postgresql://"))

    # ------------------------------------------------------------ connections

    def _normalised_pg_url(self) -> str:
        """The connection string, with a sensible TLS mode when none is given.

        Neon and Supabase require TLS and do not always say so in the URL they
        hand out, so a managed host gets sslmode=require. Forcing it on every
        host was wrong: a Postgres in Docker or on a VM usually has no
        certificate, and `require` fails outright with "server does not support
        SSL" rather than falling back.

        A hostname with no dot in it cannot be a public host. It is a container
        or service name, like the `db` in this project's own compose file, or
        localhost. Those get libpq's `prefer`, which uses TLS when the server
        offers it and connects anyway when it does not.
        """
        url = self.url
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        if "sslmode=" in url:
            return url

        from urllib.parse import urlsplit

        try:
            host = (urlsplit(url).hostname or "").lower()
        except ValueError:
            host = ""
        local = (
            not host
            or host in {"localhost", "127.0.0.1", "::1", "[::1]"}
            or "." not in host
        )
        mode = "prefer" if local else "require"
        return url + ("&" if "?" in url else "?") + f"sslmode={mode}"

    @contextmanager
    def connect(self) -> Iterator[Any]:
        """A connection scoped to one unit of work.

        Serverless invocations are short and may be frozen between requests, so
        no pool is kept here. Neon and Supabase both front the database with
        their own pooler, which is the right place for that concern.
        """
        if self.dialect == Dialect.POSTGRES:
            import psycopg
            from psycopg.rows import dict_row

            conn = psycopg.connect(self._normalised_pg_url(), row_factory=dict_row)
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            path = Path(settings.SQLITE_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            # WAL lets the API read while a pipeline run writes, which is the
            # normal state of this app rather than an edge case.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            try:
                yield conn
            finally:
                conn.close()

    # --------------------------------------------------------------- queries

    def _prepare(self, sql: str) -> str:
        return to_pg_params(sql) if self.dialect == Dialect.POSTGRES else sql

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self.ensure_schema()
        with self.connect() as conn:
            conn.execute(self._prepare(sql), tuple(params))

    def execute_many(self, sql: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return
        self.ensure_schema()
        prepared = self._prepare(sql)
        with self.connect() as conn:
            if self.dialect == Dialect.POSTGRES:
                with conn.cursor() as cur:
                    cur.executemany(prepared, [tuple(r) for r in rows])
            else:
                conn.executemany(prepared, [tuple(r) for r in rows])

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        self.ensure_schema()
        with self.connect() as conn:
            if self.dialect == Dialect.POSTGRES:
                with conn.cursor() as cur:
                    cur.execute(self._prepare(sql), tuple(params))
                    return [Row(r) for r in cur.fetchall()]
            cur = conn.execute(sql, tuple(params))
            return [Row(dict(r)) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.query_one(sql, params)
        if row is None:
            return None
        return next(iter(row.values()), None)

    def insert(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run an INSERT and return the new row's id.

        Postgres needs an explicit RETURNING clause for this and SQLite needs
        `lastrowid`, so the clause is appended when the caller has not written
        one and the two paths converge on an integer.
        """
        self.ensure_schema()
        with self.connect() as conn:
            if self.dialect == Dialect.POSTGRES:
                statement = sql if "returning" in sql.lower() else f"{sql} RETURNING id"
                with conn.cursor() as cur:
                    cur.execute(self._prepare(statement), tuple(params))
                    row = cur.fetchone()
                    return int(row["id"]) if row else 0
            cur = conn.execute(sql, tuple(params))
            return int(cur.lastrowid or 0)

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """A connection the caller drives, for multi statement atomic work."""
        self.ensure_schema()
        with self.connect() as conn:
            if self.dialect == Dialect.SQLITE:
                conn.execute("BEGIN")
                try:
                    yield _Tx(self, conn)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
            else:
                yield _Tx(self, conn)

    # ---------------------------------------------------------------- schema

    def ensure_schema(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:
                return
            # Set before running so the DDL's own use of execute() does not
            # recurse back into this method.
            self._initialised = True
            try:
                from .schema import create_all

                create_all(self)
            except Exception:
                self._initialised = False
                raise

    def reset_schema_flag(self) -> None:
        self._initialised = False


class _Tx:
    """The subset of the Database API that runs on one open connection."""

    def __init__(self, db: Database, conn: Any) -> None:
        self._db = db
        self._conn = conn

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        self._conn.execute(self._db._prepare(sql), tuple(params))

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[Row]:
        if self._db.dialect == Dialect.POSTGRES:
            with self._conn.cursor() as cur:
                cur.execute(self._db._prepare(sql), tuple(params))
                return [Row(r) for r in cur.fetchall()]
        cur = self._conn.execute(sql, tuple(params))
        return [Row(dict(r)) for r in cur.fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def insert(self, sql: str, params: Sequence[Any] = ()) -> int:
        if self._db.dialect == Dialect.POSTGRES:
            statement = sql if "returning" in sql.lower() else f"{sql} RETURNING id"
            with self._conn.cursor() as cur:
                cur.execute(self._db._prepare(statement), tuple(params))
                row = cur.fetchone()
                return int(row["id"]) if row else 0
        cur = self._conn.execute(sql, tuple(params))
        return int(cur.lastrowid or 0)


_db: Database | None = None
_db_lock = threading.Lock()


def get_db() -> Database:
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                _db = Database()
    return _db


def reset_db() -> None:
    global _db
    with _db_lock:
        _db = None
