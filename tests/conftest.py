"""Test fixtures.

Every test runs against a real SQLite database in a temporary directory rather
than a mock. The storage layer is not incidental to this system, it is where
pause, resume and the audit trail actually live, so a test that mocked it would
be testing the wrong thing.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Set before any project import, because settings reads the environment once at
# import time and the database path is resolved from it.
_TMP = Path(tempfile.mkdtemp(prefix="mas-tests-"))
os.environ["DATA_DIR"] = str(_TMP)
os.environ["SQLITE_PATH"] = str(_TMP / "test.db")
os.environ["IGNORE_DOTENV"] = "1"
os.environ["DATABASE_URL"] = ""


# Point this at a Postgres instance to run the whole suite against it instead
# of SQLite. The schema is written once in a portable dialect and claimed to
# work on both, and a claim nobody can run is not worth much:
#
#   docker run -d --name pg -p 55432:5432 -e POSTGRES_USER=agentic
#       -e POSTGRES_PASSWORD=agentic -e POSTGRES_DB=agentic postgres:16-alpine
#   MAS_TEST_POSTGRES=postgresql://agentic:agentic@127.0.0.1:55432/agentic pytest
POSTGRES_URL = os.environ.get("MAS_TEST_POSTGRES", "").strip()


@pytest.fixture(autouse=True)
def fresh_database(tmp_path, monkeypatch):
    """One empty database per test, so nothing leaks between them."""
    from mas.core import settings
    from mas.core.db import engine

    if POSTGRES_URL:
        monkeypatch.setattr(settings, "DATABASE_URL", POSTGRES_URL)
        engine.reset_db()
        # Postgres is one shared database rather than a file per test, so it is
        # emptied between tests instead of recreated.
        db = engine.get_db()
        db.ensure_schema()
        with db.connect() as conn:
            conn.execute(
                "TRUNCATE runs, steps, events, messages, artifacts, checkpoints, "
                "evidence, usage, tool_calls, memories, approvals RESTART IDENTITY CASCADE"
            )
        yield POSTGRES_URL
        engine.reset_db()
        return

    path = tmp_path / "run.db"
    monkeypatch.setattr(settings, "SQLITE_PATH", path)
    monkeypatch.setattr(settings, "DATABASE_URL", None)
    engine.reset_db()
    yield path
    engine.reset_db()


@pytest.fixture
def stub_provider(monkeypatch):
    """Replace the model with a scripted one.

    Agent behaviour is tested against fixed replies rather than a live model:
    the point of those tests is that the kernel does the right thing with a
    given reply, and a real model would make them slow, costly and flaky
    without testing anything extra.
    """
    from mas.core.llm.base import Completion, Usage

    class Script:
        def __init__(self) -> None:
            self.replies: list[str] = []
            self.calls: list[dict] = []

        def push(self, *replies: str) -> None:
            self.replies.extend(replies)

        def __call__(self, prompt, *, model, system=None, temperature=0.2,
                     max_tokens=4096, json_mode=False):
            self.calls.append(
                {"prompt": prompt, "system": system, "model": model, "json": json_mode}
            )
            text = self.replies.pop(0) if self.replies else '{"final_answer": {}}'
            return Completion(
                text=text,
                usage=Usage(provider="stub", model=model, input_tokens=10, output_tokens=5),
            )

    script = Script()

    from mas.core.llm import registry as llm

    class StubProvider:
        id = "stub"
        label = "Stub"
        key_names = ("STUB",)
        models = ()
        local = False

        def is_available(self) -> bool:
            return True

        def default_model(self) -> str:
            return "stub-large"

        def cheap_model(self) -> str:
            return "stub-small"

        def generate(self, prompt, **kwargs):
            return script(prompt, **kwargs)

    provider = StubProvider()
    monkeypatch.setattr(llm, "provider_map", lambda: {"stub": provider})
    monkeypatch.setattr(llm, "available_providers", lambda: [provider])
    return script
