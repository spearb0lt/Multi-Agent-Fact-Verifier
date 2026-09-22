"""The shared workspace every agent reads from and writes to.

This is the piece that makes the system a team rather than a chain of prompts.
In a chain, each stage sees only what the previous stage handed it, so the
Writer cannot know that the Fact Checker threw out a claim, and the Critic
cannot tell whether a gap is the Writer's fault or the Researcher's. Here every
role sees the same board, and each writes into its own section.

Two rules hold the design together:

* The whole board is JSON. A run has to survive the process that started it, so
  anything that cannot be serialised cannot live here. Large text (the bodies
  of fetched pages) deliberately does not: it goes to the `evidence` table and
  the board keeps the reference.
* Sections are owned. Any agent may read anything, but a section is written by
  one role, so a confused agent corrupts its own work and not everyone's.
"""
from __future__ import annotations

import threading
from typing import Any

from ..core.util import now_iso


class Blackboard:
    """A JSON shaped, thread safe, section owned shared state.

    The lock is not theatre: researchers run in parallel in a thread pool and
    all of them append findings. Without it, two appends racing on the same
    list lose one of them, which shows up much later as a citation pointing at
    evidence nobody gathered.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._lock = threading.RLock()
        self._data: dict[str, Any] = dict(data or {})
        self._data.setdefault("meta", {})
        self._data.setdefault("sections", {})

    # ------------------------------------------------------------- raw access

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            import copy

            return copy.deepcopy(self._data)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Blackboard:
        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._touch(key)

    def update(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                self._data[key] = value
                self._touch(key)

    def setdefault(self, key: str, default: Any) -> Any:
        with self._lock:
            if key not in self._data:
                self._data[key] = default
                self._touch(key)
            return self._data[key]

    def _touch(self, key: str) -> None:
        self._data.setdefault("sections", {})[key] = now_iso()

    # ------------------------------------------------------------ collections

    def append(self, key: str, item: Any) -> int:
        """Append to a list section and return its new length.

        Returning the length rather than nothing lets a caller log "finding 7
        of an expected 10" without a second read that another thread could
        invalidate between the two calls.
        """
        with self._lock:
            bucket = self._data.setdefault(key, [])
            if not isinstance(bucket, list):
                raise TypeError(f"Blackboard section '{key}' is not a list.")
            bucket.append(item)
            self._touch(key)
            return len(bucket)

    def extend(self, key: str, items: list[Any]) -> int:
        with self._lock:
            bucket = self._data.setdefault(key, [])
            if not isinstance(bucket, list):
                raise TypeError(f"Blackboard section '{key}' is not a list.")
            bucket.extend(items)
            self._touch(key)
            return len(bucket)

    def items(self, key: str) -> list[Any]:
        with self._lock:
            value = self._data.get(key, [])
            return list(value) if isinstance(value, list) else []

    def put(self, key: str, field: str, value: Any) -> None:
        """Set one field inside a dict section."""
        with self._lock:
            bucket = self._data.setdefault(key, {})
            if not isinstance(bucket, dict):
                raise TypeError(f"Blackboard section '{key}' is not a mapping.")
            bucket[field] = value
            self._touch(key)

    def mapping(self, key: str) -> dict[str, Any]:
        with self._lock:
            value = self._data.get(key, {})
            return dict(value) if isinstance(value, dict) else {}

    def counter(self, key: str, by: int = 1) -> int:
        with self._lock:
            value = int(self._data.get(key, 0)) + by
            self._data[key] = value
            return value

    # ---------------------------------------------------- research shorthand
    # Named accessors for the flagship workflow's sections. A workflow is free
    # to ignore these and use the generic API, but naming the common ones keeps
    # section names from being retyped as string literals in a dozen agents.

    @property
    def brief(self) -> str:
        return str(self.get("brief", ""))

    @property
    def plan(self) -> dict[str, Any]:
        return self.mapping("plan")

    @property
    def subquestions(self) -> list[dict[str, Any]]:
        return self.items("subquestions")

    @property
    def findings(self) -> list[dict[str, Any]]:
        return self.items("findings")

    @property
    def claims(self) -> list[dict[str, Any]]:
        return self.items("claims")

    @property
    def verdicts(self) -> list[dict[str, Any]]:
        return self.items("verdicts")

    @property
    def draft(self) -> dict[str, Any]:
        return self.mapping("draft")

    @property
    def critiques(self) -> list[dict[str, Any]]:
        return self.items("critiques")

    @property
    def report(self) -> dict[str, Any]:
        return self.mapping("report")

    @property
    def conflicts(self) -> list[dict[str, Any]]:
        """Claim pairs the Reconciler ruled are in conflict."""
        return self.items("conflicts")

    @property
    def revision(self) -> int:
        return int(self.get("revision", 0))

    def verdict_for(self, claim_id: str) -> dict[str, Any] | None:
        for verdict in self.verdicts:
            if verdict.get("claim_id") == claim_id:
                return verdict
        return None

    def supported_claims(self) -> list[dict[str, Any]]:
        """Claims the Fact Checker let through, in the order they were made.

        The Writer is given only these. That is the mechanism behind the whole
        verification story: an unsupported claim is not flagged in the draft,
        it never reaches the draft.
        """
        out = []
        for claim in self.claims:
            verdict = self.verdict_for(str(claim.get("id", "")))
            if verdict and verdict.get("verdict") in {"supported", "partly_supported"}:
                merged = dict(claim)
                merged["verdict"] = verdict
                out.append(merged)
        return out

    def rejected_claims(self) -> list[dict[str, Any]]:
        out = []
        for claim in self.claims:
            verdict = self.verdict_for(str(claim.get("id", "")))
            if verdict and verdict.get("verdict") in {"unsupported", "contradicted"}:
                merged = dict(claim)
                merged["verdict"] = verdict
                out.append(merged)
        return out

    # ---------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        """The compact shape the UI and the trace show, without the bulk."""
        return {
            "brief": self.brief,
            "revision": self.revision,
            "counts": {
                "subquestions": len(self.subquestions),
                "findings": len(self.findings),
                "claims": len(self.claims),
                "verdicts": len(self.verdicts),
                "supported": len(self.supported_claims()),
                "rejected": len(self.rejected_claims()),
                "critiques": len(self.critiques),
                "conflicts": len(self.conflicts),
            },
            "has_draft": bool(self.draft.get("markdown")),
            "has_report": bool(self.report.get("markdown")),
        }
