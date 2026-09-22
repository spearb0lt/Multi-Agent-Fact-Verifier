"""Provider neutral contract for turning text into vectors.

Deduplication is the part of this app that decides whether the digest is
useful, and it rests entirely on these vectors. The contract is deliberately
narrow: give me strings, get back unit length rows in a consistent space.

Every implementation returns L2 normalised vectors, so cosine similarity is a
plain dot product everywhere downstream. That removes a whole class of bug
where one provider's raw magnitudes quietly skew a threshold tuned against
another's.
"""
from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..ops import meter
from ..util import truncate

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class EmbeddingError(RuntimeError):
    def __init__(self, message: str, *, provider: str = "", hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "provider": self.provider, "hint": self.hint}


@dataclass
class EmbedResult:
    vectors: np.ndarray  # shape (n, dim), float32, each row unit length
    provider: str
    model: str
    dim: int


@dataclass
class EmbedderStatus:
    id: str
    label: str
    available: bool
    reason: str = ""
    model: str = ""
    dim: int = 0
    docs_url: str = ""
    key_names: Sequence[str] = field(default_factory=tuple)
    local: bool = False
    # Quality is a blunt ordering used only to pick a sensible default when the
    # operator has expressed no preference.
    quality: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "available": self.available,
            "reason": self.reason,
            "model": self.model,
            "dim": self.dim,
            "docs_url": self.docs_url,
            "key_names": list(self.key_names),
            "local": self.local,
            "quality": self.quality,
        }


# How much of an article is worth embedding. The headline and opening carry
# almost all of the signal for "is this the same story", and most providers cap
# input anyway, so every backend truncates to the same budget. Keeping it
# identical across providers is what makes a threshold portable between them.
EMBED_CHAR_BUDGET = 1600


def build_text(title: str, summary: str = "", content: str = "") -> str:
    """The canonical string to embed for one article.

    The title is repeated once. Two reports of the same event share a headline
    far more reliably than they share body text, so giving it extra weight
    measurably improves duplicate detection on wire copy that each outlet has
    rewritten around the edges.
    """
    title = (title or "").strip()
    body = (summary or "").strip() or (content or "").strip()
    parts = [title, title, body] if title else [body]
    return truncate(" ".join(p for p in parts if p), EMBED_CHAR_BUDGET, suffix="")


def to_blob(vector: Sequence[float]) -> bytes:
    """Pack a vector for the database, as little endian float32."""
    return struct.pack(f"<{len(vector)}f", *(float(v) for v in vector))


def from_blob(blob: bytes | memoryview | None) -> list[float]:
    if not blob:
        return []
    raw = bytes(blob)
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}f", raw[: count * 4]))


class BaseEmbedder:
    """One way of producing vectors.

    Subclasses implement `_embed_batch` and declare their dimension. Batching,
    normalisation, empty input handling and cost metering are done once here so
    every backend behaves identically.
    """

    id: str = ""
    label: str = ""
    docs_url: str = ""
    key_names: tuple[str, ...] = ()
    local: bool = False
    quality: int = 0
    max_batch: int = 64
    dim: int = 0

    # Similarity thresholds are a property of the vector space, not of the
    # task, so each backend declares its own and the settings values override
    # them only when the operator has explicitly tuned one.
    #
    # The two cuts mean different things and are deliberately far apart.
    # `duplicate_threshold` is near identical text, a syndicated wire story
    # reprinted by five outlets, and the matched article is hidden from the
    # digest. `cluster_threshold` is the same real world event reported
    # independently, and those are grouped but all kept.
    #
    # The gap matters because of a measured property of sentence encoders on
    # headlines: two outlets describing one event score about 0.76, while two
    # opposite events on the same subject, a rate hold against a rate cut,
    # still score about 0.69. Seven hundredths is not a safe margin, so the
    # clustering cut alone must never be trusted to separate them. That is what
    # the contradiction guard and the optional LLM adjudication in the
    # clustering stage are for.
    duplicate_threshold: float = 0.90
    cluster_threshold: float = 0.72

    def is_available(self) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def unavailable_reason(self) -> str:
        return (
            f"Paste a {self.label} API key in Settings, or set "
            f"{' or '.join(self.key_names)} on the server."
        )

    def model_id(self) -> str:
        return ""

    def status(self) -> EmbedderStatus:
        available = self.is_available()
        return EmbedderStatus(
            id=self.id,
            label=self.label,
            available=available,
            reason="" if available else self.unavailable_reason(),
            model=self.model_id() if available else "",
            dim=self.dim,
            docs_url=self.docs_url,
            key_names=self.key_names,
            local=self.local,
            quality=self.quality,
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def embed(self, texts: Sequence[str], *, operation: str = "") -> EmbedResult:
        import numpy as np

        cleaned = [(t or "").strip() for t in texts]
        if not cleaned:
            return EmbedResult(
                vectors=np.zeros((0, self.dim or 1), dtype=np.float32),
                provider=self.id,
                model=self.model_id(),
                dim=self.dim,
            )

        rows: list[list[float]] = []
        for start in range(0, len(cleaned), self.max_batch):
            chunk = cleaned[start : start + self.max_batch]
            # An empty string embeds to nothing useful and some providers reject
            # it outright, so a placeholder goes in and the resulting row is
            # zeroed below rather than the whole batch failing.
            safe = [text if text else "." for text in chunk]
            rows.extend(self._embed_batch(safe))

        if len(rows) != len(cleaned):
            raise EmbeddingError(
                f"{self.label} returned {len(rows)} vectors for {len(cleaned)} inputs.",
                provider=self.id,
                hint="This usually means the provider silently dropped part of a batch.",
            )

        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.ndim != 2:
            raise EmbeddingError(
                f"{self.label} returned vectors of an unexpected shape.", provider=self.id
            )

        for index, text in enumerate(cleaned):
            if not text:
                matrix[index] = 0.0

        # Unit length rows, so cosine similarity is a dot product downstream.
        # A zero row would divide by zero, so its norm is clamped to one and it
        # stays zero, which correctly scores zero similarity against everything.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        matrix = matrix / norms

        meter.record(
            provider=self.id,
            model=self.model_id(),
            kind="embed",
            operation=operation or "embed",
            units=len(cleaned),
            input_tokens=sum(len(t) for t in cleaned) // 4,
        )

        return EmbedResult(
            vectors=matrix.astype(np.float32),
            provider=self.id,
            model=self.model_id(),
            dim=int(matrix.shape[1]),
        )
