"""Pluggable text embedding, from a bundled local model to hosted APIs."""
from .base import (
    EMBED_CHAR_BUDGET,
    BaseEmbedder,
    EmbeddingError,
    EmbedResult,
    build_text,
    from_blob,
    to_blob,
)
from .registry import (
    all_status,
    available_embedders,
    cosine_matrix,
    embed_texts,
    embedder_map,
    most_similar,
    resolve,
    signature,
)

__all__ = [
    "EMBED_CHAR_BUDGET", "BaseEmbedder", "EmbedResult", "EmbeddingError",
    "build_text", "from_blob", "to_blob", "all_status", "available_embedders",
    "cosine_matrix", "embed_texts", "embedder_map", "most_similar", "resolve",
    "signature",
]
