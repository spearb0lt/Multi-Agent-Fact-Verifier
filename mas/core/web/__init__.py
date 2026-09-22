"""Web access for tools: a robots aware HTTP client, search backends, extraction.

Importing this package registers every adapter, which is what makes
`adapter_for` able to find them.
"""
from .base import (
    HttpClient,
    RawItem,
    RobotsCache,
    SourceAdapter,
    SourceError,
    adapter_for,
    all_adapters,
    dedupe_items,
    describe_adapters,
    register,
    within_window,
)
from .extract import extract_article, iter_anchors
from .search import BACKENDS as SEARCH_BACKENDS
from .search import SearchAdapter

__all__ = [
    "HttpClient",
    "RawItem",
    "RobotsCache",
    "SEARCH_BACKENDS",
    "SearchAdapter",
    "SourceAdapter",
    "SourceError",
    "adapter_for",
    "all_adapters",
    "dedupe_items",
    "describe_adapters",
    "extract_article",
    "iter_anchors",
    "register",
    "within_window",
]
