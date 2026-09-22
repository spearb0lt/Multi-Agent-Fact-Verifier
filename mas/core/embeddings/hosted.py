"""Embedding backends reached over an API.

Embedding model names churn faster than chat model names and the vendors do not
always deprecate cleanly: Google retired text-embedding-004 in January 2026 and
its own docs disagree on whether the replacement is published as
`gemini-embedding-2` or `gemini-embedding-2-preview`. Rather than hardcode one
string and break on a rename, each adapter carries an ordered list of candidate
model ids, tries them in turn on first use, and remembers the one that answered.
A wrong guess costs one failed request at startup instead of a dead feature.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from .. import settings
from ..llm import keyring
from .base import BaseEmbedder, EmbeddingError


class _ResolvedModelMixin:
    """Finds a working model id once, from an ordered candidate list."""

    candidates: tuple[str, ...] = ()
    _resolved: str = ""
    _resolve_lock: threading.Lock

    def _candidate_list(self) -> list[str]:
        configured = (settings.EMBEDDING_MODEL or "").strip()
        # An explicit setting wins, but the known good ids stay as fallbacks so
        # a typo degrades instead of failing outright.
        return [configured, *self.candidates] if configured else list(self.candidates)

    def _resolve(self, attempt: Callable[[str], Any]) -> str:
        if self._resolved:
            return self._resolved
        with self._resolve_lock:
            if self._resolved:
                return self._resolved
            errors: list[str] = []
            for candidate in self._candidate_list():
                if not candidate:
                    continue
                try:
                    attempt(candidate)
                except Exception as exc:  # noqa: BLE001 - trying the next one
                    errors.append(f"{candidate}: {str(exc)[:120]}")
                    continue
                self._resolved = candidate
                return candidate
            raise EmbeddingError(
                f"No usable {getattr(self, 'label', 'embedding')} model id.",
                provider=getattr(self, "id", ""),
                hint=(
                    "Tried: " + "; ".join(errors[:4]) + ". Set EMBEDDING_MODEL to a "
                    "model id your account can call."
                ),
            )


def _key(slug: str, fallback: str | None) -> str:
    """Prefer the key this request brought over the deployment's own."""
    return keyring.key_for(slug) or (fallback or "")


class GeminiEmbedder(_ResolvedModelMixin, BaseEmbedder):
    id = "gemini_embed"
    label = "Google Gemini embeddings"
    docs_url = "https://aistudio.google.com/apikey"
    key_names = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    quality = 9
    dim = 768
    max_batch = 32
    # Measured on the eval set: same-event pairs run 0.87 to 0.96, unrelated up to 0.64. This space
    # sits much higher than the local encoder's, so the inherited defaults
    # of 0.72 and 0.90 would have over merged badly here.
    cluster_threshold = 0.80
    duplicate_threshold = 0.97
    candidates = ("gemini-embedding-2", "gemini-embedding-2-preview", "gemini-embedding-001")

    def __init__(self) -> None:
        self._resolve_lock = threading.Lock()
        self._clients: dict[str, Any] = {}

    @property
    def api_key(self) -> str:
        return _key("gemini", settings.GOOGLE_API_KEY)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def model_id(self) -> str:
        return self._resolved or self._candidate_list()[0]

    def _client(self):
        api_key = self.api_key
        if not api_key:
            raise EmbeddingError(self.unavailable_reason(), provider=self.id)
        tag = keyring.fingerprint(api_key)
        if (cached := self._clients.get(tag)) is not None:
            return cached
        try:
            from google import genai
        except ImportError as exc:
            raise EmbeddingError(
                "The google-genai package is not installed.", provider=self.id
            ) from exc
        if len(self._clients) >= 8:
            self._clients.clear()
        client = genai.Client(api_key=api_key)
        self._clients[tag] = client
        return client

    def _call(self, model: str, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        client = self._client()
        # Gemini's flexible output size is used to match the other providers'
        # storage cost rather than defaulting to 3072 floats per article.
        wanted = settings.EMBEDDING_DIMENSIONS or self.dim
        # Each text must be wrapped in its own Content. Handing the SDK a plain
        # list of strings makes gemini-embedding-2 read them as several parts of
        # one document and return a single vector for the whole batch, silently
        # on the wire and verified against the live API. The older
        # gemini-embedding-001 batches a bare list correctly, which is exactly
        # the kind of difference that makes this worth pinning down rather than
        # assuming.
        contents = [types.Content(parts=[types.Part(text=text)]) for text in texts]
        response = client.models.embed_content(
            model=model,
            contents=contents,
            config=types.EmbedContentConfig(
                output_dimensionality=wanted, task_type="SEMANTIC_SIMILARITY"
            ),
        )
        vectors = [list(item.values) for item in (response.embeddings or [])]
        if not vectors:
            raise EmbeddingError("Gemini returned no embeddings.", provider=self.id)
        self.dim = len(vectors[0])
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._resolve(lambda candidate: self._call(candidate, texts[:1]))
        return self._call(model, texts)


_CF_API_ROOT = "https://api.cloudflare.com/client/v4"


class CloudflareEmbedder(BaseEmbedder):
    """Workers AI embeddings. The only provider here with a recurring daily
    free allowance rather than a one time credit, which makes it the best
    hosted choice for a digest that runs every day."""

    id = "cloudflare_embed"
    label = "Cloudflare Workers AI embeddings"
    docs_url = "https://dash.cloudflare.com/profile/api-tokens"
    key_names = ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID")
    quality = 8
    dim = 768
    max_batch = 64
    # Measured on the eval set: same-event pairs run 0.86 to 0.93, unrelated up to 0.51. This space
    # sits much higher than the local encoder's, so the inherited defaults
    # of 0.72 and 0.90 would have over merged badly here.
    cluster_threshold = 0.80
    duplicate_threshold = 0.96

    @property
    def api_key(self) -> str:
        return _key("cloudflare", settings.CLOUDFLARE_API_TOKEN)

    @property
    def account_id(self) -> str:
        return keyring.account_for("cloudflare") or settings.CLOUDFLARE_ACCOUNT_ID or ""

    def is_available(self) -> bool:
        return bool(self.api_key and self.account_id)

    def unavailable_reason(self) -> str:
        if self.api_key and not self.account_id:
            return (
                "Cloudflare needs an account id as well as a token, because the "
                "account is part of the request URL."
            )
        return (
            "Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID, or paste both in "
            "Settings. The token needs the Workers AI permission."
        )

    def model_id(self) -> str:
        return settings.EMBEDDING_MODEL or "@cf/baai/bge-base-en-v1.5"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import requests

        model = self.model_id()
        url = f"{_CF_API_ROOT}/accounts/{self.account_id}/ai/run/{model}"
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"text": texts},
                timeout=90,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Cloudflare could not be reached: {exc}", provider=self.id
            ) from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Cloudflare returned HTTP {response.status_code}.",
                provider=self.id,
                hint=(
                    "The daily free allowance is 10,000 neurons. Response: "
                    f"{response.text[:200]}"
                ),
            )
        payload = response.json()
        vectors = ((payload.get("result") or {}).get("data")) or []
        if not vectors:
            raise EmbeddingError(
                "Cloudflare returned no embeddings.",
                provider=self.id,
                hint=str(payload.get("errors") or "")[:200],
            )
        self.dim = len(vectors[0])
        return vectors


class OpenAICompatibleEmbedder(BaseEmbedder):
    """Any endpoint that speaks the OpenAI embeddings protocol."""

    base_url: str = "https://api.openai.com/v1"
    key_slug: str = "openai"
    supports_dimensions: bool = False

    @property
    def api_key(self) -> str:  # pragma: no cover - overridden
        return ""

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import requests

        body: dict[str, Any] = {"model": self.model_id(), "input": texts}
        if self.supports_dimensions and settings.EMBEDDING_DIMENSIONS:
            body["dimensions"] = settings.EMBEDDING_DIMENSIONS
        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"{self.label} could not be reached: {exc}", provider=self.id
            ) from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"{self.label} returned HTTP {response.status_code}.",
                provider=self.id,
                hint=response.text[:250],
            )
        payload = response.json()
        # The protocol does not promise ordering, and at least one gateway does
        # return rows out of order, so they are sorted by index rather than
        # trusted as they arrive.
        rows = sorted(payload.get("data") or [], key=lambda r: r.get("index", 0))
        vectors = [row.get("embedding") or [] for row in rows]
        if not vectors or not vectors[0]:
            raise EmbeddingError(f"{self.label} returned no embeddings.", provider=self.id)
        self.dim = len(vectors[0])
        return vectors


class OpenAIEmbedder(OpenAICompatibleEmbedder):
    id = "openai_embed"
    label = "OpenAI embeddings"
    docs_url = "https://platform.openai.com/api-keys"
    key_names = ("OPENAI_API_KEY",)
    quality = 8
    dim = 1536
    max_batch = 128
    supports_dimensions = True

    def __init__(self) -> None:
        self.base_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1"

    @property
    def api_key(self) -> str:
        return _key("openai", settings.OPENAI_API_KEY)

    def model_id(self) -> str:
        return settings.EMBEDDING_MODEL or "text-embedding-3-small"


class _CandidateModelEmbedder(OpenAICompatibleEmbedder):
    """An OpenAI protocol embedder whose model id is discovered by trying.

    The first call walks the candidate list and keeps whichever id the account
    accepts. Subsequent calls go straight to it, so the cost of not hardcoding
    a name that vendors keep renaming is one extra request per process.
    """

    candidates: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._resolved = ""
        self._lock = threading.Lock()

    def _candidate_list(self) -> list[str]:
        configured = (settings.EMBEDDING_MODEL or "").strip()
        return [configured, *self.candidates] if configured else list(self.candidates)

    def model_id(self) -> str:
        return self._resolved or self._candidate_list()[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._resolved:
            return super()._embed_batch(texts)

        with self._lock:
            if self._resolved:
                return super()._embed_batch(texts)
            errors: list[str] = []
            for candidate in self._candidate_list():
                if not candidate:
                    continue
                self._resolved = candidate
                try:
                    vectors = super()._embed_batch(texts)
                except EmbeddingError as exc:
                    errors.append(f"{candidate}: {exc.message}")
                    self._resolved = ""
                    continue
                return vectors

        raise EmbeddingError(
            f"No usable {self.label} model id.",
            provider=self.id,
            hint=(
                "Tried: " + "; ".join(errors[:4]) + ". Set EMBEDDING_MODEL to a model "
                "id your account can call."
            ),
        )


class JinaEmbedder(_CandidateModelEmbedder):
    id = "jina_embed"
    label = "Jina AI embeddings"
    docs_url = "https://jina.ai/embeddings"
    key_names = ("JINA_API_KEY",)
    base_url = "https://api.jina.ai/v1"
    quality = 9
    dim = 768
    # Jina documents no batch ceiling, but a request large enough to time out
    # is still a failure, so this stays conservative.
    max_batch = 96
    candidates = (
        "jina-embeddings-v5-text-nano",
        "jina-embeddings-v5-text-small",
        "jina-embeddings-v3",
    )

    @property
    def api_key(self) -> str:
        return _key("jina", settings.JINA_API_KEY)


class VoyageEmbedder(_CandidateModelEmbedder):
    id = "voyage_embed"
    label = "Voyage AI embeddings"
    docs_url = "https://docs.voyageai.com"
    key_names = ("VOYAGE_API_KEY",)
    base_url = "https://api.voyageai.com/v1"
    quality = 9
    dim = 1024
    max_batch = 128
    candidates = ("voyage-4-lite", "voyage-4", "voyage-3.5-lite")

    @property
    def api_key(self) -> str:
        return _key("voyage", settings.VOYAGE_API_KEY)


class CohereEmbedder(BaseEmbedder):
    id = "cohere_embed"
    label = "Cohere embeddings"
    docs_url = "https://dashboard.cohere.com/api-keys"
    key_names = ("COHERE_API_KEY",)
    quality = 8
    dim = 1024
    # Cohere documents a hard ceiling of 96 texts per call.
    max_batch = 96

    @property
    def api_key(self) -> str:
        return _key("cohere", settings.COHERE_API_KEY)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def model_id(self) -> str:
        return settings.EMBEDDING_MODEL or "embed-v4.0"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import requests

        body: dict[str, Any] = {
            "model": self.model_id(),
            "texts": texts,
            # Clustering is the task these vectors are actually used for, and
            # Cohere tunes the output differently per input type.
            "input_type": "clustering",
            "embedding_types": ["float"],
        }
        if settings.EMBEDDING_DIMENSIONS:
            body["output_dimension"] = settings.EMBEDDING_DIMENSIONS
        try:
            response = requests.post(
                "https://api.cohere.com/v2/embed",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Cohere could not be reached: {exc}", provider=self.id
            ) from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Cohere returned HTTP {response.status_code}.",
                provider=self.id,
                hint=(
                    "A trial key allows 1,000 calls per month in total. Response: "
                    f"{response.text[:200]}"
                ),
            )
        payload = response.json()
        vectors = ((payload.get("embeddings") or {}).get("float")) or []
        if not vectors:
            raise EmbeddingError("Cohere returned no embeddings.", provider=self.id)
        self.dim = len(vectors[0])
        return vectors


class HuggingFaceEmbedder(BaseEmbedder):
    """Feature extraction through the Hugging Face inference router.

    The OpenAI compatible /v1 router is chat only, so this uses the task
    endpoint. Note that the free allowance is a very small monthly credit, so
    this is a convenience rather than a way to embed in bulk.
    """

    id = "hf_embed"
    label = "Hugging Face embeddings"
    docs_url = "https://huggingface.co/settings/tokens"
    key_names = ("HUGGINGFACE_API_KEY", "HF_TOKEN")
    quality = 5
    dim = 384
    max_batch = 32
    # Measured on the eval set: same-event pairs run 0.84 to 0.93, unrelated up to 0.41. This space
    # sits much higher than the local encoder's, so the inherited defaults
    # of 0.72 and 0.90 would have over merged badly here.
    cluster_threshold = 0.78
    duplicate_threshold = 0.96

    @property
    def api_key(self) -> str:
        return _key("huggingface", settings.HUGGINGFACE_API_KEY)

    def is_available(self) -> bool:
        return bool(self.api_key)

    def model_id(self) -> str:
        return settings.EMBEDDING_MODEL or "BAAI/bge-small-en-v1.5"

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import requests

        url = (
            "https://router.huggingface.co/hf-inference/models/"
            f"{self.model_id()}/pipeline/feature-extraction"
        )
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": texts, "options": {"wait_for_model": True}},
                timeout=180,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Hugging Face could not be reached: {exc}", provider=self.id
            ) from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Hugging Face returned HTTP {response.status_code}.",
                provider=self.id,
                hint=(
                    "The free inference credit is about 0.10 US dollars a month, "
                    f"which does not go far. Response: {response.text[:200]}"
                ),
            )
        payload = response.json()
        if not isinstance(payload, list) or not payload:
            raise EmbeddingError("Hugging Face returned no embeddings.", provider=self.id)

        # Some models pool for you and return one vector per input; others
        # return per token vectors that still need mean pooling.
        vectors: list[list[float]] = []
        for row in payload:
            if row and isinstance(row[0], list):
                width = len(row[0])
                sums = [0.0] * width
                for token_vector in row:
                    for index in range(width):
                        sums[index] += float(token_vector[index])
                vectors.append([value / len(row) for value in sums])
            else:
                vectors.append([float(v) for v in row])
        self.dim = len(vectors[0])
        return vectors
