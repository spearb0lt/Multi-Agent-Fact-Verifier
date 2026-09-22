"""Backends that need no API key: a lexical fallback, ONNX, and local servers.

The lexical embedder is the floor. It always works, has no dependencies beyond
numpy, and is what keeps a fresh deployment with no keys from being useless. It
is genuinely weaker than a real sentence encoder at recognising two differently
worded reports of the same event, which is why it sorts last in the registry,
but it catches near identical wire copy well.

The ONNX embedder is the intended default. A quantised MiniLM is about 23 MB
and onnxruntime about 64 MB unpacked, which fits inside a serverless bundle
where PyTorch, at roughly a gigabyte, does not.
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from .. import settings
from ..runtime import current as runtime
from ..util import normalise_text, tcp_reachable, tokens
from .base import BaseEmbedder, EmbeddingError


class LexicalEmbedder(BaseEmbedder):
    """A hashed bag of words and character n-grams, projected to a fixed size.

    This is the signed hashing trick: each feature is hashed to a column and to
    a sign bit, so collisions cancel on average instead of accumulating. Term
    frequency is damped logarithmically because a word repeated ten times in an
    article is not ten times as informative as one used once.

    Character 4-grams are mixed in alongside words so that the vector survives
    small spelling and inflection differences between outlets covering the same
    story.
    """

    id = "lexical"
    label = "Built in lexical matching"
    local = True
    quality = 1
    dim = 1024
    max_batch = 512
    # Measured on the eval set: same-event pairs land between 0.29 and 0.57,
    # unrelated text below 0.07. An earlier cut of 0.33 sat ABOVE the lowest
    # true pair, so this backend was silently failing to group events it should
    # have. The cuts are far lower than any encoder's, which is exactly why they
    # cannot be shared.
    duplicate_threshold = 0.62
    cluster_threshold = 0.24

    def is_available(self) -> bool:
        return True

    def unavailable_reason(self) -> str:
        return ""

    def model_id(self) -> str:
        return f"hashed-ngrams-{self.dim}"

    @staticmethod
    def _feature_slots(feature: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        column = value % dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return column, sign

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        out: list[list[float]] = []
        for text in texts:
            counts: dict[str, int] = {}

            words = tokens(text)
            for word in words:
                counts[f"w:{word}"] = counts.get(f"w:{word}", 0) + 1
            # Deliberately not strict: pairing a list with its own tail is
            # meant to stop one short.
            for first, second in zip(words, words[1:], strict=False):
                key = f"b:{first}_{second}"
                counts[key] = counts.get(key, 0) + 1

            flat = normalise_text(text).replace(" ", "_")
            for index in range(max(0, len(flat) - 3)):
                key = f"c:{flat[index : index + 4]}"
                counts[key] = counts.get(key, 0) + 1

            vector = np.zeros(self.dim, dtype=np.float32)
            for feature, count in counts.items():
                column, sign = self._feature_slots(feature, self.dim)
                vector[column] += sign * (1.0 + np.log(count))
            out.append(vector.tolist())
        return out


class OnnxEmbedder(BaseEmbedder):
    """A sentence transformer running under ONNX Runtime, with no PyTorch.

    The session and tokenizer are built once and reused. On a serverless
    platform an instance may serve many invocations, so paying the load cost
    per process rather than per call is the difference between a usable cold
    path and an unusable one.
    """

    id = "onnx"
    label = "Local ONNX sentence encoder"
    docs_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
    local = True
    quality = 6
    dim = 384
    max_batch = 32
    # MiniLM was trained at 256 tokens and a headline plus lead paragraph fits
    # well inside that, so a longer window would cost time for no accuracy.
    max_tokens = 256
    # Measured on the eval set: same-event pairs run 0.72 to 0.85, unrelated
    # text below 0.04. The duplicate cut sits above the top of the same-event
    # range so independent reporting is grouped rather than discarded.
    duplicate_threshold = 0.90
    cluster_threshold = 0.68

    def __init__(self) -> None:
        self._session: Any = None
        self._tokenizer: Any = None
        self._input_names: tuple[str, ...] = ()
        self._lock = threading.Lock()
        self._load_error = ""

    @property
    def model_dir(self) -> Path:
        return Path(settings.LOCAL_EMBED_MODEL_DIR)

    def _weights_path(self) -> Path | None:
        directory = self.model_dir
        if not directory.is_dir():
            return None
        # The quantised export is preferred: about a quarter the size for a
        # negligible accuracy loss at this task.
        for name in (
            "model_qint8_avx512.onnx",
            "model_quantized.onnx",
            "model_qint8.onnx",
            "model.onnx",
        ):
            candidate = directory / name
            if candidate.is_file():
                return candidate
        found = sorted(directory.glob("*.onnx"))
        return found[0] if found else None

    def is_available(self) -> bool:
        if not settings.LOCAL_EMBED_ENABLED:
            return False
        if not runtime().can("onnx_models"):
            return False
        return self._weights_path() is not None and (self.model_dir / "tokenizer.json").is_file()

    def unavailable_reason(self) -> str:
        if not settings.LOCAL_EMBED_ENABLED:
            return "Disabled by LOCAL_EMBED_ENABLED=0."
        if not runtime().can("onnx_models"):
            return runtime().reason("onnx_models")
        if self._weights_path() is None:
            return (
                f"No ONNX model found in {self.model_dir}. Run "
                "'python scripts/fetch_model.py' to download it, about 25 MB."
            )
        if not (self.model_dir / "tokenizer.json").is_file():
            return (
                f"tokenizer.json is missing from {self.model_dir}. Run "
                "'python scripts/fetch_model.py' again."
            )
        return self._load_error or "The local model could not be loaded."

    def model_id(self) -> str:
        return self.model_dir.name or "all-MiniLM-L6-v2-onnx"

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            weights = self._weights_path()
            if weights is None:
                raise EmbeddingError(self.unavailable_reason(), provider=self.id)
            try:
                import onnxruntime
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise EmbeddingError(
                    "onnxruntime and tokenizers are required for the local embedder.",
                    provider=self.id,
                    hint="pip install onnxruntime tokenizers",
                ) from exc

            options = onnxruntime.SessionOptions()
            # A serverless container is billed for one CPU, and letting ORT
            # spawn a thread pool there costs more in contention than it saves.
            options.intra_op_num_threads = 1 if runtime().tier.value == "serverless" else 0
            options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            try:
                session = onnxruntime.InferenceSession(
                    str(weights), options, providers=["CPUExecutionProvider"]
                )
                tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
            except Exception as exc:  # noqa: BLE001 - reported to the caller
                self._load_error = str(exc)[:300]
                raise EmbeddingError(
                    f"The local ONNX model failed to load: {exc}",
                    provider=self.id,
                    hint="Delete the model directory and run scripts/fetch_model.py again.",
                ) from exc

            tokenizer.enable_truncation(max_length=self.max_tokens)
            tokenizer.enable_padding(length=None)
            self._tokenizer = tokenizer
            self._input_names = tuple(i.name for i in session.get_inputs())
            self._session = session

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        self._ensure_loaded()
        encoded = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed: dict[str, Any] = {}
        for name in self._input_names:
            if name == "input_ids":
                feed[name] = input_ids
            elif name == "attention_mask":
                feed[name] = attention
            elif name == "token_type_ids":
                # Present in BERT style exports and always zero for a single
                # segment, but the graph fails without it when it is declared.
                feed[name] = np.array([e.type_ids for e in encoded], dtype=np.int64)

        outputs = self._session.run(None, feed)
        hidden = outputs[0]
        if hidden.ndim == 2:
            # Some exports pool internally and hand back sentence vectors.
            return [row.tolist() for row in hidden]

        # Mean pooling over real tokens only. Averaging the padding as well
        # would drag short headlines toward a common vector and quietly inflate
        # their similarity to each other.
        mask = attention.astype(np.float32)[:, :, None]
        summed = (hidden * mask).sum(axis=1)
        counts = np.maximum(mask.sum(axis=1), 1e-9)
        return [row.tolist() for row in (summed / counts)]


class SentenceTransformersEmbedder(BaseEmbedder):
    """The full PyTorch sentence-transformers stack, on a server tier only."""

    id = "sentence_transformers"
    label = "sentence-transformers (local, PyTorch)"
    docs_url = "https://www.sbert.net"
    local = True
    quality = 7
    dim = 384
    max_batch = 64

    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return runtime().can("torch_models")

    def unavailable_reason(self) -> str:
        return runtime().reason("torch_models")

    def model_id(self) -> str:
        return settings.EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed.",
                    provider=self.id,
                    hint="pip install -r requirements-server.txt",
                ) from exc
            local_dir = settings.MODELS_DIR / self.model_id().split("/")[-1]
            source = str(local_dir) if local_dir.is_dir() else self.model_id()
            model = SentenceTransformer(source, cache_folder=str(settings.MODELS_DIR))
            self.dim = int(model.get_sentence_embedding_dimension() or self.dim)
            self._model = model

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        vectors = self._model.encode(
            texts, batch_size=self.max_batch, show_progress_bar=False, convert_to_numpy=True
        )
        return [row.tolist() for row in vectors]


class OllamaEmbedder(BaseEmbedder):
    """Embeddings from a local Ollama server."""

    id = "ollama_embed"
    label = "Ollama (local embeddings)"
    docs_url = "https://ollama.com/search?c=embedding"
    key_names = ("OLLAMA_BASE_URL",)
    local = True
    quality = 6
    dim = 768
    max_batch = 32

    def __init__(self) -> None:
        self._probe: tuple[float, bool] = (0.0, False)

    @property
    def base_url(self) -> str:
        root = (settings.OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
        return root[: -len("/v1")] if root.endswith("/v1") else root

    def model_id(self) -> str:
        return settings.EMBEDDING_MODEL or "nomic-embed-text"

    def is_available(self) -> bool:
        import time

        if not runtime().can("local_llm"):
            return False
        stamp, ok = self._probe
        if time.monotonic() - stamp < (30.0 if ok else 180.0):
            return ok
        ok = False
        # See util.tcp_reachable: an HTTP call to a closed local port is slow
        # enough that four of them dominated every cold configuration request.
        if tcp_reachable(self.base_url):
            try:
                import requests

                ok = requests.get(f"{self.base_url}/api/tags", timeout=2.0).status_code < 500
            except Exception:  # noqa: BLE001 - not running is the expected case
                ok = False
        self._probe = (time.monotonic(), ok)
        return ok

    def unavailable_reason(self) -> str:
        if not runtime().can("local_llm"):
            return runtime().reason("local_llm")
        return (
            f"Ollama is not answering at {self.base_url}. Start it and pull an "
            f"embedding model: ollama pull {self.model_id()}"
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        import requests

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_id(), "input": texts},
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(
                f"Ollama could not be reached: {exc}", provider=self.id
            ) from exc
        if response.status_code >= 400:
            raise EmbeddingError(
                f"Ollama returned HTTP {response.status_code}.",
                provider=self.id,
                hint=(
                    f"Pull the model first: ollama pull {self.model_id()}. "
                    f"Response: {response.text[:200]}"
                ),
            )
        payload = response.json()
        vectors = payload.get("embeddings") or []
        if not vectors:
            raise EmbeddingError("Ollama returned no embeddings.", provider=self.id)
        self.dim = len(vectors[0])
        return vectors
