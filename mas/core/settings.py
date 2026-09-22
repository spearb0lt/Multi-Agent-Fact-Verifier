"""Central configuration, read entirely from the environment.

Nothing here raises at import time. A missing credential means the matching
provider or tool reports itself unavailable, so the app boots and does useful
work with whatever subset of keys happens to be present, including none at all.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .runtime import current as runtime

# mas/core/settings.py, so three parents up is the repository root.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "").strip() or (BASE_DIR / "Semantic Models"))


def _warn_duplicate_env_keys(path: Path) -> None:
    """Complain about a name that appears twice in a .env file.

    python-dotenv reads top to bottom and the last assignment wins, so a
    duplicate line lower down silently replaces a real key higher up with an
    empty string. That is invisible: the file plainly contains the key, and the
    provider simply reports itself as unconfigured.
    """
    import re
    from collections import Counter

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    names = re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", text, re.M)
    repeated = sorted(name for name, count in Counter(names).items() if count > 1)
    if repeated:
        logging.getLogger(__name__).warning(
            "%s defines these names more than once, and the last one wins, which "
            "silently discards any value set earlier: %s",
            path.name,
            ", ".join(repeated),
        )


if os.environ.get("IGNORE_DOTENV", "").strip().lower() not in {"1", "true", "yes"}:
    try:
        from dotenv import load_dotenv

        _env_file = BASE_DIR / ".env"
        if _env_file.exists():
            _warn_duplicate_env_keys(_env_file)
            load_dotenv(_env_file, override=False)
    except ImportError:  # python-dotenv is optional in a slim deployment
        pass


def env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw and raw.strip() else default
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw and raw.strip() else default
    except ValueError:
        return default


def env_list(name: str, default: str = "") -> list[str]:
    raw = env(name, default=default) or ""
    return [part.strip() for part in raw.split(",") if part.strip()]


# ---------------------------------------------------------------- application

APP_NAME = env("APP_NAME", default="Multi-Agentic")
APP_TAGLINE = env(
    "APP_TAGLINE",
    default="A supervised team of agents that researches, verifies and writes",
)
ENVIRONMENT = env("ENVIRONMENT", default="production")
DEBUG = env_bool("DEBUG", False)
PUBLIC_BASE_URL = (env("PUBLIC_BASE_URL", "VERCEL_URL", default="") or "").rstrip("/")
if PUBLIC_BASE_URL and not PUBLIC_BASE_URL.startswith("http"):
    PUBLIC_BASE_URL = f"https://{PUBLIC_BASE_URL}"

CORS_ORIGINS = env_list("CORS_ORIGINS")

# Optional gate on the whole API for a deployment the operator does not want
# strangers spending their keys on.
APP_PASSWORD = env("APP_PASSWORD")

# Bring your own key. When on, a visitor pastes their own provider key in the
# UI and it rides their requests only, so a public deployment costs the
# operator nothing. Turn off for a private instance that must use server keys.
ALLOW_CLIENT_KEYS = env_bool("ALLOW_CLIENT_KEYS", True)

# ------------------------------------------------------------------- storage

DATABASE_URL = env("DATABASE_URL", "POSTGRES_URL", "POSTGRES_PRISMA_URL", "NEON_DATABASE_URL")
DATA_DIR = Path(env("DATA_DIR", default=str(runtime().writable_dir)))
SQLITE_PATH = Path(env("SQLITE_PATH", default=str(DATA_DIR / "agentic.db")))

# ---------------------------------------------------------- LLM provider keys

GOOGLE_API_KEY = env("GOOGLE_API_KEY", "GEMINI_API_KEY")
GROQ_API_KEY = env("GROQ_API_KEY")
OPENAI_API_KEY = env("OPENAI_API_KEY")
OPENAI_BASE_URL = env("OPENAI_BASE_URL", default="https://api.openai.com/v1")
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
MISTRAL_API_KEY = env("MISTRAL_API_KEY")
DEEPSEEK_API_KEY = env("DEEPSEEK_API_KEY")
TOGETHER_API_KEY = env("TOGETHER_API_KEY", "TOGETHERAI_API_KEY")
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY")
CEREBRAS_API_KEY = env("CEREBRAS_API_KEY")
SAMBANOVA_API_KEY = env("SAMBANOVA_API_KEY")
XAI_API_KEY = env("XAI_API_KEY", "GROK_API_KEY")
FIREWORKS_API_KEY = env("FIREWORKS_API_KEY")
PERPLEXITY_API_KEY = env("PERPLEXITY_API_KEY")
HUGGINGFACE_API_KEY = env(
    "HUGGINGFACE_API_KEY", "HUGGINGFACE_API_TOKEN", "HF_TOKEN", "HF_API_KEY"
)
HUGGINGFACE_BASE_URL = env(
    "HUGGINGFACE_BASE_URL", "HF_BASE_URL", default="https://router.huggingface.co/v1"
)

# Cloudflare Workers AI needs two values, because the account id sits in the
# URL path rather than a header and a token alone addresses nothing.
CLOUDFLARE_API_TOKEN = env("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY", "CF_API_TOKEN")
CLOUDFLARE_ACCOUNT_ID = env("CLOUDFLARE_ACCOUNT_ID", "CF_ACCOUNT_ID")

# OpenAI protocol servers running on the operator's own machine. These are only
# reachable from a server tier deployment.
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "OLLAMA_HOST", default="http://localhost:11434")
LMSTUDIO_BASE_URL = env("LMSTUDIO_BASE_URL", default="http://localhost:1234/v1")
LLAMACPP_BASE_URL = env("LLAMACPP_BASE_URL", default="http://localhost:8080/v1")
VLLM_BASE_URL = env("VLLM_BASE_URL", default="http://localhost:8000/v1")
VLLM_API_KEY = env("VLLM_API_KEY", default="EMPTY")

# Default model overrides, one per provider.
MODEL_OVERRIDES = {
    "gemini": env("GEMINI_MODEL"),
    "groq": env("GROQ_MODEL"),
    "openai": env("OPENAI_MODEL"),
    "anthropic": env("ANTHROPIC_MODEL"),
    "mistral": env("MISTRAL_MODEL"),
    "deepseek": env("DEEPSEEK_MODEL"),
    "together": env("TOGETHER_MODEL"),
    "openrouter": env("OPENROUTER_MODEL"),
    "cerebras": env("CEREBRAS_MODEL"),
    "sambanova": env("SAMBANOVA_MODEL"),
    "xai": env("XAI_MODEL"),
    "fireworks": env("FIREWORKS_MODEL"),
    "perplexity": env("PERPLEXITY_MODEL"),
    "huggingface": env("HUGGINGFACE_MODEL", "HF_MODEL"),
    "cloudflare": env("CLOUDFLARE_MODEL", "CF_MODEL"),
    "ollama": env("OLLAMA_MODEL"),
    "lmstudio": env("LMSTUDIO_MODEL"),
    "llamacpp": env("LLAMACPP_MODEL"),
    "vllm": env("VLLM_MODEL"),
}

# Extra model ids, comma separated, appended to a provider's picker.
EXTRA_MODELS = {
    key: env(f"{key.upper()}_EXTRA_MODELS", default="") for key in MODEL_OVERRIDES
}

# --------------------------------------------------------------- embeddings

# Which embedding backend to prefer. "auto" walks the registry in order and
# takes the first that is usable, which on a bare deployment is the bundled
# ONNX model and therefore needs no key at all.
EMBEDDING_PROVIDER = env("EMBEDDING_PROVIDER", default="auto")
EMBEDDING_MODEL = env("EMBEDDING_MODEL")
EMBEDDING_DIMENSIONS = env_int("EMBEDDING_DIMENSIONS", 0)  # 0 means provider default
EMBEDDING_BATCH_SIZE = env_int("EMBEDDING_BATCH_SIZE", 64)

LOCAL_EMBED_MODEL_DIR = Path(
    env("LOCAL_EMBED_MODEL_DIR", default=str(MODELS_DIR / "all-MiniLM-L6-v2-onnx"))
)
LOCAL_EMBED_ENABLED = env_bool("LOCAL_EMBED_ENABLED", True)

JINA_API_KEY = env("JINA_API_KEY")
VOYAGE_API_KEY = env("VOYAGE_API_KEY")
COHERE_API_KEY = env("COHERE_API_KEY")

# ------------------------------------------------------------- research tools

# Web search backends the Researcher agent can call.
SERPAPI_KEY = env("SERPAPI_KEY", "SERP_API_KEY")
BRAVE_API_KEY = env("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY")
TAVILY_API_KEY = env("TAVILY_API_KEY")
EXA_API_KEY = env("EXA_API_KEY", "METAPHOR_API_KEY")
SEARXNG_BASE_URL = env("SEARXNG_BASE_URL")

# Keyless breadth. Google News RSS and Wikipedia both answer a plain query with
# no credential, which is what keeps the system usable on a bare clone.
ENABLE_KEYLESS_SEARCH = env_bool("ENABLE_KEYLESS_SEARCH", True)

# ----------------------------------------------------------------- web access

USER_AGENT = env(
    "USER_AGENT",
    default=(
        "MultiAgenticBot/1.0 (+https://github.com/; autonomous research agent; "
        "contact via repository issues)"
    ),
)
RESPECT_ROBOTS = env_bool("RESPECT_ROBOTS", True)
FETCH_TIMEOUT = env_int("FETCH_TIMEOUT", 20)
FETCH_CONCURRENCY = env_int("FETCH_CONCURRENCY", 8)
PER_HOST_DELAY = env_float("PER_HOST_DELAY", 1.0)
MAX_ARTICLE_BYTES = env_int("MAX_ARTICLE_BYTES", 2_000_000)

# Vector space thresholds, used here to spot two sources telling the same story
# so that corroboration is counted across outlets rather than across reprints.
DUPLICATE_THRESHOLD = env_float("DUPLICATE_THRESHOLD", 0.86)
CLUSTER_THRESHOLD = env_float("CLUSTER_THRESHOLD", 0.72)

# ------------------------------------------------------------- agent defaults

# The provider and model a run uses when the caller names neither. Empty means
# "pick the first available provider", which on this deployment is whichever
# free tier key happens to be present.
DEFAULT_PROVIDER = env("DEFAULT_PROVIDER", default="")
DEFAULT_MODEL = env("DEFAULT_MODEL", default="")

# Hard ceilings on a single run. Every one of these pauses the run rather than
# killing it, so a breach is recoverable: raise the cap and resume.
MAX_RUN_STEPS = env_int("MAX_RUN_STEPS", 120)
MAX_RUN_TOKENS = env_int("MAX_RUN_TOKENS", 400_000)
MAX_RUN_USD = env_float("MAX_RUN_USD", 0.50)
MAX_RUN_SECONDS = env_int("MAX_RUN_SECONDS", 1800)
MAX_TOOL_CALLS = env_int("MAX_TOOL_CALLS", 80)

# How hard a single agent is allowed to think before the kernel stops it. This
# is the ReAct loop bound, not the run bound.
MAX_AGENT_ITERATIONS = env_int("MAX_AGENT_ITERATIONS", 8)

# How many researchers work at once. Free tiers are rate limited per minute, so
# the useful number is small.
AGENT_CONCURRENCY = env_int("AGENT_CONCURRENCY", 3)

# How many times the Critic may send the draft back for revision.
MAX_REVISIONS = env_int("MAX_REVISIONS", 2)

# How many times the Supervisor may send the researchers back out. Each round
# is a full plan, research, analyse and verify cycle, so this is the single
# biggest multiplier on what a run costs.
MAX_RESEARCH_ROUNDS = env_int("MAX_RESEARCH_ROUNDS", 2)

# The Critic's pass mark, out of 100. Below it the draft goes back to the
# Writer with the critique attached, until MAX_REVISIONS is spent.
QUALITY_BAR = env_int("QUALITY_BAR", 75)

# A claim needs this many independent domains behind it to count as
# corroborated. Two is the journalistic convention and the default here.
CORROBORATION_MIN = env_int("CORROBORATION_MIN", 2)

# Write a checkpoint after every step. Turning this off makes a run faster and
# unresumable, which is only sensible in a throwaway benchmark.
CHECKPOINT_EVERY_STEP = env_bool("CHECKPOINT_EVERY_STEP", True)

# Seconds between control channel polls while a run is executing. The control
# channel is what makes pause and cancel work from another process.
CONTROL_POLL_SECONDS = env_float("CONTROL_POLL_SECONDS", 1.0)

# ------------------------------------------------------------------ behaviour

DEFAULT_TIMEZONE = env("DEFAULT_TIMEZONE", default="Asia/Kolkata")
