"""Model access: nineteen providers behind one interface."""
import logging

# The Google SDK logs a warning about automatic function calling on every
# generate_content call. This project never uses AFC, so the warning says
# nothing and would otherwise print in front of every agent turn.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from .base import Completion, LLMError, ModelSpec, coerce_json, sanitise_output

__all__ = ["Completion", "LLMError", "ModelSpec", "coerce_json", "sanitise_output"]
