"""The workflows this deployment can run.

Importing this package registers them, which is how `kernel.graph.get` finds a
workflow by name without the kernel depending on anything above it.
"""
from . import claim_check, research_report
from .render import build_report, to_html

__all__ = ["build_report", "claim_check", "research_report", "to_html"]
