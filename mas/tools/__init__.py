"""Every capability an agent can invoke.

Importing this package registers the tools. Nothing else imports the modules
individually, so a tool that is not reachable from here does not exist as far
as the agents are concerned, which makes the set of capabilities a single
readable list rather than a search of the source tree.
"""
from . import reason, web
from .web import search_status

__all__ = ["reason", "search_status", "web"]
