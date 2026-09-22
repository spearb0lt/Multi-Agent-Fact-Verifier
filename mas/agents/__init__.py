"""The roles. One module each, so a reader can read one and understand it.

Every agent is instantiated once here. They hold no per run state, everything
they touch is on the context they are handed, which is what lets three
Researchers run at the same time as the same object.
"""
from .analyst import Analyst
from .base import Agent, claims_digest, evidence_digest, findings_digest
from .critic import Critic
from .editor import Editor
from .factchecker import FactChecker
from .planner import Planner
from .researcher import Researcher
from .supervisor import Supervisor
from .writer import Writer

planner = Planner()
researcher = Researcher()
analyst = Analyst()
factchecker = FactChecker()
supervisor = Supervisor()
writer = Writer()
editor = Editor()
critic = Critic()

ROSTER = {
    "Planner": planner,
    "Researcher": researcher,
    "Analyst": analyst,
    "FactChecker": factchecker,
    "Supervisor": supervisor,
    "Writer": writer,
    "Editor": editor,
    "Critic": critic,
}

__all__ = [
    "ROSTER", "Agent", "Analyst", "Critic", "Editor", "FactChecker", "Planner",
    "Researcher", "Supervisor", "Writer", "analyst", "claims_digest", "critic",
    "editor", "evidence_digest", "factchecker", "findings_digest", "planner",
    "researcher", "supervisor", "writer",
]
