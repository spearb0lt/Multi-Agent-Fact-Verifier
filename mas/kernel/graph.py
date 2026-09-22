"""The shape of a workflow: which agents exist and who may hand work to whom.

Routing here is dynamic. A node does not declare "and then the Writer runs"; it
returns the tasks it wants queued, and the Supervisor may return a different
set depending on what the Fact Checker found. That is what allows the Critic to
send a draft back for revision, and what makes this a graph with cycles rather
than a pipeline drawn as one.

Dynamic routing has an obvious failure mode: a node returning a target that
does not exist, or one nobody intended it to reach, stalls the run in a way
that looks like a hang. So the edges are declared anyway, and the orchestrator
checks every dynamic hop against them. The declaration is therefore two useful
things at once, a diagram the UI can draw without inferring anything, and a
contract that turns a routing typo into an immediate, named error.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .contracts import NodeResult, Task

if TYPE_CHECKING:  # pragma: no cover
    from .context import RunContext

Handler = Callable[["RunContext", Task], NodeResult]


@dataclass
class Node:
    """One position in the workflow, usually but not always one agent."""

    name: str
    handler: Handler
    agent: str = ""
    label: str = ""
    description: str = ""
    # How many times the orchestrator re-runs this node after an exception
    # before giving up on the run. Model calls fail transiently often enough
    # that one free retry is worth more than it costs.
    retries: int = 1
    # Whether several tasks queued for this node may run at the same time.
    parallel: bool = False

    def __post_init__(self) -> None:
        self.label = self.label or self.name.replace("_", " ").title()


@dataclass
class Edge:
    source: str
    target: str
    condition: str = ""


@dataclass
class Graph:
    """A named workflow: its nodes, its declared edges and where it starts."""

    name: str
    entry: str
    description: str = ""
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def node(
        self,
        name: str,
        *,
        agent: str = "",
        label: str = "",
        description: str = "",
        retries: int = 1,
        parallel: bool = False,
    ) -> Callable[[Handler], Handler]:
        """Register a handler as a node. Used as a decorator in a workflow module."""

        def decorate(handler: Handler) -> Handler:
            self.nodes[name] = Node(
                name=name,
                handler=handler,
                agent=agent,
                label=label,
                description=description,
                retries=retries,
                parallel=parallel,
            )
            return handler

        return decorate

    def edge(self, source: str, target: str, condition: str = "") -> None:
        self.edges.append(Edge(source, target, condition))

    def successors(self, name: str) -> set[str]:
        return {e.target for e in self.edges if e.source == name}

    def validate(self) -> None:
        """Check the graph is coherent before a run rather than during one."""
        if self.entry not in self.nodes:
            raise ValueError(f"Workflow '{self.name}' has no entry node '{self.entry}'.")
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(
                    f"Workflow '{self.name}' has an edge from unknown node '{edge.source}'."
                )
            if edge.target not in self.nodes:
                raise ValueError(
                    f"Workflow '{self.name}' has an edge to unknown node '{edge.target}'."
                )
        reachable = {self.entry}
        frontier = [self.entry]
        while frontier:
            for target in self.successors(frontier.pop()):
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        orphans = sorted(set(self.nodes) - reachable)
        if orphans:
            raise ValueError(
                f"Workflow '{self.name}' declares nodes nothing can reach: {', '.join(orphans)}."
            )

    def check_hop(self, source: str, target: str) -> None:
        """Reject a dynamic hop the graph never declared."""
        if target not in self.nodes:
            raise ValueError(
                f"Node '{source}' queued work for '{target}', which does not exist in "
                f"workflow '{self.name}'."
            )
        allowed = self.successors(source)
        if allowed and target not in allowed:
            raise ValueError(
                f"Node '{source}' queued work for '{target}', which is not a declared "
                f"successor. Declared: {', '.join(sorted(allowed)) or 'none'}."
            )

    def describe(self) -> dict[str, Any]:
        """The shape the UI draws, and the answer to 'what agents are in here'."""
        return {
            "name": self.name,
            "description": self.description,
            "entry": self.entry,
            "nodes": [
                {
                    "name": node.name,
                    "agent": node.agent,
                    "label": node.label,
                    "description": node.description,
                    "parallel": node.parallel,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "condition": e.condition}
                for e in self.edges
            ],
        }


_workflows: dict[str, Graph] = {}


def register(graph: Graph) -> Graph:
    graph.validate()
    _workflows[graph.name] = graph
    return graph


def get(name: str) -> Graph:
    if name not in _workflows:
        # Importing here rather than at module scope keeps the kernel free of a
        # dependency on the workflows that are built on top of it.
        from .. import workflows  # noqa: F401

    if name not in _workflows:
        raise LookupError(
            f"No workflow named '{name}'. Available: {', '.join(sorted(_workflows)) or 'none'}."
        )
    return _workflows[name]


def available() -> list[Graph]:
    from .. import workflows  # noqa: F401

    return [_workflows[name] for name in sorted(_workflows)]
