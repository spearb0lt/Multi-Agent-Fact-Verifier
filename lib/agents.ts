/**
 * Per role presentation, in one place.
 *
 * A role keeps its colour everywhere it appears: on the graph, in the trace,
 * in the cost table and on a message. Assigning by role rather than by
 * position is what makes the trace scannable, because the eye learns that
 * green is always the Fact Checker.
 *
 * The palette is chosen to stay distinguishable in both themes and to survive
 * the common forms of colour blindness, which is why it does not simply walk
 * around the hue wheel.
 */

export const ROLES = [
  "Planner",
  "Researcher",
  "Analyst",
  "FactChecker",
  "Reconciler",
  "Supervisor",
  "Writer",
  "Editor",
  "Critic",
] as const;

export type Role = (typeof ROLES)[number];

const COLOURS: Record<string, string> = {
  Planner: "#6a7fd4",
  Researcher: "#3f8f6f",
  Analyst: "#b5842f",
  FactChecker: "#2f8f9d",
  Reconciler: "#4a8fb5",
  Supervisor: "#8a6bbf",
  Writer: "#c06a4a",
  Editor: "#7a8a52",
  Critic: "#b04f6a",
};

const FALLBACK = "#7b7b72";

export function agentColour(agent: string): string {
  return COLOURS[agent] || FALLBACK;
}

export const ROLE_BLURB: Record<string, string> = {
  Planner: "Breaks the brief into separately researchable questions.",
  Researcher: "Searches, reads real pages and records cited findings.",
  Analyst: "Consolidates findings into distinct, checkable claims.",
  FactChecker: "Rules on whether the sources actually establish each claim.",
  Reconciler: "Decides whether any two verified claims actually conflict.",
  Supervisor: "Decides whether to write now or research the remaining gaps.",
  Writer: "Drafts the report from verified claims only.",
  Editor: "Tightens the draft without changing what it asserts.",
  Critic: "Scores the report and sends it back if it is not good enough.",
};

/** Which node in the flagship workflow each role occupies. */
export const ROLE_NODE: Record<string, string> = {
  Planner: "plan",
  Researcher: "research",
  Analyst: "analyse",
  FactChecker: "verify",
  Reconciler: "reconcile",
  Supervisor: "supervise",
  Writer: "write",
  Editor: "edit",
  Critic: "critique",
};

export function formatUsd(value: number): string {
  if (!value) return "$0";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${rest}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

export const STATUS_TONE: Record<string, { label: string; tone: string }> = {
  pending: { label: "Pending", tone: "var(--muted)" },
  running: { label: "Running", tone: "var(--accent)" },
  paused: { label: "Paused", tone: "var(--warn)" },
  waiting: { label: "Needs you", tone: "var(--warn)" },
  completed: { label: "Completed", tone: "var(--ok)" },
  failed: { label: "Failed", tone: "var(--error)" },
  cancelled: { label: "Cancelled", tone: "var(--faint)" },
};
