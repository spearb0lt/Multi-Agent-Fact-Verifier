/**
 * The browser's view of the API.
 *
 * In production the Python process serves these pages as well as the API, so
 * every call is same origin and `API_BASE` is empty. In development Next
 * serves the pages on its own port, so the base is set from the environment
 * and the API's CORS list covers the dev origins.
 *
 * Nothing about the address is baked in at build time beyond that one
 * variable, which is what keeps a built bundle portable between a laptop, a
 * container and a Render service.
 */

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE || "").replace(/\/$/, "");

export type RunStatus =
  | "pending"
  | "running"
  | "paused"
  | "waiting"
  | "completed"
  | "failed"
  | "cancelled";

export interface Spend {
  steps: number;
  tokens: number;
  tokens_in: number;
  tokens_out: number;
  usd: number;
  seconds: number;
  tool_calls: number;
  llm_calls: number;
}

export interface Budget {
  max_steps: number;
  max_tokens: number;
  max_usd: number;
  max_seconds: number;
  max_tool_calls: number;
}

export interface Run {
  run_key: string;
  brief: string;
  title: string;
  workflow: string;
  status: RunStatus;
  phase: string;
  provider: string;
  model: string;
  cheap_model: string;
  error: string;
  pause_reason: string;
  spent: Spend;
  budget: Budget;
  config: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
  executing_here?: boolean;
}

export interface RunDetail extends Run {
  evidence_count: number;
  usage_by_agent: AgentUsage[];
  tool_usage: ToolUsage[];
  pending_approval: Approval | null;
  board: Board;
  queued: string[];
  graph: GraphShape;
}

export interface AgentUsage {
  agent: string;
  calls: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
}

export interface ToolUsage {
  tool: string;
  agent: string;
  calls: number;
  failures: number;
}

export interface Approval {
  id: number;
  node: string;
  question: string;
  options: string[];
  status: string;
}

export interface Board {
  brief?: string;
  title?: string;
  plan?: { title?: string; angle?: string; subquestions?: SubQuestion[] };
  subquestions?: SubQuestion[];
  findings?: Finding[];
  claims?: Claim[];
  verdicts?: Verdict[];
  draft?: { markdown?: string; words?: number; revision?: number };
  critiques?: Critique[];
  conflicts?: Conflict[];
  ruling?: Ruling;
  report?: { markdown?: string; title?: string };
  decisions?: Decision[];
  research_rounds?: number;
  revision?: number;
  [key: string]: unknown;
}

export interface SubQuestion {
  id: string;
  question: string;
  why?: string;
  priority?: string;
}

export interface Finding {
  id: string;
  statement: string;
  sources: string[];
  confidence: string;
  note?: string;
  found_by?: string;
  subquestion?: string;
}

export interface Claim {
  id: string;
  text: string;
  sources: string[];
  kind?: string;
  importance?: string;
  contested?: boolean;
  contest_note?: string;
}

export interface Verdict {
  claim_id: string;
  claim_text?: string;
  verdict: "supported" | "partly_supported" | "unsupported" | "contradicted";
  confidence: string;
  reasoning: string;
  sources: string[];
  independent_domains: number;
  domains: string[];
  corroborated: boolean;
  correction?: string;
}

export interface Critique {
  score: number;
  verdict: "accept" | "revise";
  bar: number;
  strengths: string[];
  issues: { severity: string; issue: string; fix: string }[];
  revision: number;
}

export interface Conflict {
  a: string;
  b: string;
  a_text: string;
  b_text: string;
  a_sources: string[];
  b_sources: string[];
  relation: "contradiction" | "tension";
  explanation: string;
  better_supported: "A" | "B" | "neither";
  why: string;
  similarity: number;
}

/** The claim_check workflow's product, which research_report does not have. */
export interface Ruling {
  verdict: string;
  confidence: string;
  reasoning: string;
  sources: string[];
  domains: string[];
  independent_domains: number;
  corroborated: boolean;
  what_would_settle_it: string;
}

export interface Decision {
  decision: "write" | "research_more";
  reasoning: string;
  gaps: string[];
  forced: boolean;
}

export interface TraceEvent {
  id: number;
  kind: string;
  agent: string;
  node: string;
  level: "info" | "warning" | "error";
  message: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface Message {
  id: number;
  sender: string;
  recipient: string;
  topic: string;
  content: unknown;
  created_at: string;
}

export interface Evidence {
  ref: string;
  url: string;
  domain: string;
  title: string;
  snippet: string;
  body: string;
  published_at: string | null;
  found_by: string;
  query: string;
}

export interface GraphShape {
  name: string;
  description: string;
  entry: string;
  nodes: { name: string; agent: string; label: string; description: string; parallel: boolean }[];
  edges: { source: string; target: string; condition: string }[];
}

export interface ProviderStatus {
  id: string;
  label: string;
  available: boolean;
  reason: string;
  models: { id: string; label?: string; cheap?: boolean }[];
  default_model: string;
  key_names: string[];
  local: boolean;
  needs_account: boolean;
}

export interface Config {
  app: string;
  tagline: string;
  runtime: { tier: string; persistent_disk: boolean; max_job_seconds: number };
  allow_client_keys: boolean;
  providers: ProviderStatus[];
  default: { provider?: string; model?: string };
  embedding: { id: string; available: boolean; reason?: string };
  search: { backends: string[]; keyed: string[]; keyless: string[]; usable: boolean };
  tools: { name: string; description: string; parameters: Record<string, unknown> }[];
  workflows: GraphShape[];
  defaults: Record<string, number>;
  pacing: Record<string, unknown>;
  worker: { active: number; capacity: number };
}

const PASSWORD_KEY = "mas.password";
const KEYS_KEY = "mas.keys";

export function storedPassword(): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(PASSWORD_KEY) || "";
  } catch {
    return "";
  }
}

export function setStoredPassword(value: string): void {
  try {
    if (value) window.localStorage.setItem(PASSWORD_KEY, value);
    else window.localStorage.removeItem(PASSWORD_KEY);
  } catch {
    /* Private browsing. The header is simply not sent. */
  }
}

export function storedKeys(): Record<string, string> {
  if (typeof window === "undefined") return {};
  try {
    return JSON.parse(window.localStorage.getItem(KEYS_KEY) || "{}");
  } catch {
    return {};
  }
}

export function setStoredKeys(keys: Record<string, string>): void {
  try {
    window.localStorage.setItem(KEYS_KEY, JSON.stringify(keys));
  } catch {
    /* Nothing to do; the keys just will not persist across reloads. */
  }
}

function headers(): Record<string, string> {
  const out: Record<string, string> = { "Content-Type": "application/json" };
  const password = storedPassword();
  if (password) out["X-App-Password"] = password;
  const keys = storedKeys();
  if (Object.keys(keys).length) {
    out["X-Provider-Keys"] = JSON.stringify({ keys });
  }
  return out;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}/api${path}`, { ...init, headers: headers() });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* A non JSON error body is still an error; the status line will do. */
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  config: () => request<Config>("/config"),
  workflows: () => request<{ workflows: GraphShape[] }>("/workflows"),

  listRuns: (limit = 30) => request<{ runs: Run[] }>(`/runs?limit=${limit}`),
  getRun: (key: string) => request<RunDetail>(`/runs/${key}`),
  createRun: (body: unknown) =>
    request<Run & { worker: unknown }>("/runs", { method: "POST", body: JSON.stringify(body) }),

  start: (key: string) => request(`/runs/${key}/start`, { method: "POST" }),
  pause: (key: string) => request(`/runs/${key}/pause`, { method: "POST" }),
  cancel: (key: string) => request(`/runs/${key}/cancel`, { method: "POST" }),
  resume: (key: string, body: unknown = {}) =>
    request(`/runs/${key}/resume`, { method: "POST", body: JSON.stringify(body) }),
  answer: (key: string, body: unknown) =>
    request(`/runs/${key}/answer`, { method: "POST", body: JSON.stringify(body) }),
  remove: (key: string) => request(`/runs/${key}`, { method: "DELETE" }),

  events: (key: string, after = 0) =>
    request<{ events: TraceEvent[]; cursor: number; status: RunStatus }>(
      `/runs/${key}/events?after=${after}&limit=500`,
    ),
  messages: (key: string) => request<{ messages: Message[] }>(`/runs/${key}/messages`),
  evidence: (key: string) => request<{ evidence: Evidence[] }>(`/runs/${key}/evidence`),
  artifacts: (key: string) =>
    request<{ artifacts: { kind: string; key: string; title: string; body: string; content: Record<string, unknown>; version: number }[] }>(
      `/runs/${key}/artifacts`,
    ),
  steps: (key: string) =>
    request<{ steps: { seq: number; node: string; agent: string; action: string; status: string; duration_ms: number; tokens_in: number; tokens_out: number; cost_usd: number }[] }>(
      `/runs/${key}/steps`,
    ),
  report: (key: string) =>
    request<{ report: { title: string; markdown: string; [k: string]: unknown }; final: boolean }>(
      `/runs/${key}/report`,
    ),
  reportUrl: (key: string, format: "markdown" | "html") =>
    `${API_BASE}/api/runs/${key}/report?format=${format}`,

  verifyKeys: (keys: Record<string, string>) =>
    request<{ providers: ProviderStatus[]; supplied: string[] }>("/keys/verify", {
      method: "POST",
      body: JSON.stringify({ credentials: { keys } }),
    }),
};

/**
 * Subscribe to a run's trace.
 *
 * EventSource cannot send headers, so a password protected deployment falls
 * back to polling rather than silently failing to connect. Polling is also the
 * fallback when the stream errors, which keeps a flaky proxy from freezing the
 * view.
 */
export function subscribe(
  key: string,
  after: number,
  onEvent: (event: TraceEvent) => void,
  onDone: () => void,
): () => void {
  let stopped = false;
  let cursor = after;

  if (storedPassword()) {
    const timer = setInterval(async () => {
      if (stopped) return;
      try {
        const page = await api.events(key, cursor);
        page.events.forEach((event) => {
          cursor = event.id;
          onEvent(event);
        });
        if (!["running", "pending"].includes(page.status)) onDone();
      } catch {
        /* Keep polling: a transient failure should not end the view. */
      }
    }, 1500);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }

  // A run takes minutes, and a stream held open that long gets dropped: by a
  // proxy, by a sleeping laptop, by the network changing. Without reconnecting
  // the trace silently freezes while the agents carry on working, which reads
  // as the run having hung. So the stream is reopened from the last event seen,
  // which is what the cursor is for, and no event is repeated or missed.
  let source: EventSource | null = null;
  let retry: ReturnType<typeof setTimeout> | null = null;
  let attempt = 0;
  let finished = false;

  const connect = () => {
    if (stopped || finished) return;
    source = new EventSource(`${API_BASE}/api/runs/${key}/stream?after=${cursor}`);

    source.onopen = () => {
      attempt = 0;
    };

    source.onmessage = (raw) => {
      try {
        const event = JSON.parse(raw.data) as TraceEvent;
        // Guard against a replay after a reconnect handing back an event the
        // caller has already rendered.
        if (event.id && event.id <= cursor) return;
        cursor = event.id ?? cursor;
        onEvent(event);
      } catch {
        /* A malformed frame is skipped rather than breaking the stream. */
      }
    };

    source.addEventListener("done", () => {
      finished = true;
      source?.close();
      onDone();
    });

    source.onerror = () => {
      source?.close();
      source = null;
      if (stopped || finished) return;
      // Tell the caller so it refetches the run's state, then reconnect with a
      // backoff that tops out quickly: a run is still going and the point is to
      // resume watching it, not to be polite to a server on the same machine.
      onDone();
      attempt += 1;
      const wait = Math.min(1000 * 2 ** (attempt - 1), 8000);
      retry = setTimeout(connect, wait);
    };
  };

  connect();

  return () => {
    stopped = true;
    if (retry) clearTimeout(retry);
    source?.close();
  };
}
