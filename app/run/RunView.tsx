"use client";

/**
 * One run, live.
 *
 * The layout puts the graph and the trace side by side on purpose: the graph
 * answers "where is it" and the trace answers "what is it doing", and watching
 * a run is mostly moving between those two questions. Everything else is in
 * tabs below, because it is read after the fact rather than watched.
 *
 * Node state is derived from the event stream rather than stored. The server
 * already says which node was entered and exited, so recomputing here keeps
 * the picture correct after a reconnect without the server having to maintain
 * a second view of its own progress.
 */

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AgentGraph, { type NodeState } from "@/components/AgentGraph";
import BoardView from "@/components/BoardView";
import BudgetMeter from "@/components/BudgetMeter";
import Controls from "@/components/Controls";
import Markdown from "@/components/Markdown";
import TeamPanel from "@/components/TeamPanel";
import Trace from "@/components/Trace";
import type { Config, Evidence, Message, RunDetail, TraceEvent } from "@/lib/api";
import { api, subscribe } from "@/lib/api";
import { STATUS_TONE, agentColour, formatDuration, formatTokens, formatUsd } from "@/lib/agents";

type Tab = "board" | "report" | "messages" | "steps";

export default function RunView({ runKey }: { runKey: string }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [evidence, setEvidence] = useState<Evidence[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [report, setReport] = useState<{ title: string; markdown: string } | null>(null);
  const [steps, setSteps] = useState<Awaited<ReturnType<typeof api.steps>>["steps"]>([]);
  const [tab, setTab] = useState<Tab>("board");
  const [error, setError] = useState("");
  const [answer, setAnswer] = useState("");
  const cursor = useRef(0);

  const live = run?.status === "running";

  const loadRun = useCallback(async () => {
    try {
      const detail = await api.getRun(runKey);
      setRun(detail);
      setError("");
      return detail;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, [runKey]);

  const loadExtras = useCallback(async () => {
    const [ev, msg, stepList] = await Promise.allSettled([
      api.evidence(runKey),
      api.messages(runKey),
      api.steps(runKey),
    ]);
    if (ev.status === "fulfilled") setEvidence(ev.value.evidence);
    if (msg.status === "fulfilled") setMessages(msg.value.messages);
    if (stepList.status === "fulfilled") setSteps(stepList.value.steps);
    try {
      const r = await api.report(runKey);
      setReport(r.report as { title: string; markdown: string });
    } catch {
      setReport(null);
    }
  }, [runKey]);

  useEffect(() => {
    api.config().then(setConfig).catch(() => undefined);
    loadRun().then(() => loadExtras());
  }, [loadRun, loadExtras]);

  // The trace streams; everything else is pulled when the stream goes quiet or
  // the run changes state. Streaming the whole run state would send the same
  // large board object on every event for no benefit.
  useEffect(() => {
    let cancelled = false;
    const stop = subscribe(
      runKey,
      cursor.current,
      (event) => {
        if (cancelled) return;
        cursor.current = Math.max(cursor.current, event.id || 0);
        setEvents((prev) => (prev.some((e) => e.id === event.id) ? prev : [...prev, event]));
        if (
          event.kind.startsWith("run.") ||
          event.kind === "node.exit" ||
          event.kind === "artifact.created" ||
          event.kind === "approval.requested"
        ) {
          loadRun();
        }
        if (event.kind === "evidence.added" || event.kind === "artifact.created") {
          loadExtras();
        }
      },
      () => {
        if (cancelled) return;
        loadRun();
        loadExtras();
      },
    );
    return () => {
      cancelled = true;
      stop();
    };
  }, [runKey, loadRun, loadExtras]);

  // While a run executes, the spend counters move between events, so a slow
  // poll keeps the budget meter honest without a stream of its own.
  useEffect(() => {
    if (!live) return;
    const timer = setInterval(loadRun, 4000);
    return () => clearInterval(timer);
  }, [live, loadRun]);

  const nodeStates = useMemo(() => {
    const states: Record<string, NodeState> = {};
    const counts: Record<string, number> = {};
    events.forEach((event) => {
      if (!event.node) return;
      if (event.kind === "node.enter") {
        states[event.node] = "active";
        counts[event.node] = (counts[event.node] || 0) + 1;
      } else if (event.kind === "node.exit") {
        if (states[event.node] !== "error") states[event.node] = "done";
      } else if (event.kind === "node.error") {
        states[event.node] = "error";
      }
    });
    // A run that is not executing has no active node, whatever the last event
    // said, so a paused run does not sit there pulsing forever.
    if (run && run.status !== "running") {
      Object.keys(states).forEach((node) => {
        if (states[node] === "active") states[node] = run.status === "failed" ? "error" : "done";
      });
    }
    return { states, counts };
  }, [events, run]);

  const activeAgent = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i -= 1) {
      if (events[i].kind === "agent.start") return events[i].agent;
      if (events[i].kind.startsWith("run.") && events[i].kind !== "run.started") return "";
    }
    return "";
  }, [events]);

  if (error && !run) {
    return (
      <div className="panel p-5 max-w-lg mx-auto mt-10">
        <h2 className="font-semibold mb-2">Cannot load this run</h2>
        <p className="text-sm mb-3" style={{ color: "var(--muted)" }}>{error}</p>
        <Link href="/" className="btn">Back to runs</Link>
      </div>
    );
  }

  if (!run || !config) {
    return <p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>Loading...</p>;
  }

  const tone = STATUS_TONE[run.status] || STATUS_TONE.pending;
  const board = run.board || {};
  const claims = board.claims || [];
  const verdicts = board.verdicts || [];
  const supported = verdicts.filter((v) => ["supported", "partly_supported"].includes(v.verdict));
  const critique = (board.critiques || []).slice(-1)[0];

  const tabs: { id: Tab; label: string; badge?: number }[] = [
    { id: "board", label: "Blackboard" },
    { id: "report", label: "Report" },
    { id: "messages", label: "Messages", badge: messages.length },
    { id: "steps", label: "Steps", badge: steps.length },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-3 flex-wrap">
        <div className="min-w-0 flex-1">
          <Link href="/" className="text-[11px] hover:underline" style={{ color: "var(--muted)" }}>
            &larr; All runs
          </Link>
          <h1 className="text-[17px] font-semibold leading-snug mt-0.5">{run.brief}</h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <span className="chip" style={{ color: tone.tone, borderColor: tone.tone }}>
              {tone.label}
              {live && run.phase ? `: ${run.phase}` : ""}
            </span>
            <span className="chip font-mono" style={{ color: "var(--faint)" }}>{run.run_key}</span>
            <span className="chip font-mono" style={{ color: "var(--faint)" }}>
              {run.provider}/{run.model}
            </span>
            {activeAgent && live && (
              <span className="chip" style={{ color: agentColour(activeAgent), borderColor: agentColour(activeAgent) }}>
                {activeAgent} working
              </span>
            )}
          </div>
          {run.pause_reason && (
            <p className="mt-1.5 text-[12px] leading-snug" style={{ color: "var(--warn)" }}>
              {run.pause_reason}
            </p>
          )}
          {run.error && (
            <p className="mt-1.5 text-[12px] leading-snug" style={{ color: "var(--error)" }}>
              {run.error}
            </p>
          )}
        </div>
      </div>

      {run.pending_approval && (
        <div className="panel p-3" style={{ borderColor: "var(--warn)" }}>
          <p className="text-[13px] font-medium mb-2">{run.pending_approval.question}</p>
          <div className="flex flex-wrap gap-1.5">
            {(run.pending_approval.options || []).map((option) => (
              <button
                key={option}
                className="btn"
                onClick={async () => {
                  await api.answer(run.run_key, { reply: option, resume: true });
                  loadRun();
                }}
              >
                {option}
              </button>
            ))}
            {!run.pending_approval.options?.length && (
              <div className="flex gap-1.5 w-full">
                <input
                  className="field text-[13px] py-1.5"
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  placeholder="Your answer"
                />
                <button
                  className="btn btn-primary"
                  onClick={async () => {
                    await api.answer(run.run_key, { reply: answer, resume: true });
                    setAnswer("");
                    loadRun();
                  }}
                >
                  Send
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid xl:grid-cols-[minmax(0,1fr)_320px] gap-4 items-start">
        <div className="space-y-4 min-w-0">
          <div className="grid lg:grid-cols-[auto_minmax(0,1fr)] gap-4 items-stretch">
            <div
              className="panel p-3 flex items-start justify-center overflow-y-auto scroll-thin"
              style={{ height: 460 }}
            >
              <AgentGraph
                graph={run.graph}
                states={nodeStates.states}
                counts={nodeStates.counts}
              />
            </div>
            {/* A definite height, not a minimum. The trace scrolls itself and
                follows the newest line; given only a minimum it grows instead,
                the inner scroller never scrolls, and the live activity ends up
                below the fold on a run that is still going. */}
            <div className="panel overflow-hidden" style={{ height: 460 }}>
              <Trace events={events} live={!!live} />
            </div>
          </div>

          <div className="panel overflow-hidden flex flex-col" style={{ height: 520 }}>
            <div className="flex gap-1 px-3 py-2 border-b" style={{ borderColor: "var(--line)" }}>
              {tabs.map((item) => (
                <button
                  key={item.id}
                  className="chip"
                  style={{
                    color: tab === item.id ? "var(--text)" : "var(--muted)",
                    borderColor: tab === item.id ? "var(--accent)" : "var(--line)",
                    background: tab === item.id ? "var(--accent-soft)" : "transparent",
                    cursor: "pointer",
                  }}
                  onClick={() => setTab(item.id)}
                >
                  {item.label}
                  {item.badge ? <span style={{ color: "var(--faint)" }}>{item.badge}</span> : null}
                </button>
              ))}
              {tab === "report" && report?.markdown && (
                <div className="ml-auto flex gap-1">
                  <a className="chip" href={api.reportUrl(run.run_key, "markdown")} download style={{ color: "var(--muted)" }}>
                    .md
                  </a>
                  <a className="chip" href={api.reportUrl(run.run_key, "html")} target="_blank" rel="noreferrer" style={{ color: "var(--muted)" }}>
                    .html
                  </a>
                </div>
              )}
            </div>

            <div className="flex-1 min-h-0">
              {tab === "board" && <BoardView board={board} evidence={evidence} />}

              {tab === "report" && (
                <div className="overflow-y-auto scroll-thin p-4 h-full">
                  {report?.markdown ? (
                    <Markdown text={report.markdown} />
                  ) : (
                    <p className="text-sm" style={{ color: "var(--faint)" }}>
                      Nothing written yet. The Writer runs once the Supervisor is satisfied
                      with the verified claims.
                    </p>
                  )}
                </div>
              )}

              {tab === "messages" && (
                <div className="overflow-y-auto scroll-thin p-3 h-full space-y-1.5 text-[12px]">
                  {messages.length === 0 && (
                    <p style={{ color: "var(--faint)" }}>No agent has addressed another yet.</p>
                  )}
                  {messages.map((m) => (
                    <div key={m.id} className="flex gap-2 items-baseline">
                      <span className="font-mono shrink-0" style={{ color: agentColour(m.sender) }}>
                        {m.sender}
                      </span>
                      <span style={{ color: "var(--faint)" }}>to</span>
                      <span className="font-mono shrink-0" style={{ color: agentColour(m.recipient) }}>
                        {m.recipient}
                      </span>
                      <span className="min-w-0">
                        {m.topic && <span style={{ color: "var(--muted)" }}>[{m.topic}] </span>}
                        {typeof m.content === "string"
                          ? m.content
                          : JSON.stringify(m.content)?.slice(0, 220)}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {tab === "steps" && (
                <div className="overflow-y-auto scroll-thin h-full">
                  <table className="w-full text-[12px]">
                    <thead className="sticky top-0" style={{ background: "var(--panel)" }}>
                      <tr style={{ color: "var(--muted)" }}>
                        <th className="text-left font-medium px-3 py-1.5">#</th>
                        <th className="text-left font-medium px-2">Node</th>
                        <th className="text-left font-medium px-2">Agent</th>
                        <th className="text-left font-medium px-2">Action</th>
                        <th className="text-right font-medium px-2">Tokens</th>
                        <th className="text-right font-medium px-3">Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {steps.map((step) => (
                        <tr key={step.seq} style={{ borderTop: "1px solid var(--line)" }}>
                          <td className="px-3 py-1.5 font-mono" style={{ color: "var(--faint)" }}>{step.seq}</td>
                          <td className="px-2 font-mono">{step.node}</td>
                          <td className="px-2" style={{ color: agentColour(step.agent) }}>{step.agent}</td>
                          <td className="px-2" style={{ color: step.status === "ok" ? "var(--muted)" : "var(--warn)" }}>
                            {step.action || step.status}
                          </td>
                          <td className="px-2 text-right font-mono" style={{ color: "var(--faint)" }}>
                            {formatTokens((step.tokens_in || 0) + (step.tokens_out || 0))}
                          </td>
                          <td className="px-3 text-right font-mono" style={{ color: "var(--faint)" }}>
                            {(step.duration_ms / 1000).toFixed(1)}s
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {steps.length === 0 && (
                    <p className="p-3 text-sm" style={{ color: "var(--faint)" }}>No steps yet.</p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        <aside className="space-y-3 xl:sticky xl:top-16">
          <div className="panel p-3">
            <Controls run={run} providers={config.providers} onChanged={loadRun} />
          </div>

          <div className="panel p-3">
            <BudgetMeter spend={run.spent} budget={run.budget} />
          </div>

          <div className="panel p-3 grid grid-cols-2 gap-2 text-[11px]">
            <Stat label="Sources" value={String(run.evidence_count)} />
            <Stat label="Findings" value={String((board.findings || []).length)} />
            <Stat label="Claims" value={String(claims.length)} />
            <Stat label="Verified" value={`${supported.length} of ${verdicts.length}`} />
            <Stat label="Model calls" value={String(run.spent.llm_calls ?? 0)} />
            <Stat label="Tool calls" value={String(run.spent.tool_calls ?? 0)} />
            <Stat label="Spent" value={formatUsd(run.spent.usd ?? 0)} />
            <Stat label="Elapsed" value={formatDuration(run.spent.seconds ?? 0)} />
          </div>

          {critique && (
            <div className="panel p-3">
              <h3 className="text-xs font-semibold uppercase tracking-wide mb-1.5" style={{ color: "var(--muted)" }}>
                Critic
              </h3>
              <div className="flex items-baseline gap-2">
                <span
                  className="text-xl font-semibold"
                  style={{ color: critique.score >= critique.bar ? "var(--ok)" : "var(--warn)" }}
                >
                  {critique.score}
                </span>
                <span className="text-[11px]" style={{ color: "var(--faint)" }}>
                  out of 100, pass mark {critique.bar}
                </span>
              </div>
              {critique.issues?.slice(0, 4).map((issue, i) => (
                <p key={i} className="mt-1.5 text-[11px] leading-snug" style={{ color: "var(--muted)" }}>
                  <span style={{ color: issue.severity === "critical" ? "var(--error)" : "var(--warn)" }}>
                    {issue.severity}:{" "}
                  </span>
                  {issue.issue}
                </p>
              ))}
            </div>
          )}

          <div className="panel overflow-hidden">
            <h3 className="text-xs font-semibold uppercase tracking-wide px-3 pt-3 pb-1" style={{ color: "var(--muted)" }}>
              The team
            </h3>
            <TeamPanel usage={run.usage_by_agent} tools={run.tool_usage} active={live ? activeAgent : ""} />
          </div>
        </aside>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div style={{ color: "var(--faint)" }}>{label}</div>
      <div className="font-mono text-[13px]" style={{ color: "var(--text)" }}>{value}</div>
    </div>
  );
}
