"use client";

/**
 * The live trace: what each agent did, in order, as it happens.
 *
 * This is the view that answers "is anything actually happening", so it is
 * optimised for being watched rather than read. Events are colour coded by the
 * role that produced them, tool calls are indented under the agent that made
 * them, and the list sticks to the bottom unless the reader has scrolled up,
 * which is the one interaction that matters in a log that is still growing.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { TraceEvent } from "@/lib/api";
import { agentColour } from "@/lib/agents";

interface Props {
  events: TraceEvent[];
  live: boolean;
}

const LABEL: Record<string, string> = {
  "run.started": "run started",
  "run.resumed": "run resumed",
  "run.paused": "run paused",
  "run.completed": "run completed",
  "run.failed": "run failed",
  "run.cancelled": "run cancelled",
  "node.enter": "started",
  "node.exit": "finished",
  "node.error": "failed",
  "node.retry": "retrying",
  "agent.start": "thinking",
  "agent.thought": "thought",
  "agent.finish": "done",
  "agent.message": "message",
  "tool.call": "calls",
  "tool.result": "result",
  "tool.error": "tool failed",
  "evidence.added": "source",
  "artifact.created": "produced",
  "budget.warn": "budget",
  "budget.breach": "budget",
  "model.downgrade": "model",
  "approval.requested": "needs you",
  "approval.answered": "answered",
  log: "note",
};

/** Events worth showing when the reader has not asked for everything. */
const IMPORTANT = new Set([
  "run.started", "run.resumed", "run.paused", "run.completed", "run.failed",
  "run.cancelled", "node.enter", "node.error", "node.retry", "agent.start",
  "agent.finish", "agent.message", "tool.call", "evidence.added",
  "artifact.created", "budget.warn", "budget.breach", "model.downgrade",
  "approval.requested", "approval.answered",
]);

const INDENTED = new Set(["tool.call", "tool.result", "tool.error", "agent.thought", "agent.observation"]);

function tone(event: TraceEvent): string {
  if (event.level === "error") return "var(--error)";
  if (event.level === "warning") return "var(--warn)";
  if (event.kind.startsWith("run.")) return "var(--accent)";
  return "var(--muted)";
}

export default function Trace({ events, live }: Props) {
  const [verbose, setVerbose] = useState(false);
  const [filter, setFilter] = useState("");
  const boxRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  const shown = useMemo(() => {
    const term = filter.trim().toLowerCase();
    return events.filter((event) => {
      if (!verbose && !IMPORTANT.has(event.kind)) return false;
      if (!term) return true;
      return (
        event.message.toLowerCase().includes(term) ||
        event.agent.toLowerCase().includes(term) ||
        event.kind.toLowerCase().includes(term)
      );
    });
  }, [events, verbose, filter]);

  useEffect(() => {
    const box = boxRef.current;
    if (!box || !pinned.current) return;
    box.scrollTop = box.scrollHeight;
  }, [shown.length]);

  const onScroll = () => {
    const box = boxRef.current;
    if (!box) return;
    // Treat "close enough to the bottom" as pinned, so a stray wheel event
    // does not permanently detach the view from a run that is still going.
    pinned.current = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: "var(--line)" }}>
        <input
          className="field text-xs py-1"
          placeholder="Filter the trace"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          style={{ maxWidth: 200 }}
        />
        <label className="flex items-center gap-1.5 text-xs cursor-pointer" style={{ color: "var(--muted)" }}>
          <input type="checkbox" checked={verbose} onChange={(e) => setVerbose(e.target.checked)} />
          thoughts and results
        </label>
        <span className="ml-auto text-xs" style={{ color: "var(--faint)" }}>
          {shown.length} of {events.length}
          {live && <span style={{ color: "var(--accent)" }}> · live</span>}
        </span>
      </div>

      <div ref={boxRef} onScroll={onScroll} className="flex-1 overflow-y-auto scroll-thin px-3 py-2 font-mono text-xs min-h-0">
        {shown.length === 0 && (
          <p className="py-6 text-center" style={{ color: "var(--faint)" }}>
            {events.length ? "Nothing matches that filter." : "Waiting for the first step."}
          </p>
        )}
        {shown.map((event) => {
          const colour = event.agent ? agentColour(event.agent) : "var(--faint)";
          const indent = INDENTED.has(event.kind);
          return (
            <div
              key={event.id}
              className="flex gap-2 py-[3px] leading-relaxed"
              style={{ paddingLeft: indent ? 22 : 0 }}
            >
              <span
                className="shrink-0 w-[92px] truncate"
                style={{ color: colour, fontWeight: event.agent ? 600 : 400 }}
                title={event.agent}
              >
                {event.agent || (event.node ? event.node : "run")}
              </span>
              <span className="shrink-0 w-[62px]" style={{ color: tone(event) }}>
                {LABEL[event.kind] || event.kind}
              </span>
              <span className="break-words" style={{ color: event.level === "error" ? "var(--error)" : "var(--text)" }}>
                {event.message}
                {event.kind === "tool.call" && event.payload?.arguments ? (
                  <span style={{ color: "var(--faint)" }}>
                    {" "}
                    {JSON.stringify(event.payload.arguments).slice(0, 160)}
                  </span>
                ) : null}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
