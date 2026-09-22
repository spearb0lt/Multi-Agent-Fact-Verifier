"use client";

/**
 * Who did what, and what each of them cost.
 *
 * This is the table to point at when someone asks whether the roles are real
 * or decorative. A system where one agent spends every token and the rest
 * produce a sentence each is a single prompt wearing eight hats, and that is
 * visible here immediately.
 */

import type { AgentUsage, ToolUsage } from "@/lib/api";
import { ROLE_BLURB, agentColour, formatTokens, formatUsd } from "@/lib/agents";

interface Props {
  usage: AgentUsage[];
  tools: ToolUsage[];
  active?: string;
}

export default function TeamPanel({ usage, tools, active }: Props) {
  const total = usage.reduce((sum, row) => sum + (row.tokens_in || 0) + (row.tokens_out || 0), 0);
  const byAgent = new Map<string, ToolUsage[]>();
  tools.forEach((row) => {
    if (!byAgent.has(row.agent)) byAgent.set(row.agent, []);
    byAgent.get(row.agent)!.push(row);
  });

  if (!usage.length) {
    return (
      <p className="text-xs px-3 py-4" style={{ color: "var(--faint)" }}>
        No agent has spent anything yet.
      </p>
    );
  }

  return (
    <div className="divide-y" style={{ borderColor: "var(--line)" }}>
      {usage.map((row) => {
        const tokens = (row.tokens_in || 0) + (row.tokens_out || 0);
        const share = total ? tokens / total : 0;
        const agentTools = byAgent.get(row.agent) || [];
        const isActive = active === row.agent;
        return (
          <div key={row.agent || "unknown"} className="px-3 py-2.5" style={{ borderColor: "var(--line)" }}>
            <div className="flex items-center gap-2">
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: agentColour(row.agent) }}
              />
              <span
                className="text-[13px] font-semibold"
                style={{ color: isActive ? "var(--accent)" : "var(--text)" }}
              >
                {row.agent || "unattributed"}
              </span>
              {isActive && (
                <span className="chip" style={{ color: "var(--accent)", borderColor: "var(--accent)" }}>
                  working
                </span>
              )}
              <span className="ml-auto font-mono text-[11px]" style={{ color: "var(--faint)" }}>
                {row.calls} calls · {formatTokens(tokens)} · {formatUsd(row.cost_usd || 0)}
              </span>
            </div>

            <div className="mt-1.5 h-[4px] rounded-full overflow-hidden" style={{ background: "var(--panel-2)" }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.max(share * 100, 1.5)}%`, background: agentColour(row.agent) }}
              />
            </div>

            <p className="mt-1.5 text-[11px] leading-snug" style={{ color: "var(--muted)" }}>
              {ROLE_BLURB[row.agent] || "Part of the run."}
            </p>

            {agentTools.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {agentTools.map((tool) => (
                  <span
                    key={tool.tool}
                    className="chip font-mono"
                    style={{
                      color: tool.failures ? "var(--warn)" : "var(--muted)",
                      borderColor: tool.failures ? "var(--warn)" : "var(--line)",
                    }}
                    title={tool.failures ? `${tool.failures} of ${tool.calls} failed` : undefined}
                  >
                    {tool.tool} {tool.calls}
                    {tool.failures ? ` (${tool.failures} failed)` : ""}
                  </span>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
