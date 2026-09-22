"use client";

/**
 * What the run has spent against what it is allowed.
 *
 * Every ceiling is shown, not just the one being approached, because the
 * useful question when a run pauses is "which one stopped it" and the answer
 * has to be visible without reading the pause message. The bar that is closest
 * to full is the one that will stop the run, so it is the one highlighted.
 */

import type { Budget, Spend } from "@/lib/api";
import { formatDuration, formatTokens, formatUsd } from "@/lib/agents";

interface Props {
  spend: Spend;
  budget: Budget;
  compact?: boolean;
}

interface Row {
  name: string;
  used: number;
  limit: number;
  render: (value: number) => string;
}

export function readings(spend: Spend, budget: Budget): Row[] {
  const plain = (v: number) => String(Math.round(v));
  return [
    { name: "steps", used: spend.steps ?? 0, limit: budget.max_steps ?? 0, render: plain },
    { name: "tokens", used: spend.tokens ?? 0, limit: budget.max_tokens ?? 0, render: formatTokens },
    { name: "cost", used: spend.usd ?? 0, limit: budget.max_usd ?? 0, render: formatUsd },
    { name: "time", used: spend.seconds ?? 0, limit: budget.max_seconds ?? 0, render: formatDuration },
    { name: "tool calls", used: spend.tool_calls ?? 0, limit: budget.max_tool_calls ?? 0, render: plain },
  ];
}

export function pressureOf(spend: Spend, budget: Budget): number {
  return Math.max(
    0,
    ...readings(spend, budget).map((r) => (r.limit > 0 ? r.used / r.limit : 0)),
  );
}

export default function BudgetMeter({ spend, budget, compact }: Props) {
  const rows = readings(spend, budget);
  const pressure = pressureOf(spend, budget);
  const tightest = rows.reduce(
    (worst, row) => {
      const f = row.limit > 0 ? row.used / row.limit : 0;
      return f > worst.fraction ? { name: row.name, fraction: f } : worst;
    },
    { name: "", fraction: 0 },
  );

  return (
    <div className={compact ? "space-y-1.5" : "space-y-2.5"}>
      {!compact && (
        <div className="flex items-baseline justify-between">
          <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
            Budget
          </h3>
          <span className="text-xs" style={{ color: pressure >= 0.7 ? "var(--warn)" : "var(--faint)" }}>
            {Math.round(pressure * 100)}% of the tightest ceiling
          </span>
        </div>
      )}

      {rows.map((row) => {
        const unlimited = !row.limit;
        const fraction = unlimited ? 0 : Math.min(1, row.used / row.limit);
        const isTightest = row.name === tightest.name && tightest.fraction > 0.01;
        const colour =
          fraction >= 0.88 ? "var(--error)" : fraction >= 0.7 ? "var(--warn)" : "var(--accent)";
        return (
          <div key={row.name}>
            <div className="flex justify-between text-[11px] mb-0.5">
              <span style={{ color: isTightest ? "var(--text)" : "var(--muted)", fontWeight: isTightest ? 600 : 400 }}>
                {row.name}
              </span>
              <span style={{ color: "var(--faint)" }} className="font-mono">
                {row.render(row.used)}
                {unlimited ? " / no limit" : ` / ${row.render(row.limit)}`}
              </span>
            </div>
            <div className="h-[5px] rounded-full overflow-hidden" style={{ background: "var(--panel-2)" }}>
              <div
                className="h-full rounded-full transition-[width] duration-500"
                style={{ width: `${Math.max(fraction * 100, row.used > 0 ? 2 : 0)}%`, background: colour }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
