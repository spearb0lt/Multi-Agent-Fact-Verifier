"use client";

/**
 * Who is on the team, what each of them may do, and how the work flows.
 *
 * Built entirely from what the server reports, so it cannot drift from the
 * system it describes. If a role were removed from the workflow it would
 * disappear from this page without anyone remembering to edit it.
 */

import { useEffect, useState } from "react";
import AgentGraph from "@/components/AgentGraph";
import type { Config } from "@/lib/api";
import { api } from "@/lib/api";
import { ROLE_BLURB, agentColour } from "@/lib/agents";

export default function TeamPage() {
  const [config, setConfig] = useState<Config | null>(null);

  useEffect(() => {
    api.config().then(setConfig).catch(() => undefined);
  }, []);

  if (!config) {
    return <p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>Loading...</p>;
  }

  const workflow = config.workflows[0];
  const tierOf = (agent: string) =>
    ["Researcher", "FactChecker", "Supervisor"].includes(agent) ? "cheap" : "strong";

  return (
    <div className="space-y-5 max-w-4xl">
      <div>
        <h1 className="text-xl font-semibold mb-0.5">The team</h1>
        <p className="text-[13px]" style={{ color: "var(--muted)" }}>
          {workflow?.description}
        </p>
      </div>

      <div className="panel p-4">
        <AgentGraph graph={workflow} states={{}} />
      </div>

      <section className="space-y-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
          Roles
        </h2>
        {workflow?.nodes.map((node) => (
          <div key={node.name} className="panel p-3">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="w-2 h-2 rounded-full" style={{ background: agentColour(node.agent) }} />
              <span className="font-semibold text-[14px]">{node.agent}</span>
              <span className="chip font-mono" style={{ color: "var(--faint)" }}>{node.name}</span>
              <span className="chip" style={{ color: "var(--muted)" }}>
                {tierOf(node.agent)} model
              </span>
              {node.parallel && (
                <span className="chip" style={{ color: "var(--accent)" }}>runs in parallel</span>
              )}
            </div>
            <p className="mt-1 text-[12.5px]" style={{ color: "var(--muted)" }}>
              {ROLE_BLURB[node.agent] || node.description}
            </p>
          </div>
        ))}
      </section>

      <section className="space-y-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
          Tools the agents can choose from
        </h2>
        <div className="grid sm:grid-cols-2 gap-2">
          {config.tools.map((tool) => (
            <div key={tool.name} className="panel p-3">
              <p className="font-mono text-[13px] font-medium">{tool.name}</p>
              <p className="mt-0.5 text-[12px] leading-snug" style={{ color: "var(--muted)" }}>
                {tool.description}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
