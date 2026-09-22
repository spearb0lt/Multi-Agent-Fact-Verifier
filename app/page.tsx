"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import NewRun from "./NewRun";
import AgentGraph from "@/components/AgentGraph";
import type { Config, Run } from "@/lib/api";
import { api } from "@/lib/api";
import { STATUS_TONE, formatDuration, formatTokens, formatUsd } from "@/lib/agents";

export default function Home() {
  const [config, setConfig] = useState<Config | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [cfg, list] = await Promise.all([api.config(), api.listRuns(25)]);
      setConfig(cfg);
      setRuns(list.runs);
      setError("");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    // A run started here keeps going in the background, so the list is polled
    // rather than left stale until the reader reloads the page themselves.
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, [refresh]);

  if (error && !config) {
    return (
      <div className="panel p-5 max-w-lg mx-auto mt-10">
        <h2 className="font-semibold mb-2">Cannot reach the API</h2>
        <p className="text-sm mb-3" style={{ color: "var(--muted)" }}>
          {error}
        </p>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Start it with <code>python -m mas.main</code> and reload.
        </p>
      </div>
    );
  }

  if (!config) {
    return <p className="py-16 text-center text-sm" style={{ color: "var(--muted)" }}>Loading...</p>;
  }

  const active = runs.filter((r) => ["running", "paused", "waiting", "pending"].includes(r.status));

  return (
    <div className="grid lg:grid-cols-[1fr_360px] gap-5 items-start">
      <div className="space-y-5">
        <div>
          <h1 className="text-xl font-semibold mb-0.5">{config.app}</h1>
          <p className="text-[13px]" style={{ color: "var(--muted)" }}>
            {config.tagline}
          </p>
        </div>

        <NewRun config={config} onStarted={refresh} />

        <section>
          <h2 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--muted)" }}>
            Runs {active.length > 0 && <span style={{ color: "var(--accent)" }}>· {active.length} active</span>}
          </h2>

          {runs.length === 0 && (
            <p className="panel p-4 text-sm" style={{ color: "var(--muted)" }}>
              No runs yet. Start one above and watch the team work.
            </p>
          )}

          <div className="space-y-1.5">
            {runs.map((run) => {
              const tone = STATUS_TONE[run.status] || STATUS_TONE.pending;
              return (
                <Link
                  key={run.run_key}
                  href={`/run?key=${run.run_key}`}
                  className="panel p-3 block hover:border-current transition-colors"
                >
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <p className="text-[13.5px] leading-snug line-clamp-2">{run.brief}</p>
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                        <span className="chip" style={{ color: tone.tone, borderColor: tone.tone }}>
                          {tone.label}
                          {run.status === "running" && run.phase ? `: ${run.phase}` : ""}
                        </span>
                        <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                          {run.provider}
                        </span>
                        <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                          {run.spent?.steps ?? 0} steps
                        </span>
                        <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                          {formatTokens(run.spent?.tokens ?? 0)}
                        </span>
                        <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                          {formatUsd(run.spent?.usd ?? 0)}
                        </span>
                        <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                          {formatDuration(run.spent?.seconds ?? 0)}
                        </span>
                      </div>
                      {run.pause_reason && (
                        <p className="mt-1 text-[11px]" style={{ color: "var(--warn)" }}>
                          {run.pause_reason.slice(0, 160)}
                        </p>
                      )}
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </section>
      </div>

      <aside className="space-y-4 lg:sticky lg:top-16">
        <div className="panel p-3">
          <h3 className="text-xs font-semibold uppercase tracking-wide mb-2" style={{ color: "var(--muted)" }}>
            The workflow
          </h3>
          {config.workflows[0] && (
            <AgentGraph graph={config.workflows[0]} states={{}} />
          )}
          <p className="text-[11px] mt-2 leading-snug" style={{ color: "var(--muted)" }}>
            {config.workflows[0]?.description}
          </p>
        </div>

        <div className="panel p-3 space-y-2 text-[11px]">
          <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--muted)" }}>
            This deployment
          </h3>
          <Row label="Providers" value={config.providers.filter((p) => p.available).map((p) => p.id).join(", ") || "none"} />
          <Row label="Search" value={config.search.backends.join(", ") || "none"} />
          <Row label="Embeddings" value={config.embedding.available ? config.embedding.id : "unavailable"} />
          <Row label="Tools" value={String(config.tools.length)} />
          <Row label="Runtime" value={`${config.runtime.tier}${config.runtime.persistent_disk ? ", durable disk" : ", ephemeral disk"}`} />
          <Row label="Worker" value={`${config.worker.active} of ${config.worker.capacity} busy`} />
        </div>
      </aside>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3">
      <span style={{ color: "var(--faint)" }}>{label}</span>
      <span className="font-mono text-right break-all" style={{ color: "var(--muted)" }}>
        {value}
      </span>
    </div>
  );
}
