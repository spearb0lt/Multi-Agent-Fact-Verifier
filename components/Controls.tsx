"use client";

/**
 * Start, stop, continue.
 *
 * The controls are the reason the whole run loop is built the way it is, so
 * they are deliberately plain: one obvious button for the state the run is
 * actually in, and the destructive one kept apart and behind a confirmation.
 *
 * Resuming offers a raised ceiling and a different provider in the same place,
 * because the two most common reasons a run stops are that it hit its budget
 * and that its provider ran out of free quota, and both are fixed here.
 */

import { useState } from "react";
import type { Budget, ProviderStatus, RunDetail } from "@/lib/api";
import { api } from "@/lib/api";

interface Props {
  run: RunDetail;
  providers: ProviderStatus[];
  onChanged: () => void;
}

export default function Controls({ run, providers, onChanged }: Props) {
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [showResume, setShowResume] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [budget, setBudget] = useState<Partial<Budget>>({});
  const [provider, setProvider] = useState("");

  const status = run.status;
  const running = status === "running";
  const stopped = status === "paused" || status === "pending";
  const waiting = status === "waiting";
  const finished = ["completed", "failed", "cancelled"].includes(status);

  const act = async (name: string, fn: () => Promise<unknown>) => {
    setBusy(name);
    setError("");
    try {
      await fn();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy("");
    }
  };

  const resume = () =>
    act("resume", async () => {
      const body: Record<string, unknown> = {};
      if (Object.keys(budget).length) body.budget = budget;
      if (provider && provider !== run.provider) body.provider = provider;
      await api.resume(run.run_key, body);
      setShowResume(false);
    });

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1.5">
        {running && (
          <button
            className="btn"
            disabled={!!busy}
            onClick={() => act("pause", () => api.pause(run.run_key))}
          >
            {busy === "pause" ? "Pausing..." : "Pause"}
          </button>
        )}

        {stopped && (
          <>
            <button
              className="btn btn-primary"
              disabled={!!busy}
              onClick={() => act("resume", () => api.resume(run.run_key, {}))}
            >
              {busy === "resume" ? "Resuming..." : status === "pending" ? "Start" : "Resume"}
            </button>
            <button className="btn" onClick={() => setShowResume((v) => !v)}>
              Resume with changes
            </button>
          </>
        )}

        {waiting && (
          <span className="chip" style={{ color: "var(--warn)", borderColor: "var(--warn)" }}>
            Waiting for your answer below
          </span>
        )}

        {!finished && (
          <button
            className="btn"
            style={{ color: "var(--error)", borderColor: confirmCancel ? "var(--error)" : "var(--line)" }}
            disabled={!!busy}
            onClick={() => {
              if (!confirmCancel) {
                setConfirmCancel(true);
                return;
              }
              act("cancel", () => api.cancel(run.run_key));
            }}
            onBlur={() => setConfirmCancel(false)}
          >
            {confirmCancel ? "Really cancel?" : "Cancel"}
          </button>
        )}

        {finished && (
          <span className="text-xs" style={{ color: "var(--faint)" }}>
            This run has finished. Its record is kept.
          </span>
        )}
      </div>

      {showResume && stopped && (
        <div className="panel p-3 space-y-2">
          <p className="text-[11px]" style={{ color: "var(--muted)" }}>
            Raise a ceiling, or continue on a provider that still has quota. Everything
            already gathered is kept either way.
          </p>
          <div className="grid grid-cols-2 gap-2">
            <label className="text-[11px]" style={{ color: "var(--muted)" }}>
              Max steps
              <input
                className="field mt-0.5 text-xs py-1"
                type="number"
                min={0}
                defaultValue={run.budget.max_steps}
                onChange={(e) => setBudget((b) => ({ ...b, max_steps: Number(e.target.value) }))}
              />
            </label>
            <label className="text-[11px]" style={{ color: "var(--muted)" }}>
              Max cost, USD
              <input
                className="field mt-0.5 text-xs py-1"
                type="number"
                step="0.05"
                min={0}
                defaultValue={run.budget.max_usd}
                onChange={(e) => setBudget((b) => ({ ...b, max_usd: Number(e.target.value) }))}
              />
            </label>
            <label className="text-[11px]" style={{ color: "var(--muted)" }}>
              Max tokens
              <input
                className="field mt-0.5 text-xs py-1"
                type="number"
                min={0}
                defaultValue={run.budget.max_tokens}
                onChange={(e) => setBudget((b) => ({ ...b, max_tokens: Number(e.target.value) }))}
              />
            </label>
            <label className="text-[11px]" style={{ color: "var(--muted)" }}>
              Max seconds
              <input
                className="field mt-0.5 text-xs py-1"
                type="number"
                min={0}
                defaultValue={run.budget.max_seconds}
                onChange={(e) => setBudget((b) => ({ ...b, max_seconds: Number(e.target.value) }))}
              />
            </label>
          </div>
          <label className="text-[11px] block" style={{ color: "var(--muted)" }}>
            Provider
            <select
              className="field mt-0.5 text-xs py-1"
              value={provider || run.provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              {providers
                .filter((p) => p.available)
                .map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
            </select>
          </label>
          <button className="btn btn-primary w-full" disabled={!!busy} onClick={resume}>
            {busy === "resume" ? "Resuming..." : "Resume"}
          </button>
        </div>
      )}

      {error && (
        <p className="text-[11px]" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}
    </div>
  );
}
