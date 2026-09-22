"use client";

/**
 * Starting a run, with the ceilings visible before anything is spent.
 *
 * The budget controls are on this form rather than hidden in settings on
 * purpose. The person starting a run is the person who cares what it costs,
 * and a ceiling chosen after the fact is a ceiling that did not apply.
 */

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { Config } from "@/lib/api";
import { api } from "@/lib/api";

const CLAIM_EXAMPLES = [
  "Drinking three cups of coffee a day reduces the risk of heart disease.",
  "India overtook Japan to become the world's fourth largest economy in 2025.",
  "Intermittent fasting preserves muscle mass better than continuous calorie restriction.",
];

const EXAMPLES = [
  "What did the RBI decide at its most recent monetary policy meeting, and why?",
  "What is the current evidence on semaglutide for weight loss in non diabetic adults?",
  "How are Indian IT services companies positioning on generative AI, and what have they actually shipped?",
];

interface Props {
  config: Config;
  onStarted?: () => void;
}

export default function NewRun({ config, onStarted }: Props) {
  const router = useRouter();
  const [brief, setBrief] = useState("");
  const [workflow, setWorkflow] = useState(config.workflows[0]?.name || "research_report");
  const [provider, setProvider] = useState(config.default.provider || "");
  const [model, setModel] = useState("");
  const [depth, setDepth] = useState<"quick" | "standard" | "deep">("standard");
  const [advanced, setAdvanced] = useState(false);
  const [maxUsd, setMaxUsd] = useState(config.defaults.max_usd);
  const [maxSteps, setMaxSteps] = useState(config.defaults.max_steps);
  const [maxTokens, setMaxTokens] = useState(config.defaults.max_tokens);
  const [maxSeconds, setMaxSeconds] = useState(config.defaults.max_seconds);
  const [rounds, setRounds] = useState(config.defaults.max_research_rounds);
  const [revisions, setRevisions] = useState(config.defaults.max_revisions);
  const [concurrency, setConcurrency] = useState(config.defaults.agent_concurrency);
  const [approvePlan, setApprovePlan] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const available = config.providers.filter((p) => p.available);
  const chosen = available.find((p) => p.id === provider) || available[0];

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (brief.trim().length < 8) {
      setError("Write a brief of at least a few words.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const created = await api.createRun({
        brief: brief.trim(),
        workflow,
        provider: chosen?.id || "",
        model: model || undefined,
        depth,
        budget: {
          max_usd: maxUsd,
          max_steps: maxSteps,
          max_tokens: maxTokens,
          max_seconds: maxSeconds,
        },
        max_research_rounds: rounds,
        max_revisions: revisions,
        agent_concurrency: concurrency,
        approve_plan: approvePlan,
        start: true,
      });
      onStarted?.();
      router.push(`/run?key=${created.run_key}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  if (!available.length) {
    return (
      <div className="panel p-4">
        <h2 className="font-semibold text-[15px] mb-1">No model provider is available</h2>
        <p className="text-sm mb-3" style={{ color: "var(--muted)" }}>
          Set a key in <code>.env</code>, or paste one in Settings. Gemini, Groq and
          Cloudflare all have a free tier, and Ollama needs no key at all.
        </p>
        <a href="/settings" className="btn">
          Open settings
        </a>
      </div>
    );
  }

  const chosenWorkflow = config.workflows.find((w) => w.name === workflow) || config.workflows[0];
  const isClaimCheck = workflow === "claim_check";
  const examples = isClaimCheck ? CLAIM_EXAMPLES : EXAMPLES;

  return (
    <form onSubmit={submit} className="panel p-4 space-y-3">
      {config.workflows.length > 1 && (
        <div className="flex flex-wrap gap-1.5">
          {config.workflows.map((w) => (
            <button
              key={w.name}
              type="button"
              className="chip"
              style={{
                cursor: "pointer",
                color: w.name === workflow ? "var(--text)" : "var(--muted)",
                borderColor: w.name === workflow ? "var(--accent)" : "var(--line)",
                background: w.name === workflow ? "var(--accent-soft)" : "transparent",
              }}
              onClick={() => setWorkflow(w.name)}
            >
              {w.name === "claim_check" ? "Check one claim" : "Research a topic"}
              <span style={{ color: "var(--faint)" }}>{w.nodes.length} agents</span>
            </button>
          ))}
        </div>
      )}

      <div>
        <label className="block text-[13px] font-medium mb-1">
          {isClaimCheck
            ? "What claim should the team check?"
            : "What should the team research?"}
        </label>
        <textarea
          className="field font-sans"
          rows={3}
          placeholder={
            isClaimCheck
              ? "State the claim as a single sentence, the way someone asserted it."
              : "Ask for something specific and checkable."
          }
          value={brief}
          onChange={(e) => setBrief(e.target.value)}
        />
        {chosenWorkflow && (
          <p className="mt-1 text-[11px] leading-snug" style={{ color: "var(--faint)" }}>
            {chosenWorkflow.description}
          </p>
        )}
        <div className="mt-1.5 flex flex-wrap gap-1">
          {examples.map((example) => (
            <button
              key={example}
              type="button"
              className="chip text-left"
              style={{ color: "var(--muted)", cursor: "pointer" }}
              onClick={() => setBrief(example)}
            >
              {example.slice(0, 54)}...
            </button>
          ))}
        </div>
      </div>

      <div className="grid sm:grid-cols-3 gap-2">
        <label className="text-[11px]" style={{ color: "var(--muted)" }}>
          Provider
          <select
            className="field mt-0.5 text-[13px] py-1.5"
            value={chosen?.id}
            onChange={(e) => {
              setProvider(e.target.value);
              setModel("");
            }}
          >
            {available.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </label>

        <label className="text-[11px]" style={{ color: "var(--muted)" }}>
          Model
          <select
            className="field mt-0.5 text-[13px] py-1.5"
            value={model || chosen?.default_model}
            onChange={(e) => setModel(e.target.value)}
          >
            {(chosen?.models || []).map((m) => (
              <option key={m.id} value={m.id}>
                {m.id}
                {m.cheap ? "  (cheap)" : ""}
              </option>
            ))}
          </select>
        </label>

        <label className="text-[11px]" style={{ color: "var(--muted)" }}>
          Depth
          <select
            className="field mt-0.5 text-[13px] py-1.5"
            value={depth}
            onChange={(e) => setDepth(e.target.value as typeof depth)}
          >
            <option value="quick">Quick, 3 questions</option>
            <option value="standard">Standard, 5 questions</option>
            <option value="deep">Deep, 8 questions</option>
          </select>
        </label>
      </div>

      <div className="flex items-center gap-3 text-[11px] flex-wrap" style={{ color: "var(--muted)" }}>
        <button type="button" className="chip" style={{ cursor: "pointer" }} onClick={() => setAdvanced((v) => !v)}>
          {advanced ? "Hide limits" : "Limits and loops"}
        </button>
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={approvePlan}
            onChange={(e) => setApprovePlan(e.target.checked)}
          />
          Show me the plan before spending on research
        </label>
        <span>
          Stops at {maxSteps} steps, {Math.round(maxTokens / 1000)}k tokens, ${maxUsd}, or{" "}
          {Math.round(maxSeconds / 60)} minutes. A stop pauses and keeps everything.
        </span>
      </div>

      {advanced && (
        <div className="grid sm:grid-cols-4 gap-2 pt-1">
          {[
            { label: "Max steps", value: maxSteps, set: setMaxSteps, step: 10 },
            { label: "Max tokens", value: maxTokens, set: setMaxTokens, step: 50000 },
            { label: "Max cost, USD", value: maxUsd, set: setMaxUsd, step: 0.05 },
            { label: "Max seconds", value: maxSeconds, set: setMaxSeconds, step: 60 },
            { label: "Research rounds", value: rounds, set: setRounds, step: 1 },
            { label: "Revisions", value: revisions, set: setRevisions, step: 1 },
            { label: "Agents at once", value: concurrency, set: setConcurrency, step: 1 },
          ].map((field) => (
            <label key={field.label} className="text-[11px]" style={{ color: "var(--muted)" }}>
              {field.label}
              <input
                className="field mt-0.5 text-[13px] py-1.5"
                type="number"
                min={0}
                step={field.step}
                value={field.value}
                onChange={(e) => field.set(Number(e.target.value))}
              />
            </label>
          ))}
        </div>
      )}

      {error && (
        <p className="text-[12px]" style={{ color: "var(--error)" }}>
          {error}
        </p>
      )}

      <div className="flex items-center gap-2">
        <button className="btn btn-primary" type="submit" disabled={busy}>
          {busy ? "Starting..." : "Start the run"}
        </button>
        <span className="text-[11px]" style={{ color: "var(--faint)" }}>
          {config.search.usable
            ? `Research via ${config.search.backends.join(", ")}`
            : "No search backend configured"}
        </span>
      </div>
    </form>
  );
}
