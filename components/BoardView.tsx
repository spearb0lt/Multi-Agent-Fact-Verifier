"use client";

/**
 * The blackboard: the plan, what was found, what was claimed and what survived.
 *
 * The verification column is the one that matters. A claim shown next to the
 * verdict it received, the domains behind it and the reason the Fact Checker
 * gave is the difference between a system that says it verifies and one whose
 * verification can be inspected.
 */

import { useState } from "react";
import type { Board, Evidence } from "@/lib/api";
import { agentColour } from "@/lib/agents";

interface Props {
  board: Board;
  evidence: Evidence[];
}

const VERDICT_TONE: Record<string, { colour: string; label: string }> = {
  supported: { colour: "var(--ok)", label: "supported" },
  partly_supported: { colour: "var(--warn)", label: "partly supported" },
  unsupported: { colour: "var(--error)", label: "unsupported" },
  contradicted: { colour: "var(--error)", label: "contradicted" },
};

type Tab = "plan" | "findings" | "claims" | "sources";

export default function BoardView({ board, evidence }: Props) {
  const [tab, setTab] = useState<Tab>("claims");

  const plan = board.plan || {};
  const subquestions = board.subquestions || plan.subquestions || [];
  const findings = board.findings || [];
  const claims = board.claims || [];
  const verdicts = board.verdicts || [];
  const decisions = board.decisions || [];
  const verdictFor = (id: string) => verdicts.find((v) => v.claim_id === id);
  const byRef = new Map(evidence.map((e) => [e.ref, e]));

  const conflicts = board.conflicts || [];
  const ruling = board.ruling;

  const tabs: { id: Tab; label: string; count: number }[] = [
    { id: "plan", label: "Plan", count: subquestions.length },
    { id: "findings", label: "Findings", count: findings.length },
    { id: "claims", label: "Claims", count: claims.length },
    { id: "sources", label: "Sources", count: evidence.length },
  ];

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex gap-1 px-3 py-2 border-b" style={{ borderColor: "var(--line)" }}>
        {tabs.map((item) => (
          <button
            key={item.id}
            onClick={() => setTab(item.id)}
            className="chip"
            style={{
              color: tab === item.id ? "var(--text)" : "var(--muted)",
              borderColor: tab === item.id ? "var(--accent)" : "var(--line)",
              background: tab === item.id ? "var(--accent-soft)" : "transparent",
              cursor: "pointer",
            }}
          >
            {item.label} {item.count > 0 && <span style={{ color: "var(--faint)" }}>{item.count}</span>}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto scroll-thin px-3 py-3 min-h-0 text-sm">
        {tab === "plan" && (
          <div className="space-y-3">
            {plan.angle && (
              <p className="text-[13px] leading-relaxed" style={{ color: "var(--text)" }}>
                <span style={{ color: "var(--muted)" }}>Angle: </span>
                {plan.angle}
              </p>
            )}
            {subquestions.length === 0 && (
              <p style={{ color: "var(--faint)" }}>The Planner has not run yet.</p>
            )}
            {subquestions.map((q) => (
              <div key={q.id} className="panel p-2.5">
                <div className="flex gap-2 items-start">
                  <span className="chip font-mono shrink-0" style={{ color: agentColour("Planner") }}>
                    {q.id}
                  </span>
                  <div>
                    <p className="text-[13px] leading-snug">{q.question}</p>
                    {q.why && (
                      <p className="mt-1 text-[11px]" style={{ color: "var(--muted)" }}>
                        {q.why}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
            {decisions.length > 0 && (
              <div className="pt-2">
                <h4 className="text-[11px] uppercase tracking-wide mb-1.5" style={{ color: "var(--muted)" }}>
                  Supervisor decisions
                </h4>
                {decisions.map((d, i) => (
                  <div key={i} className="panel p-2.5 mb-1.5">
                    <span
                      className="chip"
                      style={{ color: d.decision === "write" ? "var(--ok)" : "var(--warn)" }}
                    >
                      {d.decision === "write" ? "write the report" : "research more"}
                      {d.forced && " (forced)"}
                    </span>
                    <p className="mt-1 text-[12px]" style={{ color: "var(--muted)" }}>
                      {d.reasoning}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {tab === "findings" && (
          <div className="space-y-2">
            {findings.length === 0 && (
              <p style={{ color: "var(--faint)" }}>No researcher has recorded a finding yet.</p>
            )}
            {findings.map((f) => (
              <div key={f.id} className="panel p-2.5">
                <p className="text-[13px] leading-snug">{f.statement}</p>
                <div className="mt-1.5 flex flex-wrap gap-1 items-center">
                  <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                    {f.id}
                  </span>
                  {f.sources.map((ref) => (
                    <span
                      key={ref}
                      className="chip font-mono"
                      style={{ color: "var(--accent)" }}
                      title={byRef.get(ref)?.title || ref}
                    >
                      {ref}
                    </span>
                  ))}
                  <span className="chip" style={{ color: "var(--muted)" }}>
                    {f.confidence}
                  </span>
                  {f.subquestion && (
                    <span className="chip" style={{ color: "var(--faint)" }}>
                      {f.subquestion}
                    </span>
                  )}
                </div>
                {f.note && (
                  <p className="mt-1 text-[11px]" style={{ color: "var(--muted)" }}>
                    {f.note}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}

        {tab === "claims" && (
          <div className="space-y-2">
            {ruling && (
              <div className="panel p-3" style={{ borderLeft: "3px solid var(--accent)" }}>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[15px] font-semibold">
                    {String(ruling.verdict || "").replace(/_/g, " ")}
                  </span>
                  <span
                    className="chip"
                    style={{ color: "var(--muted)" }}
                    title="Confidence in this ruling, not in the claim being true"
                  >
                    {ruling.confidence} confidence in this ruling
                  </span>
                  <span
                    className="chip"
                    style={{ color: ruling.corroborated ? "var(--ok)" : "var(--warn)" }}
                  >
                    {ruling.independent_domains} independent outlet
                    {ruling.independent_domains === 1 ? "" : "s"}
                  </span>
                </div>
                {ruling.reasoning && (
                  <p className="mt-1.5 text-[12.5px] leading-snug">{ruling.reasoning}</p>
                )}
                {ruling.what_would_settle_it && (
                  <p className="mt-1.5 text-[11px]" style={{ color: "var(--muted)" }}>
                    What would settle it: {ruling.what_would_settle_it}
                  </p>
                )}
              </div>
            )}

            {conflicts.length > 0 && (
              <div className="panel p-3" style={{ borderLeft: "3px solid var(--warn)" }}>
                <h4 className="text-[11px] uppercase tracking-wide mb-1.5" style={{ color: "var(--warn)" }}>
                  Where the sources disagree
                </h4>
                {conflicts.map((c, i) => (
                  <div key={i} className="mb-2 last:mb-0">
                    <span className="chip" style={{ color: "var(--warn)", borderColor: "var(--warn)" }}>
                      {c.relation}
                    </span>
                    <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                      {c.a} vs {c.b}
                    </span>
                    <p className="mt-1 text-[12px] leading-snug">{c.explanation}</p>
                    <p className="mt-0.5 text-[11px]" style={{ color: "var(--muted)" }}>
                      {c.better_supported === "neither"
                        ? "Neither is better supported."
                        : `Better supported: ${c.better_supported === "A" ? c.a : c.b}. ${c.why}`}
                    </p>
                  </div>
                ))}
              </div>
            )}

            {claims.length === 0 && !ruling && (
              <p style={{ color: "var(--faint)" }}>
                The Analyst has not consolidated the findings into claims yet.
              </p>
            )}
            {claims.map((claim) => {
              const verdict = verdictFor(claim.id);
              const tone = verdict ? VERDICT_TONE[verdict.verdict] : null;
              return (
                <div
                  key={claim.id}
                  className="panel p-2.5"
                  style={{ borderLeft: `3px solid ${tone ? tone.colour : "var(--line)"}` }}
                >
                  <p className="text-[13px] leading-snug">{claim.text}</p>

                  <div className="mt-1.5 flex flex-wrap gap-1 items-center">
                    <span className="chip font-mono" style={{ color: "var(--faint)" }}>
                      {claim.id}
                    </span>
                    {claim.sources.map((ref) => (
                      <span
                        key={ref}
                        className="chip font-mono"
                        style={{ color: "var(--accent)" }}
                        title={byRef.get(ref)?.title || ref}
                      >
                        {ref}
                      </span>
                    ))}
                    {tone ? (
                      <span className="chip" style={{ color: tone.colour, borderColor: tone.colour }}>
                        {tone.label}
                      </span>
                    ) : (
                      <span className="chip" style={{ color: "var(--faint)" }}>
                        not yet checked
                      </span>
                    )}
                    {verdict && (
                      <span
                        className="chip"
                        style={{ color: verdict.corroborated ? "var(--ok)" : "var(--warn)" }}
                        title="Independent domains behind this claim"
                      >
                        {verdict.independent_domains} domain
                        {verdict.independent_domains === 1 ? "" : "s"}
                      </span>
                    )}
                    {claim.contested && (
                      <span className="chip" style={{ color: "var(--warn)" }}>
                        contested
                      </span>
                    )}
                  </div>

                  {verdict?.reasoning && (
                    <p className="mt-1.5 text-[11px] leading-snug" style={{ color: "var(--muted)" }}>
                      <span style={{ color: agentColour("FactChecker") }}>Fact Checker: </span>
                      {verdict.reasoning}
                    </p>
                  )}
                  {verdict?.correction && (
                    <p className="mt-1 text-[11px]" style={{ color: "var(--warn)" }}>
                      Correction: {verdict.correction}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {tab === "sources" && (
          <div className="space-y-2">
            {evidence.length === 0 && (
              <p style={{ color: "var(--faint)" }}>Nothing has been fetched yet.</p>
            )}
            {evidence.map((item) => (
              <div key={item.ref} className="panel p-2.5">
                <div className="flex gap-2 items-start">
                  <span className="chip font-mono shrink-0" style={{ color: "var(--accent)" }}>
                    {item.ref}
                  </span>
                  <div className="min-w-0">
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[13px] leading-snug hover:underline block"
                    >
                      {item.title || item.url}
                    </a>
                    <p className="text-[11px] mt-0.5" style={{ color: "var(--muted)" }}>
                      {item.domain}
                      {item.published_at ? ` · ${item.published_at.slice(0, 10)}` : ""}
                      {item.found_by ? ` · found by ${item.found_by}` : ""}
                    </p>
                    {item.snippet && (
                      <p className="text-[11px] mt-1 leading-snug" style={{ color: "var(--faint)" }}>
                        {item.snippet.slice(0, 220)}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
