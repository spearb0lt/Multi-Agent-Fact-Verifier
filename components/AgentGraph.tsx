"use client";

/**
 * The workflow, drawn from the graph the server actually runs.
 *
 * Nothing about the shape is hardcoded here. The nodes, the edges and their
 * conditions come from `graph.describe()`, so adding an agent to the workflow
 * makes it appear in this picture without anyone editing the picture. That is
 * the point: a diagram maintained by hand stops being true, and a diagram that
 * has stopped being true is worse than none when the question being asked is
 * "is this really a multi agent system".
 *
 * Layout is a layered walk from the entry node. Back edges, which are the two
 * feedback loops, are drawn as curves to the side so they read as loops rather
 * than as crossings.
 */

import { useMemo } from "react";
import type { GraphShape } from "@/lib/api";
import { agentColour } from "@/lib/agents";

export type NodeState = "idle" | "active" | "done" | "error";

interface Props {
  graph: GraphShape;
  states: Record<string, NodeState>;
  counts?: Record<string, number>;
  onSelect?: (node: string) => void;
  selected?: string;
}

const NODE_W = 132;
const NODE_H = 46;
const GAP_X = 42;
const GAP_Y = 30;
const PAD = 20;
// Horizontal room per feedback edge, and for the condition label beside it.
const LOOP_LANE = 26;
const LOOP_LABEL = 74;

interface Placed {
  name: string;
  agent: string;
  label: string;
  parallel: boolean;
  description: string;
  x: number;
  y: number;
  layer: number;
}

function layout(graph: GraphShape) {
  const names = graph.nodes.map((n) => n.name);

  // Find the back edges first. Without this the two feedback loops push their
  // own targets forward on every relaxation pass, so the Planner ends up below
  // the Supervisor that loops back to it and the picture is nonsense. A depth
  // first walk from the entry calls an edge a back edge when its target is
  // still on the stack, which is the textbook definition and is exactly the
  // set of edges that must not contribute to a node's depth.
  const outgoing = new Map<string, string[]>();
  names.forEach((name) => outgoing.set(name, []));
  graph.edges.forEach((e) => outgoing.get(e.source)?.push(e.target));

  const backEdges = new Set<string>();
  const onStack = new Set<string>();
  const visited = new Set<string>();

  const walk = (node: string) => {
    visited.add(node);
    onStack.add(node);
    for (const next of outgoing.get(node) || []) {
      if (onStack.has(next)) {
        backEdges.add(`${node}->${next}`);
      } else if (!visited.has(next)) {
        walk(next);
      }
    }
    onStack.delete(node);
  };
  walk(graph.entry);
  // A node unreachable from the entry would not be in a validated graph, but
  // drawing must not depend on that holding.
  names.forEach((name) => {
    if (!visited.has(name)) walk(name);
  });

  const forward = graph.edges.filter(
    (e) => e.source !== e.target && !backEdges.has(`${e.source}->${e.target}`),
  );

  // Longest path over the remaining acyclic edges, so a node always sits below
  // everything that feeds it.
  const layer = new Map<string, number>();
  names.forEach((name) => layer.set(name, 0));

  for (let pass = 0; pass < names.length; pass += 1) {
    let changed = false;
    for (const edge of forward) {
      const proposed = (layer.get(edge.source) ?? 0) + 1;
      if (proposed > (layer.get(edge.target) ?? 0)) {
        layer.set(edge.target, proposed);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const byLayer = new Map<number, string[]>();
  graph.nodes.forEach((node) => {
    const depth = layer.get(node.name) ?? 0;
    if (!byLayer.has(depth)) byLayer.set(depth, []);
    byLayer.get(depth)!.push(node.name);
  });

  // Rows are numbered from zero upward with no gaps, so a layer nothing landed
  // in does not leave a band of empty canvas.
  const used = Array.from(byLayer.keys()).sort((a, b) => a - b);
  const row = new Map(used.map((depth, index) => [depth, index]));

  const placed: Placed[] = [];
  const widest = Math.max(...Array.from(byLayer.values()).map((l) => l.length), 1);
  const width = PAD * 2 + widest * NODE_W + (widest - 1) * GAP_X;

  used.forEach((depth) => {
      const names = byLayer.get(depth)!;
      const rowWidth = names.length * NODE_W + (names.length - 1) * GAP_X;
      const startX = (width - rowWidth) / 2;
      names.forEach((name, index) => {
        const node = graph.nodes.find((n) => n.name === name)!;
        placed.push({
          name,
          agent: node.agent,
          label: node.label,
          parallel: node.parallel,
          description: node.description,
          x: startX + index * (NODE_W + GAP_X),
          y: PAD + (row.get(depth) ?? 0) * (NODE_H + GAP_Y),
          layer: row.get(depth) ?? 0,
        });
      });
    });

  const rows = used.length;

  // Each feedback edge gets its own lane down the right hand side, and the
  // canvas is widened to hold them. Without the extra width the curves and
  // their labels are drawn outside the viewBox and simply disappear.
  const loops = graph.edges.filter((e) => {
    const from = layer.get(e.source) ?? 0;
    const to = layer.get(e.target) ?? 0;
    return to <= from && e.source !== e.target;
  });
  const lanes = new Map(loops.map((e, i) => [`${e.source}->${e.target}`, i]));
  const lane = (key: string) => (lanes.get(key) ?? 0) * LOOP_LANE;
  const rightMargin = loops.length ? loops.length * LOOP_LANE + LOOP_LABEL : 0;

  return {
    placed,
    lane,
    width: width + rightMargin,
    contentWidth: width,
    height: PAD * 2 + rows * NODE_H + (rows - 1) * GAP_Y,
  };
}

export default function AgentGraph({ graph, states, counts, onSelect, selected }: Props) {
  const { placed, width, height, contentWidth, lane } = useMemo(() => layout(graph), [graph]);
  const index = useMemo(() => new Map(placed.map((p) => [p.name, p])), [placed]);

  return (
    <div className="overflow-x-auto scroll-thin">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ maxWidth: width, minWidth: Math.min(width, 520) }}
        role="img"
        aria-label={`Workflow ${graph.name}: ${graph.nodes.length} agents`}
      >
        <defs>
          <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--faint)" />
          </marker>
          <marker id="arrow-loop" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
            <path d="M0,0 L8,4 L0,8 z" fill="var(--warn)" />
          </marker>
        </defs>

        {graph.edges.map((edge, i) => {
          const from = index.get(edge.source);
          const to = index.get(edge.target);
          if (!from || !to) return null;

          const isLoop = to.layer <= from.layer;
          const x1 = from.x + NODE_W / 2;
          const y1 = from.y + NODE_H;
          const x2 = to.x + NODE_W / 2;
          const y2 = to.y;

          if (isLoop) {
            // Swing out into this edge's own lane, to the right of every node,
            // so a feedback edge reads as a loop rather than as a line
            // crossing every layer between its two ends.
            const side = contentWidth + lane(`${edge.source}->${edge.target}`) + 12;
            const path = `M ${from.x + NODE_W} ${from.y + NODE_H / 2}
                          C ${side} ${from.y + NODE_H / 2}, ${side} ${to.y + NODE_H / 2},
                            ${to.x + NODE_W} ${to.y + NODE_H / 2}`;
            return (
              <g key={i}>
                <path
                  d={path}
                  fill="none"
                  stroke="var(--warn)"
                  strokeWidth="1.3"
                  strokeDasharray="4 3"
                  markerEnd="url(#arrow-loop)"
                />
                {edge.condition && (
                  <text
                    x={side + 6}
                    y={(from.y + to.y) / 2 + NODE_H / 2}
                    fontSize="9"
                    fill="var(--warn)"
                  >
                    {edge.condition}
                  </text>
                )}
              </g>
            );
          }

          const midY = (y1 + y2) / 2;
          return (
            <g key={i}>
              <path
                d={`M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`}
                fill="none"
                stroke="var(--faint)"
                strokeWidth="1.2"
                markerEnd="url(#arrow)"
              />
              {edge.condition && (
                <text
                  x={x2 + NODE_W / 2 + 8}
                  y={y2 - 6}
                  fontSize="9"
                  fill="var(--faint)"
                >
                  {edge.condition}
                </text>
              )}
            </g>
          );
        })}

        {placed.map((node) => {
          const state = states[node.name] || "idle";
          const colour = agentColour(node.agent);
          const count = counts?.[node.name];
          const isSelected = selected === node.name;

          const fill =
            state === "active"
              ? "var(--accent-soft)"
              : state === "error"
                ? "var(--error-soft)"
                : state === "done"
                  ? "var(--panel-2)"
                  : "var(--panel)";
          const stroke =
            state === "active"
              ? "var(--accent)"
              : state === "error"
                ? "var(--error)"
                : isSelected
                  ? "var(--accent)"
                  : "var(--line)";

          return (
            <g
              key={node.name}
              transform={`translate(${node.x}, ${node.y})`}
              onClick={() => onSelect?.(node.name)}
              style={{ cursor: onSelect ? "pointer" : "default" }}
            >
              <title>{`${node.agent || node.label}: ${node.description}`}</title>
              <rect
                width={NODE_W}
                height={NODE_H}
                rx="8"
                fill={fill}
                stroke={stroke}
                strokeWidth={state === "active" || isSelected ? 2 : 1}
              />
              <rect width="3" height={NODE_H} rx="1.5" fill={colour} />
              <text x="12" y="19" fontSize="11.5" fontWeight="600" fill="var(--text)">
                {node.agent || node.label}
              </text>
              <text x="12" y="33" fontSize="9.5" fill="var(--muted)">
                {node.label}
                {node.parallel ? "  (parallel)" : ""}
              </text>
              {state === "active" && (
                <circle cx={NODE_W - 12} cy="14" r="4" fill="var(--accent)">
                  <animate
                    attributeName="opacity"
                    values="1;0.25;1"
                    dur="1.4s"
                    repeatCount="indefinite"
                  />
                </circle>
              )}
              {state === "done" && (
                <path
                  d={`M ${NODE_W - 18} 13 l 3.5 3.5 l 6 -7`}
                  stroke="var(--ok)"
                  strokeWidth="1.8"
                  fill="none"
                  strokeLinecap="round"
                />
              )}
              {typeof count === "number" && count > 0 && (
                <text
                  x={NODE_W - 10}
                  y={NODE_H - 8}
                  fontSize="9"
                  textAnchor="end"
                  fill="var(--faint)"
                >
                  {count}x
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
