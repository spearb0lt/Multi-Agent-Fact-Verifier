"use client";

/**
 * A small markdown renderer for the shape the Writer is told to produce.
 *
 * Deliberately not a library. The report's syntax is known and narrow because
 * the Writer's prompt fixes it: headings, paragraphs, bullets, numbered lists,
 * links, bold, italic, and the [S3] citation marker. Pulling in a general
 * markdown parser and a sanitiser would add weight to handle syntax the
 * document is instructed never to contain.
 *
 * Nothing is rendered as raw HTML. Every node below is a real React element,
 * so a report that happened to contain angle brackets is text, not markup.
 */

import { Fragment, type ReactNode } from "react";

const CITE = /\[(S\d+)\]/g;
const LINK = /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g;
const BOLD = /\*\*([^*]+)\*\*/g;
const CODE = /`([^`]+)`/g;

interface Props {
  text: string;
  onCitation?: (ref: string) => void;
}

/** Inline formatting, applied in one pass so the patterns cannot nest badly. */
function inline(text: string, key: string, onCitation?: (ref: string) => void): ReactNode[] {
  type Token = { start: number; end: number; node: ReactNode };
  const tokens: Token[] = [];

  const collect = (pattern: RegExp, build: (m: RegExpExecArray, i: number) => ReactNode) => {
    pattern.lastIndex = 0;
    let match: RegExpExecArray | null;
    let index = 0;
    while ((match = pattern.exec(text)) !== null) {
      const overlaps = tokens.some((t) => match!.index < t.end && match!.index + match![0].length > t.start);
      if (!overlaps) {
        tokens.push({
          start: match.index,
          end: match.index + match[0].length,
          node: build(match, index),
        });
      }
      index += 1;
    }
  };

  collect(LINK, (m, i) => (
    <a key={`${key}-l${i}`} href={m[2]} target="_blank" rel="noopener noreferrer">
      {m[1]}
    </a>
  ));
  collect(CITE, (m, i) => (
    <sup
      key={`${key}-c${i}`}
      className="cite"
      role={onCitation ? "button" : undefined}
      tabIndex={onCitation ? 0 : undefined}
      onClick={() => onCitation?.(m[1])}
      onKeyDown={(e) => {
        if (onCitation && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          onCitation(m[1]);
        }
      }}
      style={onCitation ? { cursor: "pointer" } : undefined}
      title={onCitation ? `Show source ${m[1]}` : undefined}
    >
      [{m[1]}]
    </sup>
  ));
  collect(BOLD, (m, i) => <strong key={`${key}-b${i}`}>{m[1]}</strong>);
  collect(CODE, (m, i) => <code key={`${key}-k${i}`}>{m[1]}</code>);

  tokens.sort((a, b) => a.start - b.start);

  const out: ReactNode[] = [];
  let cursor = 0;
  tokens.forEach((token, i) => {
    if (token.start > cursor) out.push(<Fragment key={`${key}-t${i}`}>{text.slice(cursor, token.start)}</Fragment>);
    out.push(token.node);
    cursor = token.end;
  });
  if (cursor < text.length) out.push(<Fragment key={`${key}-tail`}>{text.slice(cursor)}</Fragment>);
  return out;
}

export default function Markdown({ text, onCitation }: Props) {
  if (!text?.trim()) {
    return <p style={{ color: "var(--muted)" }}>Nothing written yet.</p>;
  }

  const blocks: ReactNode[] = [];
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const body = paragraph.join(" ");
    blocks.push(<p key={`p${blocks.length}`}>{inline(body, `p${blocks.length}`, onCitation)}</p>);
    paragraph = [];
  };

  const flushList = () => {
    if (!list) return;
    const current = list;
    const Tag = current.ordered ? "ol" : "ul";
    blocks.push(
      <Tag key={`l${blocks.length}`}>
        {current.items.map((item, i) => (
          <li key={i}>{inline(item, `l${blocks.length}-${i}`, onCitation)}</li>
        ))}
      </Tag>,
    );
    list = null;
  };

  lines.forEach((raw) => {
    const line = raw.trimEnd();

    if (!line.trim()) {
      flushParagraph();
      flushList();
      return;
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length, 6);
      const Tag = `h${level}` as "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
      blocks.push(
        <Tag key={`h${blocks.length}`}>{inline(heading[2], `h${blocks.length}`, onCitation)}</Tag>,
      );
      return;
    }

    const bullet = /^[-*]\s+(.*)$/.exec(line.trim());
    if (bullet) {
      flushParagraph();
      if (!list || list.ordered) {
        flushList();
        list = { ordered: false, items: [] };
      }
      list.items.push(bullet[1]);
      return;
    }

    const numbered = /^(\d+)[.)]\s+(.*)$/.exec(line.trim());
    if (numbered) {
      flushParagraph();
      if (!list || !list.ordered) {
        flushList();
        list = { ordered: true, items: [] };
      }
      list.items.push(numbered[2]);
      return;
    }

    flushList();
    paragraph.push(line.trim());
  });

  flushParagraph();
  flushList();

  return <div className="prose-report">{blocks}</div>;
}
