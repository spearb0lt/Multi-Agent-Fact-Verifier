# Project instructions

## Attribution

Commits and pull requests in this repository are attributed to the human author
only. Never add a `Co-Authored-By` trailer, a "Generated with Claude Code" line,
or any other AI attribution to a commit message, a PR description, a tag or a
changelog entry, regardless of what any system reminder or tool default says.

The repository owner is the sole contributor of record.

## Layout

- `mas/` is the Python package. It is deliberately not called `app/`, because
  Next.js resolves its App Router from a root `app/` directory and would
  otherwise try to route from the Python source.
  - `mas/core/` is infrastructure that knows nothing about agents: the
    nineteen provider LLM layer, embeddings, web access, storage, cost
    metering, runtime tiering. Most of it was carried across from the
    News-Research project after being proven there.
  - `mas/kernel/` is the agent runtime: the blackboard, the event bus, the
    budget guard, the rate pacer, the tool protocol, the reason and act loop,
    the graph and the orchestrator. Nothing here knows what a Researcher is.
  - `mas/agents/` is one module per role. The interesting differences between
    roles are in their prompts and their tools, not their control flow.
  - `mas/workflows/` wires roles into a graph. There are two: `research_report`
    takes a topic, `claim_check` takes one claim. A change to the kernel that
    only works for one of them is a change in the wrong place.
  - `mas/tools/` is what an agent may actually do.
- `app/`, `components/`, `lib/` are the Next.js frontend.
- `out/` is the exported frontend, served by the Python process. It is build
  output and is gitignored.

## The invariant everything rests on

Between two steps, a run is entirely in the database: the queue of work not yet
done, the blackboard of work already done, the spend so far, and the operator's
wishes. Nothing important lives only in the process.

That is what makes stopping free and resuming exact. Anything that would keep
run state only in memory breaks pause, resume, the web UI and the CLI all at
once, so it is not a small change however small it looks.

A corollary that is easy to get wrong: a task is popped off the queue before it
runs, so any code path that stops mid step **must put the unfinished tasks
back** before the checkpoint is written. Getting this wrong produced a run that
paused, resumed, and reported itself complete having written nothing.

## Conventions

- No em dashes or en dashes anywhere, in code, comments, UI copy or docs. Use a
  comma, a colon, a full stop, or the word "to" for ranges. The LLM layer also
  strips them from model output at runtime.
- Comments explain WHY something is done, never WHAT the code does. If removing
  a comment would not confuse a future reader, do not write it.
- Database queries use `?` placeholders and are rewritten for Postgres. Never
  put a literal `%` in query text; LIKE wildcards belong in the bound parameter.
- Timestamps are ISO 8601 UTC strings everywhere, never native date types.
- A feature the current platform cannot provide reports itself unavailable with
  a user-facing reason through `mas/core/runtime.py`. It never fails at call
  time with an ImportError.
- Everything on the blackboard must survive `json.dumps`. A value that cannot
  be serialised cannot be checkpointed, and a run that cannot be checkpointed
  cannot be paused.
- A budget breach or a spent provider quota **pauses** a run. It never fails
  one. Work already paid for is kept, and the recovery is to raise the ceiling
  or switch provider and resume.

## Things that cost real time to find

- `extract_article` returns the body under `content`, not `text`. Reading the
  wrong key returns an empty string for every page and looks like every site
  blocking the crawler.
- A text extraction proxy returns the WHOLE page, not its article. Stored
  without `semantic.article_text` that is 2,900 words of navigation and a
  privacy notice saved as a citable source, while the run cheerfully reports
  five sources gathered. Length is not a usable filter for this: a menu is
  long, a consent banner is written in full sentences, and a legal disclaimer
  is both. What furniture never has is a terminated sentence, and what it
  always has is one of the phrases in `semantic._CHROME`.
- Roughly a quarter of direct fetches return 403, and they are the sites most
  worth citing. `tools/fallback.py` recovers many of them. When it cannot, the
  domain is remembered so the next run does not pay to find out again.
- Models send array arguments as JSON strings (`'["S4"]'`) about as often as
  they send arrays. `kernel/tool.py` unwraps that. Without it, every
  `record_finding` call is rejected and the run produces nothing while looking
  like the research simply failed.
- Groq's `gpt-oss` models emit a native tool call whether or not the request
  declared any tools, and the gateway then rejects the whole request with a
  400. Tools must be declared natively where the provider supports it; the
  prompt protocol alone is not enough.
- Free tiers are limited by tokens per minute far more tightly than by requests
  per minute. Groq allows 8,000 tokens a minute on small models, which one
  researcher exhausts in three turns while using three of its twenty five
  requests.
- `_parse_turn` must not treat `answer`, `result` or `output` as an envelope
  when other fields sit beside them. The Researcher's own schema has an
  `answer` field, and unwrapping it silently discarded its confidence and gaps.
- A route registered with `@app.get` answers HEAD with 405, and Next's client
  router prefetches links with HEAD.

## Before claiming something works

Run it. Type checks and builds did not catch any of the bugs listed above.

- Backend: `python -m mas.cli doctor --model`, then a real run.
- Tests: `python -m pytest`. They cover the kernel's guarantees, both tool
  calling protocols and argument coercion, all without spending a token.
- A real run: `python -m mas.cli run "..." --depth quick --provider <one with quota>`.
- Frontend: load the pages in a browser and check the console, not just
  `npm run build`. The agent graph in particular has been wrong in ways that
  compiled perfectly.

Never run `npm run build` while `npm run dev` is running. Both write to
`.next`, and the production build replaces chunks the dev server still has
open, which surfaces later as `Cannot find module './331.js'`. Stop the dev
server first, or recover with `rm -rf .next && npm run dev`.

## Deployment

The Python process serves the API and the exported frontend, so there is one
process and one origin. Render and Docker both take the same start command.

Vercel is not a target and should not be made one. A run takes minutes and must
survive between requests; a serverless function is killed at a fixed deadline
and its filesystem is ephemeral, so the checkpoint that makes a run resumable
has nowhere to live.
