# Architecture

This document explains how the system works and, specifically, where each
property people mean by "agentic" actually lives in the code. Every claim below
points at a file, because the useful version of this answer is one a reader can
check.

## The two halves

**`mas/core/`** knows nothing about agents. It provides model access across
nineteen providers, embeddings, robots-aware web access, storage, cost metering
and runtime capability tiering. Most of it was carried across from an earlier
news aggregation project after being proven against real sources there.

**`mas/kernel/`** is the agent runtime. It knows about goals, tools, budgets,
steps and checkpoints, and nothing about research. The flagship workflow is
built on top of it and could be replaced without touching it.

That split is the reason the system can honestly be called a framework with a
workflow on it rather than one program described in agent vocabulary.

## Agentic properties, and where each one is

| Property | Where | What it actually does |
|---|---|---|
| Goal decomposition | `agents/planner.py` | Turns a brief into separately researchable subquestions with search hints |
| Autonomous tool use | `kernel/react.py` | A reason and act loop: the model picks the tool and the arguments, turn by turn |
| Tool protocol | `kernel/tool.py` | Declared schemas, validation, coercion, and a recorded invocation for every call |
| Parallel agents | `kernel/orchestrator.py` | A fan out runs in a thread pool and rejoins as one step |
| Shared memory | `kernel/blackboard.py` | A thread safe, JSON shaped board every role reads and one role writes per section |
| Agent messaging | `kernel/bus.py` | Agents post to a bus; the graph decides who runs next |
| Dynamic routing | `agents/supervisor.py` | Chooses between writing and another research round, on the evidence |
| Self correction | `agents/critic.py` | Scores the report and sends it back with specific faults, bounded by a counter |
| Verification | `agents/factchecker.py` | Rules on each claim; failures never reach the Writer |
| Cross run memory | `kernel/store.py` | Lessons like "this domain returns no extractable text" survive the run |
| Budget awareness | `kernel/budget.py` | Ceilings that pause, and a pressure reading roles degrade against |
| Rate awareness | `kernel/pacer.py` | Requests and tokens per minute, per provider |
| Durability | `kernel/orchestrator.py` | Queue and blackboard checkpointed after every step |
| Human in the loop | `kernel/store.py`, orchestrator | A node can park the run on a question and resume on the answer |

## The run loop

`orchestrator.run` is a state machine over a work queue.

```
while queue:
    check the control channel          # pause or cancel, from any process
    check the budget                   # a breach raises and pauses
    pop the next task, or a parallel group
    execute it
    apply its messages, artifacts and routing
    checkpoint the queue and the blackboard
```

Stopping is not a special case. A pause, a cancel, a budget breach, a spent
provider quota and a crash all arrive as exceptions and all take the same path:
write the checkpoint, release the lease, record why. Only the resulting status
differs.

### The invariant

Between two steps, everything describing a run is in the database. Nothing
important lives only in the process. That is what makes stopping free and
resuming exact, and it is why a run started in the terminal can be paused from
the browser.

### The cost of it

A step interrupted part way through is abandoned and re-run on resume, so nodes
execute **at least once**, not exactly once. Everything expensive is
deduplicated underneath: evidence by normalised URL, findings by normalised
statement, artifacts by version. So a re-run costs model calls, not
correctness. Making it exactly once would mean checkpointing inside a node,
which would mean every agent understanding persistence.

### The subtlety that bit once

A task is removed from the queue before it runs. Any path that stops mid step
must put the unfinished tasks back before checkpointing. Missing this produced
a run that paused, resumed, and reported itself complete having written
nothing. `tests/test_orchestrator.py` has a regression test named after it.

## Tool calling

Two protocols, chosen per provider in `ReactLoop.__init__`.

**Native function calling** where the provider has it, which is the sixteen
OpenAI-compatible adapters. This is not an optimisation. Models trained for
native tool use, notably the `gpt-oss` family, emit a tool call whether or not
the request declared any tools, and the gateway rejects the whole request with
a 400 when it did not. Declaring the tools is the only correct behaviour.

**A JSON protocol in the prompt** everywhere else, including Gemini, Anthropic
and Cloudflare. It costs a few hundred tokens of system prompt and works on
every model, including the small local ones.

Both paths are covered in `tests/test_react.py`, because a bug in one is
invisible from the other.

## Cost control

Three mechanisms, in the order they engage:

1. **Pacing** (`pacer.py`). Requests and tokens per minute, per provider,
   reserved before a call and corrected afterwards. Turns a rate limit failure
   into a delay.
2. **Degradation** (`policy.py`). Past 70 percent of the tightest ceiling,
   strong roles drop to the cheap model and the Planner asks fewer questions.
   A smaller finished report beats a larger unfinished one.
3. **Ceilings** (`budget.py`). Steps, tokens, dollars, seconds and tool calls.
   A breach pauses the run with everything intact.

Spend is attributed per role and per step in the `usage` table, which is what
makes the per-agent cost table in the UI possible, and what would show
immediately if one role were quietly doing all the work.

## Storage

Eleven tables in `mas/core/db/schema.py`, portable across SQLite and Postgres.
All are append only except `runs`, so stopping the process at any instant
leaves a consistent record.

The ones that carry the design:

- `runs` holds `control`, which is how another process asks a running
  orchestrator to stop. It is a column rather than a thread event precisely
  because the process that starts a run is often not the one that stops it.
- `checkpoints` holds the queue and the blackboard. Resuming reads the newest
  row and replays nothing.
- `events` is the trace. The UI streams it by id cursor, so a browser that
  reconnects mid run misses nothing.
- `tool_calls` records every invocation with its arguments and result. This is
  the table that distinguishes an agent choosing a tool from a pipeline calling
  a function.

## Frontend

Client rendered, exported to static files, served by the Python process. One
origin, so the event stream needs no CORS and no API address is baked into the
bundle at build time.

The agent graph is drawn from `graph.describe()` rather than from a hardcoded
picture, so adding a role to the workflow makes it appear without anyone
editing the diagram. A diagram maintained by hand stops being true, and a
diagram that has stopped being true is worse than none when the question being
asked is whether this is really a multi agent system.

## Extending it

A new tool is a class with a name, a description, a JSON Schema and a `call`,
decorated with `@tool`. A new role is a module in `mas/agents/` with a system
prompt and a tool list. A new workflow is a `Graph` with nodes and declared
edges, registered in `mas/workflows/`.

The kernel needs no changes for any of those, which is the test of whether the
split described at the top is real.
