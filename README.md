# Multi-Agentic

A supervised team of eight LLM agents that researches a question on the open
web, verifies what it found against the sources, and writes a cited report.

Every run can be stopped and resumed. A budget ceiling or a spent free tier
quota pauses it rather than killing it, so nothing gathered is ever lost, and
the recovery is to raise the ceiling or switch provider and carry on.

```
          Planner ──────────────────────────┐
             │ subquestions                 │
          Researcher × N  (parallel)        │
             │ findings                     │ gaps to close
          Analyst                           │
             │ claims                       │
          FactChecker × N  (parallel)       │
             │ verdicts                     │
          Reconciler                        │   do any two claims conflict?
             │                              │
          Supervisor ───────────────────────┘
             │ write
          Writer ◄──────────┐
             │              │
          Editor            │ revise
             │              │
          Critic ───────────┘
             │ accept
          Report
```

There is a second, cheaper workflow for checking a single claim rather than
researching a topic:

```
          Planner            frames the claim as three angles
             │
          Researcher × 3     for it, against it, around it  (parallel)
             │
          FactChecker        weighs all of it, rules with a confidence
             │
          Writer             explains the finding, cited
```

```bash
python -m mas.cli run "Coffee reduces heart disease risk." --workflow claim_check
```

## What makes it a multi-agent system rather than a long prompt

- **Eight roles with different jobs, models and tools.** The Researcher gets
  the cheap model and the web; the Writer gets the strong model and no tools at
  all, so it cannot go back to the internet to avoid a judgement the
  verification stage already made.
- **Agents choose their own actions.** A Researcher runs a reason and act loop:
  it searches, decides which result is worth opening, reads the page, and
  decides when it has a finding. Every one of those choices is recorded with
  its arguments and result.
- **They collaborate over a shared blackboard**, not a chain. Each role writes
  its own section and every role can read all of it, which is why the Writer
  knows which claims the Fact Checker threw out.
- **Two feedback loops.** The Supervisor can send the researchers back out
  against named gaps. The Critic can send a draft back to the Writer with a
  specific list of faults. Both are bounded by counters rather than by trust.
- **Verification removes claims rather than flagging them.** A claim that fails
  the Fact Checker never reaches the Writer. The finished report lists what was
  excluded and why.
- **Contradiction is detected, not hoped for.** Vectors find which verified
  claims are about the same thing, and the Fact Checker is asked about only
  those pairs. Corroboration is counted in independent domains, in code. The
  report names the disagreements it found.
- **Runs are durable.** The queue, the blackboard and the spend are
  checkpointed after every step, so a run survives the process that started it.

## Quick start

```bash
# 1. Python side
uv venv --python 3.12 .venv           # or: python -m venv .venv
.venv/Scripts/pip install -r requirements.txt    # Linux/macOS: .venv/bin/pip

# 2. Keys. Every one is optional; one LLM key is enough to be useful.
cp .env.example .env                  # then fill in what you have

# 3. Check the machine can actually do the work
python -m mas.cli doctor --model

# 4. Run the team
python -m mas.cli run "What did the RBI decide at its last policy meeting, and why?"
```

With no keys at all it still runs end to end: Google News and Wikipedia answer
queries without a credential, the bundled ONNX model deduplicates evidence, and
a local Ollama or LM Studio can drive the agents.

### The web UI

```bash
npm install
npm run build          # exports the frontend into out/
python -m mas.main     # serves the API and the UI on http://localhost:8000
```

One process, one port. For frontend development instead:

```bash
python -m mas.main --reload     # API on 8000
npm run dev                     # UI on 3000, proxied by NEXT_PUBLIC_API_BASE
```

## Watching what it costs

This is built to run on free tiers, so the accounting is not an afterthought.

```bash
python -m mas.cli status <run>     # spend per agent, tool calls, ceilings
python -m mas.cli messages <run>   # who told whom what
python -m mas.cli trace <run>      # the full event log
```

Every run has five ceilings, and **breaching any of them pauses the run**:

| Ceiling | Default | What it stops |
|---|---|---|
| `--max-steps` | 120 | runaway graphs |
| `--max-tokens` | 400,000 | the usual way a budget goes |
| `--max-usd` | 0.50 | paid providers only; free tiers price at zero |
| `--max-seconds` | 1800 | a run that has stalled |
| `--max-tool-calls` | 80 | a researcher that will not stop searching |

Below the ceiling, pressure is reported and acted on: past 70 percent the
strong roles drop to the cheap model and the Planner asks fewer questions, so a
run arrives at a smaller finished report rather than a larger unfinished one.

## Stopping and starting

```bash
python -m mas.cli run "..."                  # Ctrl+C pauses, it does not lose work
python -m mas.cli pause  <run>               # from anywhere, including another process
python -m mas.cli resume <run>
python -m mas.cli resume <run> --max-usd 2   # raise a ceiling and carry on
python -m mas.cli resume <run> --provider groq   # continue on a provider with quota left
python -m mas.cli cancel <run>               # stop for good
```

A paused run keeps every source, finding, claim and verdict. Resuming starts
from the last completed step.

## Models

Nineteen providers behind one interface: Gemini, Groq, Cloudflare Workers AI,
OpenRouter, Cerebras, SambaNova, Mistral, DeepSeek, Together, Fireworks,
Anthropic, OpenAI, xAI, Perplexity, Hugging Face, and Ollama, LM Studio,
llama.cpp and vLLM running locally.

Roles are assigned a tier rather than a model, so one key configures the whole
team:

| Tier | Roles | Why |
|---|---|---|
| cheap | Researcher, FactChecker, Supervisor | high volume, mechanical judgement |
| strong | Planner, Analyst, Writer, Editor, Critic | once or twice a run, sets the quality of everything |

Tool calling uses the provider's native function calling where it exists, and a
JSON protocol in the prompt everywhere else. Both paths are tested.

## Free tier notes

Free tiers are limited by **tokens per minute** far more tightly than by
requests, and the pacer in `mas/kernel/pacer.py` respects both. Tune with:

```
PROVIDER_RPM=gemini:12,groq:25
PROVIDER_TPM=groq:7000,gemini:200000
```

A quota that runs out mid run pauses it. Wait for the window, or:

```bash
python -m mas.cli resume <run> --provider cloudflare
```

## Deployment

The Python process serves the API and the built UI, so there is one thing to
deploy.

- **Docker**: `docker compose up --build`
- **Render**: the included `render.yaml` creates a web service with a 1 GB
  disk. The disk matters: checkpoints are what make runs resumable.
- **Oracle Cloud, a VM, a Raspberry Pi**: `pip install -r requirements.txt`
  then `python -m mas.main`.

**Vercel is not supported and the code is not shaped for it.** An agent run
takes minutes and must survive between requests; a serverless function is
killed at a fixed deadline and has an ephemeral filesystem, so a run would be
cut off with nowhere to write the checkpoint that would let it continue. Local
or a small always-on container is the right home for this.

## Tests

```bash
python -m pytest
```

They spend nothing. The kernel's guarantees (pause, resume, budget breach,
lease exclusion, retries, approval gates, requeue of interrupted work), both
tool calling protocols, argument coercion and rate pacing are all covered
against a scripted model.

## Layout

```
mas/core/      provider layer, embeddings, web access, storage, cost metering
mas/kernel/    blackboard, bus, budget, pacer, tools, reason and act loop,
               graph, orchestrator
mas/agents/    one module per role
mas/workflows/ roles wired into a graph
mas/tools/     what an agent may actually do
mas/api/       FastAPI, and the worker that runs a run in the background
app/ components/ lib/   the Next.js UI
tests/         the kernel's guarantees, without spending a token
```

`CLAUDE.md` holds the conventions and a list of the bugs that cost real time to
find, which is worth reading before changing the kernel.
