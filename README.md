# Policy Adjudicator

Decides applications (here: travel-insurance claims) against policy documents, using an
agent that gathers evidence with tools and returns a structured decision
(`APPROVE` / `DENY` / `REFER`). Every decision is checked before it is accepted. The
checks cover citations, arithmetic, completeness and injection, plus an independent
judge for each finding. Anything that fails is referred to a human rather than guessed.

The repo ships with a synthetic policy and 13 labelled cases, so it runs end-to-end
before the real brief arrives. **Swap the data and prompts for the real task. The
architecture, safeguards and evaluation harness carry over.**

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                    # add your OPENAI_API_KEY

adjudicator make-sample-data            # policy PDF (with a diagram), cases, labels
adjudicator ingest                      # build + inspect the processed policy index
adjudicator run --split dev --cases dev-002,dev-003   # cheap partial run while iterating
adjudicator run --split dev             # full dev run + evaluation report
adjudicator make-variants --source dev  # labelled unseen-data variants
adjudicator run --split variants
adjudicator run --split test            # held-out split: run rarely, never tune on it
adjudicator show runs/<run-id> dev-003  # decision trail for a demo or debugging
adjudicator judge-eval runs/<run-id>    # measure the claim-support judge (false-accept rate)
adjudicator audit-labels                # check labels, splits and leakage (no model calls)
adjudicator consistency runs/<a> runs/<b>   # run-to-run agreement across repeats
adjudicator run --resume runs/<run-id>  # continue an interrupted run
adjudicator --replay record run --split dev   # record model calls
adjudicator --replay replay run --split dev   # replay them offline: no key, no spend
adjudicator compare --split dev --models gpt-5.5:medium,gpt-5.4-mini:low \
    --retrieval tools,full_context --repeats 2
```

Quality gates (`make check`, also run by `.github/workflows/ci.yml`): `ruff check`,
`ruff format --check`, `pyright` (strict), `pytest` (147 offline tests), and the label audit.
All run with no API key and no spend.

## Configuration

**Every tunable value is in [`config/settings.toml`](config/settings.toml).** That
includes models and reasoning effort per role, retrieval mode, attempt and turn limits,
validation thresholds, security switches, concurrency, the run budget and prices. The
API key comes from the environment or `.env`. Prompts live in [`prompts/`](prompts/) as
versioned files, and their hashes are recorded in every run.

> Prices in the config are **placeholders**. Update them before trusting cost figures,
> because the budget guard depends on them.

## How it works

```
policy PDFs ──► ingest: text + page renders + vision transcription of diagrams
                → injection screen + redaction → section hierarchy with page spans
                → cached under a hash of (source, models, prompts, settings, code version)

case folder ──► load: form + attachments (retried); images transcribed (cached)
                → injection screen; flagged images quarantined
                → agent loop (Responses API, strict tools, structured output)
                → validate (deterministic checks, then LLM claim-support judge)
                → retry with validator feedback, up to max_attempts
                → DECIDED | REFERRED | FAILED   (+ audit.jsonl per case)
```

| Module | Responsibility |
|---|---|
| `config.py` | Typed settings; validates that every model has a price |
| `llm/client.py` | Responses API gateway: budget guard, timeouts, cost, audit |
| `policy/` | PDF ingest, section hierarchy, BM25 search, full-context rendering |
| `application/loader.py` | Case loading, retries, image transcription cache |
| `security.py` | Pattern and classifier injection screening, redaction |
| `tools.py` | Strict tool schemas and handlers; untrusted-content labelling |
| `calculations.py` | Deterministic arithmetic; every result has an id |
| `agent.py` | Tool-calling loop ending in an `AgentDecision` |
| `validation.py` | Fail-closed checks, including the claim-support judge |
| `pipeline.py` / `runner.py` | Per-case state machine; immutable runs and manifests |
| `evaluation/` | Scoring, error analysis, variants, model/strategy comparison |
| `evaluation/judge_eval.py` | Probes the judge with corrupted findings |
| `explain.py` | One-case trace for walkthroughs |

The design rationale and how each piece maps to the assessor feedback are in
[`docs/DESIGN.md`](docs/DESIGN.md). [`docs/REPORT_TEMPLATE.md`](docs/REPORT_TEMPLATE.md)
is a skeleton for the submission report, and [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)
is a demo script.

## Outputs of a run (`runs/<run-id>/`)

`manifest.json` (git SHA and dirty flag, settings fingerprint and snapshot, prompt hashes,
policy cache key, resolved model snapshot ids, case ids) · `policy_index.json` and `prompts/`
(copies of what the run used) · `results/<case>.json` · `predictions.jsonl` · `cases/<case>/audit.jsonl`
· `eval_report.md` / `eval_summary.json` / `eval_cases.json` · `run.log`.

A run directory is never reused for a *new* run. `run --resume` continues the same run: it
keeps finished cases, re-runs missing or infrastructure-failed ones, counts the budget
already spent, and refuses to resume if models, retrieval, agent, validation or security
settings, the prompts, or the processed policy changed. Scoring refuses a run that didn't
finish, a result from another run, or labels that don't match the run's cases.

## Reconstructing a decision later

Each run keeps the manifest, the processed policy (`policy_index.json`), the prompt files,
every model and tool call (`audit.jsonl`, with full tool outputs up to the limits in
`[audit]`), the calculations, the validation history and the rejected candidates. The
manifest records the model snapshot ids the provider actually served, because aliases move,
and whether the working tree had uncommitted changes. Not captured: the page images (they
stay in `.cache/` under the policy cache key) and the provider's own retention.
`store=false` is not a zero-retention guarantee.

## Determinism, evaluation and production notes

- [`docs/EVALUATION.md`](docs/EVALUATION.md): the evaluation protocol (intervals, failure
  stages, paired comparison, consistency, held-out discipline). **No results have been
  produced yet.**
- [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md): what is built, what is
  designed but not built, and what is out of scope.
- The model is the only non-deterministic part. See "Determinism" in
  [`docs/DESIGN.md`](docs/DESIGN.md) for the controls (pinned snapshots, record/replay,
  consensus gate) and their limits.

## Known limitations

- Section detection relies on numbered headings (`5.3 Title`). A real policy with
  different typography needs a revised `HEADING` pattern, or a layout-aware parser.
- Page footers end up in the last section on each page. They're harmless for citation
  checks but add noise to search.
- Scanned PDFs submitted as applications must be supplied as images.
- Variant labels encode sample-policy facts (for example the 60-day claim window).
  Re-derive them for the real policy.
- Search is lexical. `list_policy_sections` lets the agent recover from a synonym miss, but
  hybrid or embedding retrieval was not built or compared.
- The judge sees diagram page images, but whether it catches a misread arrow is unmeasured
  until `judge-eval` is run with a human-checked sample.
- The judge is itself a model. It reduces unsupported findings but doesn't eliminate
  them, so measure its agreement with labels before relying on it.
