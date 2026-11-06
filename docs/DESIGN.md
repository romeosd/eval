# Design notes

## Principles

1. **Engineering over prompting.** Rules that can be checked in code are checked in code:
   arithmetic, citation existence, page/section consistency, case identity, completeness
   and quarantine. The model reads and reasons; code verifies.
2. **Fail closed.** A decision is accepted only if every required check ran and passed.
   Errors, timeouts and unscreenable content lead to `REFER` (a human decides) or
   `FAILED` (no prediction). They never lead to a pass.
3. **No decision logic hard-coded.** Policy values (limits, waiting periods, thresholds)
   are read from the policy by the model and cited. Code supplies generic arithmetic
   only, so the system transfers to a new policy without code changes.
4. **Measure, don't assume.** Every claim in the report should point to a run directory.

## Key decisions and trade-offs

| Decision | Why | Trade-off / alternative tested |
|---|---|---|
| Agentic tool navigation over BM25 + section tools, not vector RAG | Policy wording is precise and the corpus is small; lexical search plus whole-section reads is deterministic, cheap and auditable | `retrieval.mode = "full_context"` puts the whole policy in the prompt. `compare` measures both; pick on accuracy and cost |
| Vision transcription of visual pages at ingest, *and* page images available at decision time | A transcription makes diagram rules searchable and citable; the image lets the decision model check the original | Extra vision cost per visual page, paid once per cache key |
| Structured outputs (`responses.parse`) + strict tool schemas | No manual JSON parsing; schema violations are impossible rather than handled | Strict mode forbids defaults, so the output models are written for it |
| Stateless calls (`store=false`) with encrypted reasoning items replayed | The application owns the full record (audit log, run snapshot) instead of depending on provider-held conversation state. `store=false` is **not** a zero-retention guarantee: provider retention and processing terms still apply. What is replayed is the opaque reasoning continuation item, not the model's chain of thought | Larger requests than `previous_response_id`; sensitive audit logs must be protected by the deployer |
| Separate models per role (adjudicator / judge / vision / classifier) | Spend the strong model on the decision, use cheap ones for verification | The judge's own error rate must be measured |
| Layered injection defence (patterns, then classifier, then redaction, then output and citation checks) | Prompt instructions alone are not a control | Pattern false positives (e.g. "please approve this claim") only redact a sentence |
| Retry with validator feedback, then refer | Most *decision* failures (wrong page, paraphrased quote, unsupported claim) are fixable | Each retry costs a full agent loop; capped by `max_attempts` |
| Failures are classed as `decision` or `infrastructure` | A dead verifier says nothing about the decision, and a new agent run cannot fix it. Infrastructure failures re-run validation only (`max_revalidations`), then refer | A flaky judge turns into referrals rather than extra spend |
| Completeness gate before the agent runs | Policy forbids deciding on incomplete evidence, so a case with an unloadable document is referred without an agent run | The agent never sees such cases; the `COMPLETENESS` check remains as a backstop |
| Judge also receives the page image for cited diagram pages | Diagram transcriptions are model output and can misread an arrow or a condition; the original image is the authority when the two disagree | Extra image tokens per diagram citation (`judge_sees_diagram_images`) |
| `list_policy_sections` tool alongside BM25 | Lexical search misses synonyms ("improper sharing" vs "disclosure"); scanning titles lets the agent recover | One extra cheap tool call; hybrid or embedding retrieval is not tested |
| Run directory snapshots the processed policy and prompt files; manifest records resolved model snapshots and `git_dirty` | A decision should be reconstructable after the policy, prompts or model aliases change | Disk use per run; page images stay in the cache under the same key |
| `judge-eval` measures the judge on corrupted findings | The judge gates every decision, so its false-accept rate must be known, not assumed | Corruptions are automatic and need a manual spot check; controls are biased because they already passed the judge |
| Immutable run directories and manifests | Scores can always be traced to the exact code, config and prompts | Disk usage grows; clean `runs/` manually |
| Image transcription cache | Reruns while iterating shouldn't pay for vision again (candidates pay API costs) | Cache key includes file hash, model, prompt and detail, so changes invalidate it |

## Feedback → implementation traceability

| Assessor feedback (other candidates) | Where it is addressed |
|---|---|
| Diagrams missed before reaching the model | `policy/ingest.py` (visual detection + transcription), `tools.view_policy_page` |
| Application and policy images assessed separately | Both are returned as images inside the same agent loop (`tools.py`) |
| Reuse of processed policy, rebuild on change | `PolicyIngestor.cache_key`, atomic cache writes |
| Hierarchy and page traceability | `policy/sections.py`; citations carry `section_id` + `page` |
| Cited evidence must support the decision | `validation.py`: quote-in-source, page-in-section, per-finding and outcome-level judge (on by default) |
| Injection defence only via instructions | `security.py`, quarantine in `loader.py` and `tools.py`, `QUARANTINE` + `OUTPUT_SCREEN` checks |
| Injection in policy content and images | Ingest screens page text and diagram text; loader screens image transcriptions |
| Missing documents not recovered before concluding | Load retries; failed docs recorded; `COMPLETENESS` forces `REFER` |
| Failed final attempt saved as prediction | `pipeline._Trace.finish`: only validated decisions become predictions |
| Judge timeout left a result marked passed | `LLMClient` timeout raises an error, which becomes an error issue; `ValidationReport.passed` is recomputed on load |
| Case counts not matching files; stale reruns | `runner.load_run`, `metrics.score_run` integrity checks; new run dir each time |
| Task identity and source completeness | `TASK_IDENTITY`, `COMPLETENESS` checks |
| Model comparison, cost/quality, case-level errors | `evaluation/compare.py`, `metrics.py` categories, `eval_report.md` |
| Weak reasoning and evidence-handling robustness | `evaluation/judge_eval.py` (judge false-accept rate); judge sees diagram images |
| Testing beyond provided examples | `evaluation/variants.py` (missing docs, direct/paraphrased injection, threshold crossings) |


## Determinism

The model is the only non-deterministic component. Controls, in order of strength:

| Control | What it gives | Limit |
|---|---|---|
| Deterministic code around the model (ordering, tie-breaks, data generation, canonical keys) | Same inputs, same outputs, tested | None |
| Record and replay (`--replay`) | Bit-identical reruns offline, for regression-testing everything except the model | Cannot measure variance |
| Pinned snapshots (`require_pinned_models`) | Refuses an alias that resolves to a different snapshot | Provider can still change serving infrastructure |
| Fixed reasoning effort per role | Removes one source of variance | The API does not expose seed or temperature for these models |
| Consensus gate (`consensus_runs` > 1) | Outcome and amount must agree across independent runs, else REFER | k times the cost; a systematically wrong model still agrees with itself |
| `consistency` command | Measures agreement across repeated runs | Measurement, not a guarantee |

## Number provenance

Every number given to `calculate` carries a unit and a source. A policy number must appear,
with that unit, in a policy section the agent has read (including diagram transcriptions); an
evidence number must appear in the cited document; a derived number must equal an earlier
calculation. Mixed units (dollars plus hours) are refused. This closes the failure where a
valid-looking calculation uses a hallucinated threshold, for example treating "$5,000" as
"5000 hours". It verifies that a value and unit exist in the cited source, not that they
come from the right clause.

## Decision authority

Outcomes are decided by the model and verified by code; they are not computed by a rules
engine. The alternative (extract rules, validate them, execute deterministically) is the
production path for stable, high-volume policies. It was not chosen here: qualitative terms
("unattended", "material fact") do not compile to rules; an extracted rule is one artifact
that, if wrong, is wrong everywhere; and the sample is too small to validate extracted
rules. See `PRODUCTION_READINESS.md`.
