# Evaluation protocol

Accuracy is the least informative number in this project. The protocol below is built to
answer four questions: *is it right, where does it go wrong, would a change help, and can
the result be reproduced?* **No result has been produced yet; this document is the plan and
the tooling that carries it out.**

## Principles

1. **Audit the data before the model.** `adjudicator audit-labels` checks labels, splits
   and leakage with no model calls.
2. **Report uncertainty.** Every headline rate carries a 95% Wilson interval and the report
   states how far one flipped case moves accuracy.
3. **Attribute every miss to a stage**, so each failure points at a fix.
4. **Compare by pairs, not by headline.** Two configurations are compared on the cases where
   they disagree (exact sign test), never on two accuracies.
5. **Measure the variance.** Repeat runs; report agreement across them.
6. **One change per iteration**, with before and after numbers from named runs.
7. **Freeze the held-out split** and use it once.

## Data

| Split | Purpose | Notes |
|---|---|---|
| `dev` | Iteration and error analysis | Look at every miss |
| `variants` | Robustness to missing documents, injections, threshold crossings | Derived from `dev`; variants of one source case are **not independent** |
| `test` | Final check | Run once, at the end. The report counts runs on this split and warns after the second |

Labels (`data/labels/<split>/<case>.json`): `expected_outcome`; `expected_payable_amount`
(APPROVE only); `expected_sections` (the policy section numbers a correct decision must rely
on, e.g. `5.3`); `notes`; `synthetic` and `source_case_id` for variants. Variant labels are
derived by rules in `evaluation/variants.py` that encode facts about the sample policy:
**re-derive them for any other policy.**

## Metrics

| Metric | Definition |
|---|---|
| Decision accuracy | Prediction equals the expected outcome. A system that refers everything scores on REFER cases only |
| Accuracy on self-decided cases | Same, restricted to cases the system decided without a forced referral |
| Unsafe approval rate | Among cases where DENY or REFER was expected, the share predicted APPROVE. **The number to keep at zero** |
| Payable amount accuracy | Among correctly approved cases, amount within one cent |
| Section recall | Share of expected sections the decision cited |
| Forced referral rate | Cases REFERRED by safeguards, not by the model's own choice |
| First-attempt pass rate | Cases whose first candidate passed validation: how often the agent needs correcting |
| Cost per correct decision | Total spend over correct cases |

### Failure stages (one per miss, assigned in this order)

`infrastructure` (API or budget failure) → `retrieval` (an expected section was never read)
→ `diagram` (section read, its diagram page never inspected) → `over_referral` (safeguards
referred a case a human was not expected to see) → `unsafe_approval` (APPROVE after
validation passed) → `amount` (right outcome, wrong amount) → `reasoning` (the evidence was
read and the conclusion is still wrong). In `full_context` mode retrieval and diagram are
not blamed, because the whole policy is in the prompt.

Also reported: validation errors by check across every attempt (which safeguard fires most),
rejected tool calls (for example calculation inputs that failed provenance), and accuracy by
slice (expected outcome, variant kind, injection present, diagram rule needed).

## Procedure

```bash
adjudicator audit-labels                                  # 1. data first; must print no ERROR
adjudicator run --split dev --cases dev-003,dev-006       # 2. cheap smoke test
adjudicator run --split dev                               # 3. baseline; read every miss
adjudicator make-variants && adjudicator run --split variants
adjudicator judge-eval runs/<dev-run>                     # 4. how reliable is the judge?
adjudicator compare --split dev --models A,B --retrieval tools,full_context --repeats 3
adjudicator consistency runs/<run-1> runs/<run-2> runs/<run-3>
# 5. iterate: change ONE thing, rerun dev, record before and after
adjudicator run --split test                              # 6. once, last
```

## Statistics and what can be claimed

- With *n* cases one flip moves accuracy by 1/*n*. At *n* = 13, 13/13 has a 95% interval of
  about 77-100%: a smoke test, not proof of reliability.
- Variants multiply the case count, not the evidence: the report states the number of
  independent groups.
- The paired test is exact. At these sample sizes almost nothing is significant; say so
  rather than ranking configurations.
- Repeats of one configuration are not independent samples of the world; they measure
  run-to-run variance only.

## Determinism and variance

The model is the only non-deterministic component; everything around it is deterministic
and tested. Controls: pinned snapshot names with `require_pinned_models`; fixed reasoning
effort; record and replay of model calls (`--replay record|replay`) to reproduce a run
exactly offline; optional `consensus_runs` > 1 so disagreeing runs become a referral;
`consistency` to measure agreement across repeats. Replay cannot measure variance, because it
returns recorded answers: measure variance with replay `off`.

## Reporting checklist

State the split and run ids; show intervals; give the failure-stage table; report unsafe
approvals first; show before and after for every change; include the paired test for any
model comparison; list what was not measured; say how labels were produced.

## Known limitations

Labels were written by the author of the system; a second labeller would be better. The
judge's error rate is unmeasured until `judge-eval` is run and hand-checked. Section-level
recall does not check that the right *clause* within a section was used.
