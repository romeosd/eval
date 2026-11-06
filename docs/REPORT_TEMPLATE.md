# Submission report — template

> Fill every bracket with your own words and numbers from a named run directory.
> Cut any section that doesn't apply. Aim for evidence over adjectives.

## 1. Problem and assumptions (½ page)
- What is being decided, by whom, and what a wrong answer costs. Is an unsafe approval
  worse than a referral? Say so, because it drives the metrics you prioritise.
- Every place the brief was ambiguous, and the assumption you made.

## 2. Approach (1 page)
- Architecture diagram and a one-paragraph walkthrough of one case.
- Model choice per role, with the run ids that justify each choice.
- Retrieval strategy, and why it fits the data (tools vs full context: results).

## 3. Evaluation (the core: 1–2 pages)
- Datasets: provided cases, held-out split, generated variants, and how labels were made.
- Headline table: accuracy, unsafe-approval rate, forced-referral rate, section recall,
  cost per case, cost per correct decision.
- Comparison table from `adjudicator compare`, with repeats and variance.
- **Changes driven by evaluation:** for each, the failure seen (case id), the change,
  and the before/after numbers.
- Error analysis: every remaining miss, by category, with the root cause.

## 4. Safeguards (½ page)
- What fails closed and how it was tested (reference the tests).
- Injection results on the variant split.

## 5. Limitations and next steps (½ page)
- What you'd do with another week, and what production would need (human review queue,
  monitoring, drift checks, data retention).

## 6. Use of AI assistance
- What you designed and wrote yourself, what you delegated, and how you verified it
  (review, strict typing, tests, eval runs).
