# Evaluation: run `20261105T033028-dev-2415c0` (split `dev`, 2 cases)

| Metric | Value |
|---|---|
| Decision accuracy | 0% (0/2; 95% CI 0%-66%) |
| Accuracy on self-decided cases | n/a |
| Payable amount accuracy (correct APPROVEs) | n/a |
| **Unsafe approval rate** (APPROVE when DENY/REFER expected) | 0% (0/1; 95% CI 0%-79%) |
| Forced referrals (by safeguards) | 0% (0/2; 95% CI 0%-66%) |
| Infrastructure failures | 100% (2/2; 95% CI 34%-100%) |
| Passed validation on first attempt | 0% (0/2; 95% CI 0%-66%) |
| Mean expected-section recall | 0% |
| Cost total / per case | $0.0048 / $0.0024 |
| Cost per correct decision | n/a |
| Mean latency per case | 5.3s |

## What these numbers can and cannot support

- **2 cases from 2 independent groups.** One flipped case moves accuracy by 50%; variants of one source case share its failure modes, so they add less evidence than their count suggests.
- Intervals are 95% Wilson intervals: treat a difference smaller than their width as noise. A model comparison needs the paired test in `compare`, not two accuracies.
- A perfect score on a small set is a smoke test, not proof of reliability.
- **Partial run** (case filter applied): not comparable with full runs.

## Where misses come from

| Failure stage | Cases |
|---|---|
| infrastructure | 2 |

Validation errors by check (all attempts):

none

Rejected tool calls: 

none

## Accuracy by slice

| Slice | Value | Accuracy |
|---|---|---|
| expected_outcome | APPROVE | 0% (0/1; 95% CI 0%-79%) |
| expected_outcome | DENY | 0% (0/1; 95% CI 0%-79%) |
| origin | original | 0% (0/2; 95% CI 0%-66%) |
| injection | absent | 0% (0/2; 95% CI 0%-66%) |
| diagram_rule | needed | 0% (0/2; 95% CI 0%-66%) |

## Confusion (rows = expected, columns = predicted)

| expected \ predicted | APPROVE | DENY | REFER | NONE |
|---|---|---|---|---|
| APPROVE | 0 | 0 | 0 | 1 |
| DENY | 0 | 0 | 0 | 1 |
| REFER | 0 | 0 | 0 | 0 |

## Case-level analysis

| Case | Expected | Predicted | Status | Category | Stage | Sections (cited / expected) | Attempts | Cost |
|---|---|---|---|---|---|---|---|---|
| dev-002 | APPROVE | - | failed | infrastructure_failure | infrastructure | - / 5.3, 3.1 | 0 | $0.0024 |
| dev-003 | DENY | - | failed | infrastructure_failure | infrastructure | - / 5.3 | 0 | $0.0024 |
