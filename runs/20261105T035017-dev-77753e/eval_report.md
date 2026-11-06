# Evaluation: run `20261105T035017-dev-77753e` (split `dev`, 2 cases)

| Metric | Value |
|---|---|
| Decision accuracy | 50% (1/2; 95% CI 9%-91%) |
| Accuracy on self-decided cases | 50% (1/2; 95% CI 9%-91%) |
| Payable amount accuracy (correct APPROVEs) | n/a |
| **Unsafe approval rate** (APPROVE when DENY/REFER expected) | 0% (0/1; 95% CI 0%-79%) |
| Forced referrals (by safeguards) | 0% (0/2; 95% CI 0%-66%) |
| Infrastructure failures | 0% (0/2; 95% CI 0%-66%) |
| Passed validation on first attempt | 50% (1/2; 95% CI 9%-91%) |
| Mean expected-section recall | 75% |
| Cost total / per case | $0.9658 / $0.4829 |
| Cost per correct decision | $0.9658 |
| Mean latency per case | 121.8s |

## What these numbers can and cannot support

- **2 cases from 2 independent groups.** One flipped case moves accuracy by 50%; variants of one source case share its failure modes, so they add less evidence than their count suggests.
- Intervals are 95% Wilson intervals: treat a difference smaller than their width as noise. A model comparison needs the paired test in `compare`, not two accuracies.
- A perfect score on a small set is a smoke test, not proof of reliability.
- **Partial run** (case filter applied): not comparable with full runs.

## Where misses come from

| Failure stage | Cases |
|---|---|
| reasoning | 1 |

Validation errors by check (all attempts):

claim_support: 4

Rejected tool calls: 

none

## Accuracy by slice

| Slice | Value | Accuracy |
|---|---|---|
| expected_outcome | APPROVE | 0% (0/1; 95% CI 0%-79%) |
| expected_outcome | DENY | 100% (1/1; 95% CI 21%-100%) |
| origin | original | 50% (1/2; 95% CI 9%-91%) |
| injection | absent | 50% (1/2; 95% CI 9%-91%) |
| diagram_rule | needed | 50% (1/2; 95% CI 9%-91%) |

## Confusion (rows = expected, columns = predicted)

| expected \ predicted | APPROVE | DENY | REFER | NONE |
|---|---|---|---|---|
| APPROVE | 0 | 1 | 0 | 0 |
| DENY | 0 | 1 | 0 | 0 |
| REFER | 0 | 0 | 0 | 0 |

## Case-level analysis

| Case | Expected | Predicted | Status | Category | Stage | Sections (cited / expected) | Attempts | Cost |
|---|---|---|---|---|---|---|---|---|
| dev-002 | APPROVE | DENY | decided | wrong_outcome | reasoning | 2.3, 3.4, 4.2, 5.1, 5.2, 5.3, 5.4 / 5.3, 3.1 | 1 | $0.2311 |
| dev-003 | DENY | DENY | decided | correct | - | 3.4, 5.1, 5.2, 5.3 / 5.3 | 3 | $0.7347 |
