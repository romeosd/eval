"""Run-to-run consistency: how much does the same system disagree with itself?

A hosted model cannot be made bit-deterministic, so this measures the variance instead of
assuming it away. Agreement is judged on what matters downstream (the prediction and the
payable amount); evidence overlap shows whether the same policy and documents were relied on.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, ConfigDict

from adjudicator.evaluation.metrics import load_labels
from adjudicator.runner import RunIntegrityError, load_run
from adjudicator.schemas import CaseResult


class CaseConsistency(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    outcome_keys: list[str]
    agreement: float  # share of runs matching the most common outcome
    evidence_overlap: float | None  # mean pairwise Jaccard of cited policy sections/documents
    correct_runs: int | None
    unstable: bool


class ConsistencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_ids: list[str]
    n_runs: int
    n_cases: int
    mean_agreement: float
    identical_cases: int
    unstable_cases: list[str]
    correctness_flips: list[str]  # right in some runs, wrong in others
    mean_evidence_overlap: float | None
    snapshot_drift: bool  # the resolved model snapshot differed between the runs
    per_case: list[CaseConsistency]


def outcome_key(result: CaseResult) -> str:
    amount = "" if result.payable_amount is None else f"{result.payable_amount:.2f}"
    return f"{result.prediction.value if result.prediction else 'NONE'}/{amount}"


def evidence_key(result: CaseResult) -> frozenset[str]:
    if result.decision is None:
        return frozenset()
    keys: set[str] = set()
    for finding in result.decision.findings:
        keys.update(f"policy:{c.section_id}" for c in finding.policy_citations)
        keys.update(f"evidence:{c.document_id}" for c in finding.evidence_citations)
    return frozenset(keys)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def analyse(run_dirs: list[Path], labels_dir: Path | None = None) -> ConsistencySummary:
    if len(run_dirs) < 2:
        raise RunIntegrityError("Consistency needs at least two runs of the same split")
    loaded = [load_run(d) for d in run_dirs]
    splits = {manifest.split for manifest, _ in loaded}
    case_sets = {tuple(sorted(r.case_id for r in results)) for _, results in loaded}
    if len(splits) != 1 or len(case_sets) != 1:
        raise RunIntegrityError("Runs must cover the same split and the same cases")
    labels = load_labels(labels_dir, splits.pop()) if labels_dir else {}
    by_case: dict[str, list[CaseResult]] = {}
    for _, results in loaded:
        for result in results:
            by_case.setdefault(result.case_id, []).append(result)

    per_case: list[CaseConsistency] = []
    for case_id, results in sorted(by_case.items()):
        keys = [outcome_key(r) for r in results]
        modal = max(keys.count(k) for k in set(keys))
        overlaps = [_jaccard(evidence_key(a), evidence_key(b)) for a, b in combinations(results, 2)]
        label = labels.get(case_id)
        correct = sum(r.prediction == label.expected_outcome for r in results) if label else None
        per_case.append(
            CaseConsistency(
                case_id=case_id,
                outcome_keys=keys,
                agreement=modal / len(keys),
                evidence_overlap=mean(overlaps) if overlaps else None,
                correct_runs=correct,
                unstable=len(set(keys)) > 1,
            )
        )
    drift = any(
        {tuple(ids) for ids in manifest.resolved_models.values()}
        != {tuple(ids) for ids in loaded[0][0].resolved_models.values()}
        for manifest, _ in loaded
    )
    overlaps_all = [c.evidence_overlap for c in per_case if c.evidence_overlap is not None]
    return ConsistencySummary(
        run_ids=[m.run_id for m, _ in loaded],
        n_runs=len(loaded),
        n_cases=len(per_case),
        mean_agreement=mean(c.agreement for c in per_case),
        identical_cases=sum(not c.unstable for c in per_case),
        unstable_cases=[c.case_id for c in per_case if c.unstable],
        correctness_flips=[
            c.case_id
            for c in per_case
            if c.correct_runs is not None and 0 < c.correct_runs < len(c.outcome_keys)
        ],
        mean_evidence_overlap=mean(overlaps_all) if overlaps_all else None,
        snapshot_drift=drift,
        per_case=per_case,
    )


def render(summary: ConsistencySummary) -> str:
    overlap = (
        "n/a" if summary.mean_evidence_overlap is None else f"{summary.mean_evidence_overlap:.0%}"
    )
    lines = [
        f"# Consistency across {summary.n_runs} runs ({summary.n_cases} cases)",
        "",
        f"- Mean outcome agreement: **{summary.mean_agreement:.0%}** "
        f"({summary.identical_cases}/{summary.n_cases} cases identical in every run)",
        f"- Mean overlap of cited evidence: {overlap}",
        f"- Cases that flip between right and wrong: "
        f"{', '.join(summary.correctness_flips) or 'none'}",
        f"- Unstable cases: {', '.join(summary.unstable_cases) or 'none'}",
    ]
    if summary.snapshot_drift:
        lines.append("- **Model snapshot differed between runs**: agreement is not comparable.")
    lines += ["", "| Case | Outcomes per run | Agreement | Correct runs |", "|---|---|---|---|"]
    lines += [
        f"| {c.case_id} | {', '.join(c.outcome_keys)} | {c.agreement:.0%} | "
        f"{'n/a' if c.correct_runs is None else f'{c.correct_runs}/{len(c.outcome_keys)}'} |"
        for c in summary.per_case
    ]
    return "\n".join(lines) + "\n"
