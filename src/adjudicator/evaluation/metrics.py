"""Score a finished run against labels, with case-level error analysis.

Integrity rules: the set of labelled cases must equal the set of cases in the run
(no silently skipped or extra cases), and only the run's own result files are read.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean

from adjudicator.evaluation.diagnostics import (
    check_failure_histogram,
    count_runs,
    failure_stage,
    model_drift,
    needs_diagram,
    slice_stats,
    tool_error_counts,
)
from adjudicator.evaluation.report import render_markdown
from adjudicator.evaluation.results import CaseScore, ClassMetrics, EvalSummary
from adjudicator.evaluation.stats import wilson
from adjudicator.runner import RunIntegrityError, load_run
from adjudicator.schemas import (
    CaseLabel,
    CaseResult,
    CaseStatus,
    Outcome,
    PolicyIndex,
    RunManifest,
)

__all__ = [
    "CaseScore",
    "ClassMetrics",
    "EvalSummary",
    "load_labels",
    "render_markdown",
    "score_case",
    "score_run",
    "summarise",
    "write_report",
]


def load_labels(labels_dir: Path, split: str) -> dict[str, CaseLabel]:
    directory = labels_dir / split
    labels = [CaseLabel.model_validate_json(p.read_text("utf-8")) for p in directory.glob("*.json")]
    return {label.case_id: label for label in labels}


def score_run(run_dir: Path, labels_dir: Path) -> tuple[EvalSummary, list[CaseScore]]:
    manifest, results = load_run(run_dir)
    labels = load_labels(labels_dir, manifest.split)
    if manifest.case_filter is not None:
        labels = {k: v for k, v in labels.items() if k in manifest.case_filter}
    run_ids, label_ids = {r.case_id for r in results}, set(labels)
    if run_ids != label_ids:
        raise RunIntegrityError(
            f"Label/case mismatch for split {manifest.split!r}: "
            f"unlabelled={sorted(run_ids - label_ids)} "
            f"missing_from_run={sorted(label_ids - run_ids)}"
        )
    index_path = run_dir / "policy_index.json"
    index = (
        PolicyIndex.model_validate_json(index_path.read_text("utf-8"))
        if index_path.is_file()
        else None
    )
    mode = _manifest_setting(manifest, "retrieval", "mode", "tools")
    scores = [
        score_case(r, labels[r.case_id], index=index, retrieval_mode=mode)
        for r in sorted(results, key=lambda r: r.case_id)
    ]
    summary = summarise(manifest.run_id, manifest.split, results, scores)
    held_out = _manifest_setting(manifest, "eval", "held_out_split", "test")
    return summary.model_copy(
        update={
            "partial": manifest.case_filter is not None,
            "check_failures": check_failure_histogram(results),
            "tool_errors": tool_error_counts(run_dir),
            "held_out_runs": count_runs(run_dir.parent, held_out),
            "model_drift": model_drift(manifest),
        }
    ), scores


def _manifest_setting(manifest: RunManifest, section: str, key: str, default: str) -> str:
    value = manifest.settings.get(section)
    if isinstance(value, dict):
        item = value.get(key)
        if isinstance(item, str):
            return item
    return default


def score_case(
    result: CaseResult,
    label: CaseLabel,
    *,
    index: PolicyIndex | None = None,
    retrieval_mode: str = "tools",
) -> CaseScore:
    outcome_correct = result.prediction == label.expected_outcome
    amount_correct: bool | None = None
    if label.expected_outcome is Outcome.APPROVE and outcome_correct:
        amount_correct = (
            result.payable_amount is not None
            and label.expected_payable_amount is not None
            and abs(result.payable_amount - label.expected_payable_amount) <= 0.01
        )
    cited = sorted(
        {
            c.section_id.split(":", 1)[-1]
            for f in (result.decision.findings if result.decision else [])
            for c in f.policy_citations
        }
    )
    recall: float | None = None
    if label.expected_sections:
        hit = sum(
            any(c == e or c.startswith(f"{e}.") for c in cited) for e in label.expected_sections
        )
        recall = hit / len(label.expected_sections)
    return CaseScore(
        case_id=result.case_id,
        expected=label.expected_outcome,
        predicted=result.prediction,
        status=result.status,
        outcome_correct=outcome_correct,
        amount_correct=amount_correct,
        expected_sections=label.expected_sections,
        cited_sections=cited,
        section_recall=recall,
        category=_categorise(result, label, outcome_correct, amount_correct, recall),
        failure_stage=failure_stage(
            result,
            label,
            outcome_correct=outcome_correct,
            amount_correct=amount_correct,
            index=index,
            retrieval_mode=retrieval_mode,
        ),
        attempts=len(result.validation_history),
        cost_usd=result.cost_usd,
        synthetic=label.synthetic,
        source_group=label.source_case_id or label.case_id,
        injection_present=bool(result.injection_findings),
        needs_diagram=needs_diagram(index, label),
        notes=label.notes,
    )


def _categorise(
    result: CaseResult,
    label: CaseLabel,
    outcome_correct: bool,
    amount_correct: bool | None,
    recall: float | None,
) -> str:
    if result.status is CaseStatus.FAILED:
        return "infrastructure_failure"
    if outcome_correct:
        if amount_correct is False:
            return "wrong_amount"
        if recall is not None and recall < 1 and result.status is CaseStatus.DECIDED:
            return "correct_weak_grounding"
        return "correct"
    if result.prediction is Outcome.APPROVE:
        return "unsafe_approval"
    if result.status is CaseStatus.REFERRED:
        return "forced_referral"
    if label.expected_outcome is Outcome.REFER:
        return "missed_referral"
    return "wrong_outcome"


def summarise(
    run_id: str, split: str, results: list[CaseResult], scores: list[CaseScore]
) -> EvalSummary:
    n = len(scores)
    amount_scores = [s.amount_correct for s in scores if s.amount_correct is not None]
    recalls = [s.section_recall for s in scores if s.section_recall is not None]
    confusion: dict[str, dict[str, int]] = {
        e.value: {p: 0 for p in [*Outcome, "NONE"]} for e in Outcome
    }
    for s in scores:
        confusion[s.expected.value][s.predicted.value if s.predicted else "NONE"] += 1
    per_class: dict[str, ClassMetrics] = {}
    for outcome in Outcome:
        tp = sum(s.expected is outcome and s.predicted is outcome for s in scores)
        predicted = sum(s.predicted is outcome for s in scores)
        support = sum(s.expected is outcome for s in scores)
        per_class[outcome.value] = ClassMetrics(
            precision=tp / predicted if predicted else None,
            recall=tp / support if support else None,
            support=support,
        )
    not_approvable = [s for s in scores if s.expected is not Outcome.APPROVE]
    decided = [s for s in scores if s.status is CaseStatus.DECIDED]
    correct = sum(s.outcome_correct for s in scores)
    total_cost = sum(r.cost_usd for r in results)
    unsafe = sum(s.predicted is Outcome.APPROVE for s in not_approvable)
    intervals = {
        "decision_accuracy": wilson(correct, n),
        "self_decided_accuracy": wilson(sum(s.outcome_correct for s in decided), len(decided)),
        "amount_accuracy": wilson(sum(amount_scores), len(amount_scores)),
        "unsafe_approval_rate": wilson(unsafe, len(not_approvable)),
        "forced_referral_rate": wilson(sum(r.status is CaseStatus.REFERRED for r in results), n),
        "failure_rate": wilson(sum(r.status is CaseStatus.FAILED for r in results), n),
        "first_attempt_pass_rate": wilson(
            sum(bool(r.validation_history) and r.validation_history[0].passed for r in results), n
        ),
    }
    return EvalSummary(
        run_id=run_id,
        split=split,
        n_cases=n,
        n_independent_groups=len({s.source_group for s in scores}),
        intervals=intervals,
        slices=slice_stats(scores),
        failure_stages=dict(
            sorted(Counter(s.failure_stage for s in scores if s.failure_stage).items())
        ),
        check_failures={},
        tool_errors={},
        held_out_runs=0,
        model_drift=[],
        decision_accuracy=correct / n,
        self_decided_accuracy=(
            sum(s.outcome_correct for s in decided) / len(decided) if decided else None
        ),
        amount_accuracy=(sum(amount_scores) / len(amount_scores)) if amount_scores else None,
        mean_section_recall=mean(recalls) if recalls else None,
        unsafe_approval_rate=(
            sum(s.predicted is Outcome.APPROVE for s in not_approvable) / len(not_approvable)
            if not_approvable
            else 0.0
        ),
        referral_rate=sum(s.predicted is Outcome.REFER for s in scores) / n,
        forced_referral_rate=sum(r.status is CaseStatus.REFERRED for r in results) / n,
        failure_rate=sum(r.status is CaseStatus.FAILED for r in results) / n,
        first_attempt_pass_rate=sum(
            bool(r.validation_history) and r.validation_history[0].passed for r in results
        )
        / n,
        total_cost_usd=round(total_cost, 4),
        cost_per_correct_usd=round(total_cost / correct, 4) if correct else None,
        mean_cost_usd=round(mean(r.cost_usd for r in results), 4),
        mean_duration_s=round(mean(r.duration_s for r in results), 2),
        per_class=per_class,
        confusion=confusion,
        categories=dict(Counter(s.category for s in scores)),
    )


def write_report(run_dir: Path, labels_dir: Path) -> EvalSummary:
    summary, scores = score_run(run_dir, labels_dir)
    (run_dir / "eval_summary.json").write_text(summary.model_dump_json(indent=2), "utf-8")
    (run_dir / "eval_cases.json").write_text(
        "[" + ",".join(s.model_dump_json() for s in scores) + "]", "utf-8"
    )
    (run_dir / "eval_report.md").write_text(render_markdown(summary, scores), "utf-8")
    return summary
