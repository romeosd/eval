"""Markdown rendering of a scored run."""

from __future__ import annotations

from adjudicator.evaluation.results import CaseScore, EvalSummary
from adjudicator.evaluation.stats import single_case_swing
from adjudicator.schemas import Outcome


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _interval(summary: EvalSummary, key: str) -> str:
    interval = summary.intervals.get(key)
    return interval.text() if interval else "n/a"


def _caveats(summary: EvalSummary) -> list[str]:
    swing = single_case_swing(summary.n_cases)
    lines = [
        "## What these numbers can and cannot support",
        "",
        f"- **{summary.n_cases} cases from {summary.n_independent_groups} independent groups.** "
        f"One flipped case moves accuracy by {_pct(swing)}; variants of one source case share "
        "its failure modes, so they add less evidence than their count suggests.",
        "- Intervals are 95% Wilson intervals: treat a difference smaller than their width as "
        "noise. A model comparison needs the paired test in `compare`, not two accuracies.",
        "- A perfect score on a small set is a smoke test, not proof of reliability.",
    ]
    if summary.partial:
        lines.append("- **Partial run** (case filter applied): not comparable with full runs.")
    if summary.model_drift:
        lines.append(
            f"- **Model snapshot drift**: {', '.join(summary.model_drift)} resolved to more than "
            "one snapshot during this run. Pin dated snapshot names to compare runs fairly."
        )
    if summary.held_out_runs > 1:
        lines.append(
            f"- **Held-out split has been run {summary.held_out_runs} times.** Any change made "
            "between those runs in response to its results makes it a development set."
        )
    return [*lines, ""]


def _headline(summary: EvalSummary) -> list[str]:
    return [
        "| Metric | Value |",
        "|---|---|",
        f"| Decision accuracy | {_interval(summary, 'decision_accuracy')} |",
        f"| Accuracy on self-decided cases | {_interval(summary, 'self_decided_accuracy')} |",
        f"| Payable amount accuracy (correct APPROVEs) | {_interval(summary, 'amount_accuracy')} |",
        f"| **Unsafe approval rate** (APPROVE when DENY/REFER expected) | "
        f"{_interval(summary, 'unsafe_approval_rate')} |",
        f"| Forced referrals (by safeguards) | {_interval(summary, 'forced_referral_rate')} |",
        f"| Infrastructure failures | {_interval(summary, 'failure_rate')} |",
        f"| Passed validation on first attempt | {_interval(summary, 'first_attempt_pass_rate')} |",
        f"| Mean expected-section recall | {_pct(summary.mean_section_recall)} |",
        f"| Cost total / per case | ${summary.total_cost_usd:.4f} / ${summary.mean_cost_usd:.4f} |",
        "| Cost per correct decision | "
        + (f"${summary.cost_per_correct_usd:.4f}" if summary.cost_per_correct_usd else "n/a")
        + " |",
        f"| Mean latency per case | {summary.mean_duration_s:.1f}s |",
        "",
    ]


def _stages(summary: EvalSummary) -> list[str]:
    lines = ["## Where misses come from", ""]
    if not summary.failure_stages:
        return [*lines, "No misses.", ""]
    lines += ["| Failure stage | Cases |", "|---|---|"]
    lines += [f"| {stage} | {n} |" for stage, n in summary.failure_stages.items()]
    lines += ["", "Validation errors by check (all attempts):", ""]
    checks = summary.check_failures
    lines.append(", ".join(f"{k}: {v}" for k, v in checks.items()) if checks else "none")
    lines += ["", "Rejected tool calls: ", ""]
    tools = summary.tool_errors
    lines.append(", ".join(f"{k}: {v}" for k, v in tools.items()) if tools else "none")
    return [*lines, ""]


def _slices(summary: EvalSummary) -> list[str]:
    lines = ["## Accuracy by slice", "", "| Slice | Value | Accuracy |", "|---|---|---|"]
    for dimension, groups in summary.slices.items():
        lines += [f"| {dimension} | {name} | {i.text()} |" for name, i in groups.items()]
    return [*lines, ""]


def render_markdown(summary: EvalSummary, scores: list[CaseScore]) -> str:
    lines = [
        f"# Evaluation: run `{summary.run_id}` (split `{summary.split}`, {summary.n_cases} cases)",
        "",
        *_headline(summary),
        *_caveats(summary),
        *_stages(summary),
        *_slices(summary),
        "## Confusion (rows = expected, columns = predicted)",
        "",
        "| expected \\ predicted | " + " | ".join([*Outcome, "NONE"]) + " |",
        "|---" * (len(Outcome) + 2) + "|",
    ]
    for expected, row in summary.confusion.items():
        lines.append(f"| {expected} | " + " | ".join(str(v) for v in row.values()) + " |")
    lines += [
        "",
        "## Case-level analysis",
        "",
        "| Case | Expected | Predicted | Status | Category | Stage | Sections (cited / expected) "
        "| Attempts | Cost |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scores:
        lines.append(
            f"| {s.case_id}{' (synthetic)' if s.synthetic else ''} | {s.expected} | "
            f"{s.predicted or '-'} | {s.status} | {s.category} | {s.failure_stage or '-'} | "
            f"{', '.join(s.cited_sections) or '-'} / {', '.join(s.expected_sections) or '-'} | "
            f"{s.attempts} | ${s.cost_usd:.4f} |"
        )
    return "\n".join(lines) + "\n"
