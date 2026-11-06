"""Diagnostics that explain *why* a run scored as it did.

Failure-stage attribution answers "which stage should I fix?" instead of "what was the
accuracy?". Stages are assigned in a fixed order so every miss has exactly one cause.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from adjudicator.evaluation.results import CaseScore
from adjudicator.evaluation.stats import Interval, wilson
from adjudicator.schemas import (
    CaseLabel,
    CaseResult,
    CaseStatus,
    Outcome,
    PolicyIndex,
    RunManifest,
)

STAGES = (
    "infrastructure",  # API or budget failure: no decision was attempted
    "retrieval",  # an expected section was never read
    "diagram",  # the section was read but its diagram page was never inspected
    "over_referral",  # safeguards referred a case a human was not expected to see
    "unsafe_approval",  # APPROVE where DENY or REFER was expected, after validation passed
    "amount",  # right outcome, wrong payable amount
    "reasoning",  # evidence was read; the conclusion is still wrong
)


def expected_section_ids(index: PolicyIndex | None, label: CaseLabel) -> list[str]:
    if index is None:
        return []
    ids: list[str] = []
    for number in label.expected_sections:
        ids.extend(
            s.section_id
            for s in index.all_sections()
            if s.number == number or s.number.startswith(f"{number}.")
        )
    return sorted(set(ids))


def diagram_pages(index: PolicyIndex, section_id: str) -> list[str]:
    section = index.section(section_id)
    if section is None:
        return []
    return [
        f"{p.document_id}:{p.page_number}"
        for p in index.all_pages()
        if p.document_id == section.document_id
        and section.page_start <= p.page_number <= section.page_end
        and p.diagram is not None
        and p.diagram.has_visual_content
    ]


def needs_diagram(index: PolicyIndex | None, label: CaseLabel) -> bool | None:
    if index is None:
        return None
    return any(diagram_pages(index, sid) for sid in expected_section_ids(index, label))


def failure_stage(
    result: CaseResult,
    label: CaseLabel,
    *,
    outcome_correct: bool,
    amount_correct: bool | None,
    index: PolicyIndex | None,
    retrieval_mode: str,
) -> str | None:
    if result.status is CaseStatus.FAILED:
        return "infrastructure"
    if outcome_correct:
        return "amount" if amount_correct is False else None
    if index is not None:
        read, viewed = set(result.policy_sections_read), set(result.policy_pages_viewed)
        expected = expected_section_ids(index, label)
        if retrieval_mode != "full_context" and any(sid not in read for sid in expected):
            return "retrieval"
        if retrieval_mode != "full_context" and any(
            page not in viewed for sid in expected for page in diagram_pages(index, sid)
        ):
            return "diagram"
    if result.status is CaseStatus.REFERRED and label.expected_outcome is not Outcome.REFER:
        return "over_referral"
    if result.prediction is Outcome.APPROVE and label.expected_outcome is not Outcome.APPROVE:
        return "unsafe_approval"
    return "reasoning"


def check_failure_histogram(results: list[CaseResult]) -> dict[str, int]:
    """Validation errors by check, over every attempt (including ones that later recovered)."""
    counts = Counter(
        issue.check for r in results for report in r.validation_history for issue in report.errors()
    )
    return dict(sorted(counts.items()))


def tool_error_counts(run_dir: Path) -> dict[str, int]:
    """Rejected tool calls from the audit logs, e.g. calculation inputs failing provenance."""
    counts: Counter[str] = Counter()
    for audit in sorted((run_dir / "cases").glob("*/audit.jsonl")):
        for line in audit.read_text("utf-8").splitlines():
            event = json.loads(line)
            if event.get("kind") == "tool_call" and event.get("is_error"):
                counts[str(event["tool"])] += 1
    return dict(sorted(counts.items()))


def count_runs(runs_dir: Path, split: str) -> int:
    count = 0
    for path in runs_dir.glob("*/manifest.json"):
        if RunManifest.model_validate_json(path.read_text("utf-8")).split == split:
            count += 1
    return count


def model_drift(manifest: RunManifest) -> list[str]:
    return sorted(name for name, ids in manifest.resolved_models.items() if len(ids) > 1)


def _grouped(scores: list[CaseScore], label_of: Callable[[CaseScore], str]) -> dict[str, Interval]:
    groups: dict[str, list[bool]] = {}
    for score in scores:
        groups.setdefault(label_of(score), []).append(score.outcome_correct)
    return {name: wilson(sum(flags), len(flags)) for name, flags in sorted(groups.items())}


def _diagram_label(score: CaseScore) -> str:
    if score.needs_diagram is None:
        return "unknown"
    return "needed" if score.needs_diagram else "not needed"


def slice_stats(scores: list[CaseScore]) -> dict[str, dict[str, Interval]]:
    """Correctness by case property, so a weak spot is visible rather than averaged away."""
    return {
        "expected_outcome": _grouped(scores, lambda s: s.expected.value),
        "origin": _grouped(
            scores, lambda s: s.case_id.split("__", 1)[1] if "__" in s.case_id else "original"
        ),
        "injection": _grouped(scores, lambda s: "present" if s.injection_present else "absent"),
        "diagram_rule": _grouped(scores, _diagram_label),
    }
