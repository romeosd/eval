"""Pydantic models for scored runs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from adjudicator.evaluation.stats import Interval
from adjudicator.schemas import CaseStatus, Outcome


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseScore(_Model):
    case_id: str
    expected: Outcome
    predicted: Outcome | None
    status: CaseStatus
    outcome_correct: bool
    amount_correct: bool | None
    expected_sections: list[str]
    cited_sections: list[str]
    section_recall: float | None
    category: str
    failure_stage: str | None  # where a miss originated; None when the case is correct
    attempts: int
    cost_usd: float
    synthetic: bool
    source_group: str  # case id, or the source case a variant was derived from
    injection_present: bool
    needs_diagram: bool | None  # None when the policy index is unavailable
    notes: str


class ClassMetrics(_Model):
    precision: float | None
    recall: float | None
    support: int


class EvalSummary(_Model):
    run_id: str
    split: str
    n_cases: int
    n_independent_groups: int  # variants of one source case are not independent evidence
    partial: bool = False  # True when the run covered only a filtered subset of the split
    decision_accuracy: float
    self_decided_accuracy: float | None  # accuracy where the system decided without forced referral
    amount_accuracy: float | None
    mean_section_recall: float | None
    unsafe_approval_rate: float
    referral_rate: float
    forced_referral_rate: float
    failure_rate: float
    first_attempt_pass_rate: float
    total_cost_usd: float
    mean_cost_usd: float
    mean_duration_s: float
    cost_per_correct_usd: float | None
    per_class: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]
    categories: dict[str, int]
    intervals: dict[str, Interval]  # 95% Wilson intervals for the headline rates
    slices: dict[str, dict[str, Interval]]  # correctness by case property
    failure_stages: dict[str, int]
    check_failures: dict[str, int]  # validation errors by check, across every attempt
    tool_errors: dict[str, int]  # tool calls rejected (e.g. calculation inputs without provenance)
    held_out_runs: int  # runs recorded on the held-out split so far
    model_drift: list[str]  # configured names that resolved to more than one snapshot
