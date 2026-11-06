import asyncio
import json

import pytest

from adjudicator.evaluation.diagnostics import (
    check_failure_histogram,
    expected_section_ids,
    failure_stage,
    needs_diagram,
    tool_error_counts,
)
from adjudicator.evaluation.metrics import score_run, write_report
from adjudicator.evaluation.variants import generate_variants
from adjudicator.llm.client import FunctionCall
from adjudicator.runner import execute_run, load_run
from adjudicator.schemas import CaseLabel, CaseResult, CaseStatus, Outcome

from .conftest import respond
from .helpers import referral, theft_approval


def _result(status=CaseStatus.DECIDED, prediction=Outcome.REFER, **overrides) -> CaseResult:
    fields = {
        "run_id": "r",
        "case_id": "c",
        "status": status,
        "prediction": prediction,
        "payable_amount": None,
        "decision": None,
        "rejected_candidates": [],
        "validation_history": [],
        "referral_reasons": [],
        "injection_findings": [],
        "usage": [],
        "calculations": [],
        "policy_sections_read": [],
        "policy_pages_viewed": [],
        "error": None,
        "duration_s": 0.0,
        **overrides,
    }
    return CaseResult(**fields)


def _label(outcome=Outcome.DENY, sections=("5.3",)) -> CaseLabel:
    return CaseLabel(
        case_id="c",
        expected_outcome=outcome,
        expected_payable_amount=None,
        expected_sections=list(sections),
        notes="",
    )


def _stage(index, result, label, *, mode="tools", correct=False, amount=None):
    return failure_stage(
        result,
        label,
        outcome_correct=correct,
        amount_correct=amount,
        index=index,
        retrieval_mode=mode,
    )


# ------------------------------------------------------------------ stage attribution


def test_expected_sections_resolve_to_policy_ids_including_children(services):
    assert expected_section_ids(services.index, _label(sections=("5.3",))) == ["travel-policy:5.3"]
    assert "travel-policy:5.1" in expected_section_ids(services.index, _label(sections=("5",)))
    assert needs_diagram(services.index, _label(sections=("5.3",))) is True
    assert needs_diagram(services.index, _label(sections=("3.1",))) is False
    assert needs_diagram(None, _label()) is None


def test_stage_is_none_for_a_correct_case_and_amount_for_a_wrong_amount(services):
    result = _result()
    assert _stage(services.index, result, _label(Outcome.REFER, ()), correct=True) is None
    assert _stage(services.index, result, _label(), correct=True, amount=False) == "amount"


def test_stage_infrastructure_wins(services):
    failed = _result(CaseStatus.FAILED, None)
    assert _stage(services.index, failed, _label()) == "infrastructure"


def test_stage_retrieval_when_expected_section_was_never_read(services):
    assert _stage(services.index, _result(), _label()) == "retrieval"


def test_stage_diagram_when_section_read_but_diagram_page_not_inspected(services):
    read = _result(policy_sections_read=["travel-policy:5.3"])
    assert _stage(services.index, read, _label()) == "diagram"


def test_stage_over_referral_unsafe_approval_and_reasoning(services):
    inspected = {
        "policy_sections_read": ["travel-policy:5.3"],
        "policy_pages_viewed": ["travel-policy:4"],
    }
    forced = _result(CaseStatus.REFERRED, Outcome.REFER, **inspected)
    approve = _result(CaseStatus.DECIDED, Outcome.APPROVE, **inspected)
    wrong = _result(CaseStatus.DECIDED, Outcome.REFER, **inspected)
    assert _stage(services.index, forced, _label()) == "over_referral"
    assert _stage(services.index, approve, _label()) == "unsafe_approval"
    assert _stage(services.index, wrong, _label()) == "reasoning"


def test_full_context_mode_never_blames_retrieval_or_diagram(services):
    stage = _stage(services.index, _result(), _label(), mode="full_context")
    assert stage == "reasoning"


def test_without_an_index_attribution_falls_back_to_outcome_based_stages():
    assert _stage(None, _result(), _label()) == "reasoning"


# ------------------------------------------------------------------ end to end on real runs


def _reads_then_refers(case_id, calls):
    def agent(request):
        target = request.purpose.split(":")[1]
        if target == case_id and not any(
            i.get("type") == "function_call_output" for i in request.input
        ):
            return respond(calls=calls)
        return respond(referral(target))

    return agent


def _section_call(section):
    return FunctionCall(
        call_id="s", name="get_policy_section", arguments=json.dumps({"section_id": section})
    )


def _page_call():
    args = json.dumps({"document_id": "travel-policy", "page": 4})
    return FunctionCall(call_id="p", name="view_policy_page", arguments=args)


def _scored(settings, transport, agent, split="dev"):
    transport.adjudicate = agent
    run_dir = asyncio.run(execute_run(settings, split, transport=transport))
    summary, scores = score_run(run_dir, settings.paths.labels_dir)
    return run_dir, summary, {s.case_id: s for s in scores}


def test_a_model_that_never_reads_the_policy_is_blamed_on_retrieval(settings, transport):
    _, summary, scores = _scored(
        settings, transport, lambda r: respond(referral(r.purpose.split(":")[1]))
    )
    assert scores["dev-003"].failure_stage == "retrieval"
    assert scores["dev-007"].failure_stage is None  # expected REFER, correctly referred
    assert summary.failure_stages["retrieval"] == 6


def test_reading_the_section_but_not_the_diagram_is_a_diagram_miss(settings, transport):
    agent = _reads_then_refers("dev-003", [_section_call("travel-policy:5.3")])
    _, _, scores = _scored(settings, transport, agent)
    assert scores["dev-003"].failure_stage == "diagram"


def test_inspecting_the_diagram_leaves_a_reasoning_failure(settings, transport):
    agent = _reads_then_refers("dev-003", [_section_call("travel-policy:5.3"), _page_call()])
    _, _, scores = _scored(settings, transport, agent)
    assert scores["dev-003"].failure_stage == "reasoning"


# ------------------------------------------------------------------ run-level diagnostics


def test_validation_failures_are_counted_by_check(settings, transport):
    agent = lambda r: respond(theft_approval("wrong-case"))  # noqa: E731
    run_dir, summary, _ = _scored(settings, transport, agent)
    _, results = load_run(run_dir)
    histogram = check_failure_histogram(results)
    assert histogram["task_identity"] >= 3 and summary.check_failures == histogram


def test_rejected_calculation_inputs_are_counted_from_the_audit_log(settings, transport):
    bad = {
        "operation": "sum",
        "inputs": [
            {"value": 5000, "unit": "hours", "source": "policy", "ref": "travel-policy:3.1"}
        ],
        "datetimes": [],
    }
    call = FunctionCall(call_id="c", name="calculate", arguments=json.dumps(bad))
    run_dir, summary, _ = _scored(settings, transport, _reads_then_refers("dev-002", [call]))
    assert tool_error_counts(run_dir) == {"calculate": 1} and summary.tool_errors == {
        "calculate": 1
    }


def test_held_out_split_use_is_counted_and_flagged(settings, transport):
    transport.adjudicate = lambda r: respond(referral(r.purpose.split(":")[1]))
    asyncio.run(execute_run(settings, "test", transport=transport))
    second = asyncio.run(execute_run(settings, "test", transport=transport))
    summary, _ = score_run(second, settings.paths.labels_dir)
    assert summary.held_out_runs == 2
    report = write_report(second, settings.paths.labels_dir)
    assert report.held_out_runs == 2
    assert "run 2 times" in (second / "eval_report.md").read_text()


def test_variants_count_as_few_independent_groups_and_are_sliced(settings, transport):
    generate_variants(settings, "dev", "variants")
    _, summary, _ = _scored(
        settings, transport, lambda r: respond(referral(r.purpose.split(":")[1])), split="variants"
    )
    assert summary.n_cases == 27 and summary.n_independent_groups == 8
    assert {"direct_injection", "late_submission", "missing_attachment"} <= set(
        summary.slices["origin"]
    )
    assert summary.slices["injection"]["present"].n >= 1


def test_model_snapshot_drift_is_flagged(settings, transport):
    run_dir, _, _ = _scored(
        settings, transport, lambda r: respond(referral(r.purpose.split(":")[1]))
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resolved_models"]["gpt-5.5"] = ["gpt-5.5-a", "gpt-5.5-b"]
    manifest_path.write_text(json.dumps(manifest))
    summary, _ = score_run(run_dir, settings.paths.labels_dir)
    assert summary.model_drift == ["gpt-5.5"]


def test_report_states_what_the_numbers_cannot_support(settings, transport):
    run_dir, _, _ = _scored(
        settings, transport, lambda r: respond(referral(r.purpose.split(":")[1]))
    )
    write_report(run_dir, settings.paths.labels_dir)
    text = (run_dir / "eval_report.md").read_text()
    for expected in (
        "95% CI",
        "What these numbers can and cannot support",
        "Where misses come from",
        "Accuracy by slice",
        "One flipped case moves accuracy by 12%",
    ):
        assert expected in text


@pytest.mark.parametrize("field", ["intervals", "slices", "failure_stages"])
def test_summary_exposes_structured_diagnostics(settings, transport, field):
    _, summary, _ = _scored(
        settings, transport, lambda r: respond(referral(r.purpose.split(":")[1]))
    )
    assert getattr(summary, field)
