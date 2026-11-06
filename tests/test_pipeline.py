import asyncio

from adjudicator.llm.client import FunctionCall
from adjudicator.pipeline import CasePipeline
from adjudicator.schemas import CaseStatus, InjectionVerdict, Outcome

from .conftest import build_services, respond
from .helpers import cap_calls, referral, theft_approval, theft_denial


def _has_tool_output(request) -> bool:
    return any(item.get("type") == "function_call_output" for item in request.input)


def _scripted_agent(decision_factory):
    """Turn 1: compute the cap. Turn 2: return the decision."""

    def adjudicate(request):
        if not _has_tool_output(request):
            return respond(calls=cap_calls())
        return respond(decision_factory(request))

    return adjudicate


def _run(services, tmp_path, case_id, split="dev"):
    pipeline = CasePipeline(services, "run-test", tmp_path / "run")
    return asyncio.run(pipeline.run(services.settings.paths.cases_dir / split / case_id, split))


def test_valid_decision_is_decided(settings, transport, tmp_path):
    transport.adjudicate = _scripted_agent(lambda _r: theft_approval())
    result = _run(build_services(settings, transport), tmp_path, "dev-002")
    assert result.status is CaseStatus.DECIDED
    assert (result.prediction, result.payable_amount) == (Outcome.APPROVE, 500.0)
    assert result.decision is not None and result.cost_usd > 0
    assert (tmp_path / "run" / "cases" / "dev-002" / "audit.jsonl").is_file()


def test_failed_final_attempt_is_never_the_prediction(settings, transport, tmp_path):
    transport.adjudicate = _scripted_agent(lambda _r: theft_approval("wrong-case"))
    result = _run(build_services(settings, transport), tmp_path, "dev-002")
    assert result.status is CaseStatus.REFERRED
    assert result.prediction is Outcome.REFER
    assert result.decision is None and result.payable_amount is None
    assert len(result.rejected_candidates) == settings.agent.max_attempts
    assert not any(report.passed for report in result.validation_history)


def test_validation_feedback_is_sent_on_retry(settings, transport, tmp_path):
    transport.adjudicate = _scripted_agent(
        lambda r: theft_approval() if "REJECTED" in r.input[0]["content"] else theft_approval("x")
    )
    result = _run(build_services(settings, transport), tmp_path, "dev-002")
    assert result.status is CaseStatus.DECIDED
    assert len(result.validation_history) == 2


def test_budget_exhaustion_fails_without_prediction(settings, transport, tmp_path):
    services = build_services(settings, transport)
    services.llm.budget.record(settings.run.budget_usd)  # spend the whole budget
    result = _run(services, tmp_path, "dev-001")
    assert result.status is CaseStatus.FAILED and result.prediction is None


def test_unlocatable_injection_forces_referral(settings, transport, tmp_path):
    transport.adjudicate = _scripted_agent(lambda _r: referral("dev-001"))
    services = build_services(settings, transport)  # policy ingest uses the clean default
    transport.overrides.append(
        (
            "injection_screen:dev-001:form",
            lambda _r: respond(
                InjectionVerdict(contains_instructions=True, suspicious_spans=["zzz"], reason="?")
            ),
        )
    )
    result = _run(services, tmp_path, "dev-001")
    assert result.status is CaseStatus.REFERRED
    assert not transport.purposes("adjudicate")  # the decision model never saw the content


def test_turn_limit_refers(settings, transport, tmp_path):
    limited = settings.model_copy(
        update={"agent": settings.agent.model_copy(update={"max_tool_turns": 2})}
    )
    transport.adjudicate = lambda _r: respond(
        calls=[FunctionCall(call_id="l", name="list_application_documents", arguments="{}")]
    )
    result = _run(build_services(limited, transport), tmp_path, "dev-001")
    assert result.status is CaseStatus.REFERRED and "tool turns" in result.referral_reasons[0]


def test_missing_document_refers_without_calling_the_agent(settings, transport, tmp_path):
    result = _run(build_services(settings, transport), tmp_path, "dev-007")
    assert result.status is CaseStatus.REFERRED and result.prediction is Outcome.REFER
    assert "receipt_sunglasses" in result.referral_reasons[0]
    assert not transport.purposes("adjudicate")


def test_verifier_outage_revalidates_then_refers_without_rerunning_agent(
    settings, transport, tmp_path
):
    fast = settings.model_copy(
        update={"validation": settings.validation.model_copy(update={"judge_timeout_s": 0.05})}
    )
    transport.delay_s["claim_support"] = 0.5
    transport.adjudicate = _scripted_agent(lambda _r: theft_approval())
    result = _run(build_services(fast, transport), tmp_path, "dev-002")
    assert result.status is CaseStatus.REFERRED and result.decision is None
    assert result.referral_reasons[0] == "verification unavailable"
    assert not any("attempt2" in p for p in transport.purposes("adjudicate"))
    assert len(result.validation_history) == 1
    judge_rounds = len(transport.purposes("outcome_support"))
    assert judge_rounds == 1 + fast.validation.max_revalidations


def test_calculations_are_recorded_in_the_result(settings, transport, tmp_path):
    transport.adjudicate = _scripted_agent(lambda _r: theft_approval())
    result = _run(build_services(settings, transport), tmp_path, "dev-002")
    assert [c.calculation_id for c in result.calculations] == ["calc-1"]


def _consensus(settings, runs):
    return settings.model_copy(
        update={"agent": settings.agent.model_copy(update={"consensus_runs": runs})}
    )


def test_consensus_runs_that_agree_decide(settings, transport, tmp_path):
    transport.adjudicate = _scripted_agent(lambda _r: theft_approval())
    result = _run(build_services(_consensus(settings, 2), transport), tmp_path, "dev-002")
    assert result.status is CaseStatus.DECIDED and result.prediction is Outcome.APPROVE
    assert {p.split(":")[2] for p in transport.purposes("adjudicate")} == {"run0", "run1"}
    assert {report.run for report in result.validation_history} == {0, 1}


def test_consensus_runs_that_disagree_are_referred(settings, transport, tmp_path):
    approve = _scripted_agent(lambda _r: theft_approval())

    def agent(request):
        return approve(request) if ":run0:" in request.purpose else respond(theft_denial())

    transport.adjudicate = agent
    result = _run(build_services(_consensus(settings, 2), transport), tmp_path, "dev-002")
    assert result.status is CaseStatus.REFERRED and result.prediction is Outcome.REFER
    assert result.decision is None and "disagree" in result.referral_reasons[0]


def test_consensus_needs_every_run_to_validate(settings, transport, tmp_path):
    good = _scripted_agent(lambda _r: theft_approval())

    def agent(request):
        if ":run0:" in request.purpose:
            return good(request)
        return respond(theft_approval("wrong-case"))  # run 1 never passes validation

    transport.adjudicate = agent
    result = _run(build_services(_consensus(settings, 2), transport), tmp_path, "dev-002")
    assert result.status is CaseStatus.REFERRED
    assert "consensus not reached: 1 of 2" in result.referral_reasons[0]
