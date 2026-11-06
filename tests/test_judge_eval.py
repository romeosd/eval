import asyncio

import pytest

from adjudicator.evaluation.judge_eval import (
    CORRUPTIONS,
    run_judge_eval,
)
from adjudicator.llm.client import ModelCallError
from adjudicator.runner import execute_run
from adjudicator.schemas import SupportVerdict

from .conftest import respond
from .helpers import cap_calls, load_case, referral, theft_approval


def _agent(request):
    case_id = request.purpose.split(":")[1]
    if case_id != "dev-002":
        return respond(referral(case_id))
    if not any(item.get("type") == "function_call_output" for item in request.input):
        return respond(calls=cap_calls())
    return respond(theft_approval())


@pytest.fixture
def decided_run(settings, transport):
    transport.adjudicate = _agent
    return asyncio.run(execute_run(settings, "dev", transport=transport, tag="j"))


def _judge_eval(settings, transport, run_dir):
    return asyncio.run(run_judge_eval(settings, run_dir, transport=transport))


def test_judge_that_accepts_everything_is_caught(settings, transport, decided_run):
    summary = _judge_eval(settings, transport, decided_run)  # default fake judge: supported
    assert summary.false_accept_rate == 1.0 and summary.false_reject_rate == 0.0
    assert summary.by_kind["control"].n == 2  # two findings in the one decided case
    assert (decided_run / "judge_eval.md").is_file() and (decided_run / "judge_eval.json").is_file()


def test_judge_that_rejects_everything_is_caught(settings, transport, decided_run):
    transport.overrides.append(
        ("claim_support", lambda _r: respond(SupportVerdict(supported=False, explanation="no")))
    )
    summary = _judge_eval(settings, transport, decided_run)
    assert summary.false_accept_rate == 0.0 and summary.false_reject_rate == 1.0


def test_judge_outage_is_reported_not_counted_as_a_verdict(settings, transport, decided_run):
    def down(_request):
        raise ModelCallError("down")

    transport.overrides.append(("claim_support", down))
    summary = _judge_eval(settings, transport, decided_run)
    assert summary.error_rate == 1.0
    assert summary.false_accept_rate is None and summary.false_reject_rate is None


def test_corruptions_change_what_they_claim_to(services, context):
    case = load_case(services, context, "dev-002")
    finding = theft_approval().findings[1]  # cites section 3.1 and the $1,200 receipt

    amount = CORRUPTIONS["wrong_amount"](finding, 0, case, services.index)
    assert amount is not None and "$2,400.00" in amount.statement  # first amount doubled
    assert "$1,200" not in amount.statement

    policy = CORRUPTIONS["wrong_policy_quote"](finding, 0, case, services.index)
    assert policy is not None
    assert policy.policy_citations[0].section_id != "travel-policy:3.1"

    first = theft_approval().findings[0]  # cites the police report; says "within 24 hours"
    evidence = CORRUPTIONS["unrelated_evidence"](first, 0, case, services.index)
    assert evidence is not None
    assert evidence.evidence_citations[0].document_id == "receipt_laptop"
    hours = CORRUPTIONS["wrong_amount"](first, 0, case, services.index)
    assert hours is not None and "within 48 hours" in hours.statement  # any number is perturbed
    no_number = first.model_copy(update={"statement": "The police report was timely."})
    assert CORRUPTIONS["wrong_amount"](no_number, 0, case, services.index) is None
