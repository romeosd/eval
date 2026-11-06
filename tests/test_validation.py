import asyncio

from adjudicator.schemas import (
    EvidenceCitation,
    Outcome,
    PolicyCitation,
    SupportVerdict,
    ValidationReport,
)
from adjudicator.security import REDACTION_MARKER
from adjudicator.validation import CLAIM_SUPPORT

from .conftest import build_services, respond
from .helpers import cap_calculation, load_case, referral, theft_approval, toolbox_for


def _validate(services, context, decision, case_id="dev-002"):
    case = load_case(services, context, case_id)
    toolbox = toolbox_for(services, case, context)
    if case_id == "dev-002":  # the only sample case whose evidence supports the $500 cap
        cap_calculation(toolbox)
    return asyncio.run(services.validator.validate(decision, case, toolbox, context, attempt=1))


def _checks_failed(report):
    return {issue.check for issue in report.errors()}


def test_well_grounded_decision_passes(services, context):
    report = _validate(services, context, theft_approval())
    assert report.passed, report.errors()
    assert CLAIM_SUPPORT in report.checks_run


def test_fabricated_policy_quote_fails(services, context):
    decision = theft_approval()
    decision.findings[0].policy_citations[0] = PolicyCitation(
        section_id="travel-policy:5.3", page=4, quote="Police reports may be filed at any time"
    )
    assert "policy_citations" in _checks_failed(_validate(services, context, decision))


def test_page_outside_section_fails(services, context):
    decision = theft_approval()
    decision.findings[1].policy_citations[0].page = 3
    assert "policy_citations" in _checks_failed(_validate(services, context, decision))


def test_wrong_case_id_fails(services, context):
    assert "task_identity" in _checks_failed(
        _validate(services, context, theft_approval("dev-001"))
    )


def test_amount_without_provenance_fails(services, context):
    decision = theft_approval(calc_id="calc-99")
    decision.payable_amount = 450.0
    assert "calculation_provenance" in _checks_failed(_validate(services, context, decision))


def test_citing_quarantined_content_fails(services, context):
    decision = referral("dev-006")
    decision.findings = theft_approval().findings[:1]
    decision.findings[0].evidence_citations = [
        EvidenceCitation(document_id="doctor_letter", quote=REDACTION_MARKER)
    ]
    report = _validate(services, context, decision, case_id="dev-006")
    assert "quarantine" in _checks_failed(report)


def test_incomplete_evidence_forbids_non_referral(services, context):
    decision = theft_approval("dev-007")
    decision.outcome = Outcome.DENY
    decision.payable_amount = None
    assert "completeness" in _checks_failed(_validate(services, context, decision, "dev-007"))


def test_judge_rejection_blocks_decision(settings, transport, context):
    transport.overrides.append(
        (
            "claim_support",
            lambda _r: respond(SupportVerdict(supported=False, explanation="quote unrelated")),
        )
    )
    services = build_services(settings, transport)
    assert CLAIM_SUPPORT in _checks_failed(_validate(services, context, theft_approval()))


def test_judge_timeout_fails_closed(settings, transport, context):
    fast = settings.model_copy(
        update={"validation": settings.validation.model_copy(update={"judge_timeout_s": 0.05})}
    )
    transport.delay_s["claim_support"] = 0.5
    services = build_services(fast, transport)
    report = _validate(services, context, theft_approval())
    assert not report.passed
    assert any("judge unavailable" in i.message for i in report.errors())


def test_required_checks_must_all_run(services, context):
    report = _validate(services, context, theft_approval())
    assert report.passed
    tampered = report.model_dump()
    tampered["checks_run"].remove(CLAIM_SUPPORT)
    tampered["passed"] = True  # a stored verdict is never trusted; it is recomputed
    assert not ValidationReport.model_validate(tampered).passed


def _judge_requests(transport):
    return [r for r in transport.requests if r.purpose.startswith("claim_support")]


def _image_count(request):
    content = request.input[0]["content"]
    return sum(part["type"] == "input_image" for part in content)


def test_judge_receives_page_image_only_for_diagram_citations(settings, transport, context):
    services = build_services(settings, transport)
    _validate(services, context, theft_approval())
    by_finding = {r.purpose.rsplit(":", 1)[-1]: _image_count(r) for r in _judge_requests(transport)}
    assert by_finding == {"f0": 1, "f1": 0}  # f0 cites the flowchart page, f1 cites a text page


def test_judge_images_can_be_disabled(settings, transport, context):
    off = settings.model_copy(
        update={
            "validation": settings.validation.model_copy(
                update={"judge_sees_diagram_images": False}
            )
        }
    )
    _validate(build_services(off, transport), context, theft_approval())
    assert all(_image_count(r) == 0 for r in _judge_requests(transport))


def test_judge_outage_is_classed_as_infrastructure(settings, transport, context):
    fast = settings.model_copy(
        update={"validation": settings.validation.model_copy(update={"judge_timeout_s": 0.05})}
    )
    transport.delay_s["claim_support"] = 0.5
    report = _validate(build_services(fast, transport), context, theft_approval())
    assert report.errors() and {i.kind for i in report.errors()} == {"infrastructure"}


def test_unsupported_verdict_is_a_decision_error(settings, transport, context):
    transport.overrides.append(
        ("claim_support", lambda _r: respond(SupportVerdict(supported=False, explanation="no")))
    )
    report = _validate(build_services(settings, transport), context, theft_approval())
    assert {i.kind for i in report.errors()} == {"decision"}
