"""Builders for decisions used across tests."""

from __future__ import annotations

import asyncio
import json

from adjudicator.llm.client import CallContext, FunctionCall
from adjudicator.pipeline import Services
from adjudicator.schemas import (
    AgentDecision,
    Case,
    EvidenceCitation,
    Finding,
    Outcome,
    PolicyCitation,
)
from adjudicator.tools import Toolbox


def load_case(services: Services, context: CallContext, case_id: str, split: str = "dev") -> Case:
    case_dir = services.settings.paths.cases_dir / split / case_id
    return asyncio.run(services.loader.load(case_dir, split, context))


def toolbox_for(services: Services, case: Case, context: CallContext) -> Toolbox:
    return Toolbox(services.index, services.search_index, case, services.settings, context)


CAP_ARGS = {
    "operation": "min",
    "inputs": [
        {"value": 1200, "unit": "usd", "source": "evidence", "ref": "receipt_laptop"},
        {"value": 500, "unit": "usd", "source": "policy", "ref": "travel-policy:3.1"},
    ],
    "datetimes": [],
}


def cap_calls() -> list[FunctionCall]:
    """Read section 3.1, then cap the $1,200 laptop at its $500 single-item limit."""
    return [
        FunctionCall(
            call_id="r1", name="get_policy_section", arguments='{"section_id": "travel-policy:3.1"}'
        ),
        FunctionCall(call_id="c1", name="calculate", arguments=json.dumps(CAP_ARGS)),
    ]


def cap_calculation(toolbox: Toolbox) -> str:
    results = [toolbox.execute(call) for call in cap_calls()]
    assert not any(r.is_error for r in results), results
    return str(results[-1].payload["calculation_id"])


def theft_approval(case_id: str = "dev-002", calc_id: str = "calc-1") -> AgentDecision:
    return AgentDecision(
        case_id=case_id,
        outcome=Outcome.APPROVE,
        payable_amount=500.0,
        findings=[
            Finding(
                statement="The police report was filed within 24 hours of discovering the theft.",
                policy_citations=[
                    PolicyCitation(
                        section_id="travel-policy:5.3",
                        page=4,
                        quote="Police report obtained within 24 hours of discovering the theft",
                    )
                ],
                evidence_citations=[
                    EvidenceCitation(
                        document_id="police_report", quote="Report filed: 2026-04-10 21:30"
                    )
                ],
                calculation_ids=[],
            ),
            Finding(
                statement="The $1,200 laptop is capped at the $500 single-item limit.",
                policy_citations=[
                    PolicyCitation(
                        section_id="travel-policy:3.1",
                        page=2,
                        quote="no more than $500 for any single item",
                    )
                ],
                evidence_citations=[
                    EvidenceCitation(
                        document_id="receipt_laptop",
                        quote="UltraBook 14 laptop ............ $1,200.00",
                    )
                ],
                calculation_ids=[calc_id],
            ),
        ],
        conflicts_identified=[],
        missing_information=[],
        confidence="high",
        rationale="Timely police report and attended item; payment capped at the item limit.",
    )


def referral(case_id: str) -> AgentDecision:
    return AgentDecision(
        case_id=case_id,
        outcome=Outcome.REFER,
        payable_amount=None,
        findings=[],
        conflicts_identified=[],
        missing_information=["needs a human"],
        confidence="low",
        rationale="Referred.",
    )


def theft_denial(case_id: str = "dev-002") -> AgentDecision:
    """A well-cited DENY on the same evidence, used to create a disagreeing consensus run."""
    approval = theft_approval(case_id)
    findings = [f.model_copy(update={"calculation_ids": []}) for f in approval.findings]
    return approval.model_copy(
        update={"outcome": Outcome.DENY, "payable_amount": None, "findings": findings}
    )
