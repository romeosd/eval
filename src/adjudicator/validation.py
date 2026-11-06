"""Fail-closed validation of a candidate decision.

A decision is only accepted when *every* required check ran and none reported an
error. Deterministic checks run first (cheap, exact); the LLM claim-support judge
runs last and is enabled by default. A judge error or timeout is recorded as an
unsupported finding — never as a pass.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from adjudicator.config import ImageDetail, Settings
from adjudicator.images import input_image, input_text
from adjudicator.llm.client import CallContext, LLMClient, ModelCallError, ModelRequest
from adjudicator.prompts import PromptLibrary, PromptName
from adjudicator.schemas import (
    AgentDecision,
    CalculationRecord,
    Case,
    Finding,
    Outcome,
    PolicyIndex,
    SupportVerdict,
    ValidationIssue,
    ValidationReport,
)
from adjudicator.security import REDACTION_MARKER, InjectionScreener
from adjudicator.tools import Toolbox

TASK_IDENTITY = "task_identity"
OUTCOME_CONSISTENCY = "outcome_consistency"
COMPLETENESS = "completeness"
CITATION_PRESENCE = "citation_presence"
POLICY_CITATIONS = "policy_citations"
EVIDENCE_CITATIONS = "evidence_citations"
QUARANTINE = "quarantine"
CALCULATION_PROVENANCE = "calculation_provenance"
OUTPUT_SCREEN = "output_screen"
CLAIM_SUPPORT = "claim_support"

DETERMINISTIC_CHECKS: tuple[str, ...] = (
    TASK_IDENTITY,
    OUTCOME_CONSISTENCY,
    COMPLETENESS,
    CITATION_PRESENCE,
    POLICY_CITATIONS,
    EVIDENCE_CITATIONS,
    QUARANTINE,
    CALCULATION_PROVENANCE,
    OUTPUT_SCREEN,
)

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_JUDGE_CONTEXT_CHARS = 6_000


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


class DecisionValidator:
    def __init__(
        self,
        llm: LLMClient,
        settings: Settings,
        prompts: PromptLibrary,
        index: PolicyIndex,
        screener: InjectionScreener,
    ) -> None:
        self._llm = llm
        self._config = settings.validation
        self._judge_model = settings.models.judge
        self._judge_prompt = prompts.get(PromptName.CLAIM_SUPPORT_JUDGE).text
        self._image_detail: ImageDetail = settings.retrieval.image_detail
        self._index = index
        self._screener = screener

    def required_checks(self) -> list[str]:
        checks = list(DETERMINISTIC_CHECKS)
        if self._config.claim_support_check:
            checks.append(CLAIM_SUPPORT)
        return checks

    async def validate(
        self,
        decision: AgentDecision,
        case: Case,
        toolbox: Toolbox,
        context: CallContext,
        *,
        attempt: int,
    ) -> ValidationReport:
        issues: list[ValidationIssue] = []
        checks_run: list[str] = []
        deterministic = {
            TASK_IDENTITY: self._task_identity,
            OUTCOME_CONSISTENCY: self._outcome_consistency,
            COMPLETENESS: self._completeness,
            CITATION_PRESENCE: self._citation_presence,
            POLICY_CITATIONS: self._policy_citations,
            EVIDENCE_CITATIONS: self._evidence_citations,
            QUARANTINE: self._quarantine,
            CALCULATION_PROVENANCE: self._calculation_provenance,
            OUTPUT_SCREEN: self._output_screen,
        }
        for name, check in deterministic.items():
            issues.extend(check(decision, case, toolbox))
            checks_run.append(name)

        has_errors = any(issue.severity == "error" for issue in issues)
        if self._config.claim_support_check and not has_errors:
            # Only spend judge tokens on decisions that are structurally sound.
            issues.extend(await self._claim_support(decision, case, toolbox, context))
            checks_run.append(CLAIM_SUPPORT)

        report = ValidationReport(
            attempt=attempt,
            checks_required=self.required_checks(),
            checks_run=checks_run,
            issues=issues,
        )
        context.audit.event(
            "validation",
            attempt=attempt,
            passed=report.passed,
            checks_run=checks_run,
            issues=[issue.model_dump(mode="json") for issue in issues],
        )
        return report

    # ------------------------------------------------------------------ deterministic

    def _task_identity(
        self, decision: AgentDecision, case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        if decision.case_id != case.case_id:
            return [
                _error(
                    TASK_IDENTITY,
                    f"decision is for case {decision.case_id!r} but the task is {case.case_id!r}",
                )
            ]
        return []

    def _outcome_consistency(
        self, decision: AgentDecision, _case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        amount = decision.payable_amount
        if decision.outcome is Outcome.APPROVE and (amount is None or amount <= 0):
            return [_error(OUTCOME_CONSISTENCY, "APPROVE requires a positive payable_amount")]
        if decision.outcome is not Outcome.APPROVE and amount not in (None, 0):
            return [
                _error(
                    OUTCOME_CONSISTENCY,
                    f"{decision.outcome.value} must not carry a payable_amount ({amount})",
                )
            ]
        return []

    def _completeness(
        self, decision: AgentDecision, case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        missing = case.unrecovered_documents
        if missing and decision.outcome is not Outcome.REFER:
            ids = ", ".join(d.document_id for d in missing)
            issues.append(
                _error(
                    COMPLETENESS,
                    f"documents {ids} could not be loaded; a decision other than REFER is not "
                    "permitted on incomplete evidence",
                )
            )
        if decision.missing_information and decision.outcome is Outcome.APPROVE:
            issues.append(
                _warning(COMPLETENESS, "APPROVE issued while missing_information is non-empty")
            )
        return issues

    def _citation_presence(
        self, decision: AgentDecision, _case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if decision.outcome is not Outcome.REFER:
            if not decision.findings:
                issues.append(_error(CITATION_PRESENCE, "APPROVE/DENY requires findings"))
            if not any(f.policy_citations for f in decision.findings):
                issues.append(_error(CITATION_PRESENCE, "APPROVE/DENY requires a policy citation"))
            if not any(f.evidence_citations for f in decision.findings):
                issues.append(
                    _error(CITATION_PRESENCE, "APPROVE/DENY requires an application citation")
                )
        for i, finding in enumerate(decision.findings):
            if not finding.policy_citations and not finding.evidence_citations:
                issues.append(_error(CITATION_PRESENCE, "finding has no citations", i))
        return issues

    def _policy_citations(
        self, decision: AgentDecision, _case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for i, finding in enumerate(decision.findings):
            for citation in finding.policy_citations:
                section = self._index.section(citation.section_id)
                if section is None:
                    issues.append(
                        _error(POLICY_CITATIONS, f"unknown section {citation.section_id!r}", i)
                    )
                    continue
                if not section.page_start <= citation.page <= section.page_end:
                    issues.append(
                        _error(
                            POLICY_CITATIONS,
                            f"page {citation.page} is outside section {section.section_id} "
                            f"(pages {section.page_start}-{section.page_end})",
                            i,
                        )
                    )
                    continue
                page = self._index.page(section.document_id, citation.page)
                source = page.searchable_text() if page else section.text
                issue = self._quote_issue(POLICY_CITATIONS, citation.quote, source, i)
                if issue:
                    issues.append(issue)
        return issues

    def _evidence_citations(
        self, decision: AgentDecision, case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for i, finding in enumerate(decision.findings):
            for citation in finding.evidence_citations:
                document = case.document(citation.document_id)
                if document is None or document.load_status == "failed":
                    issues.append(
                        _error(
                            EVIDENCE_CITATIONS,
                            f"cites unavailable document {citation.document_id!r}",
                            i,
                        )
                    )
                    continue
                issue = self._quote_issue(EVIDENCE_CITATIONS, citation.quote, document.text, i)
                if issue:
                    issues.append(issue)
        return issues

    def _quarantine(
        self, decision: AgentDecision, case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        spans = [f.span for f in case.injection_findings if f.span]
        spans += [s for p in self._index.all_pages() for s in p.quarantined_spans]
        issues: list[ValidationIssue] = []
        for i, finding in enumerate(decision.findings):
            quotes = [c.quote for c in finding.policy_citations] + [
                c.quote for c in finding.evidence_citations
            ]
            for quote in quotes:
                if REDACTION_MARKER.lower() in quote.lower() or any(
                    fuzz.partial_ratio(normalise(quote), normalise(span)) >= 90 for span in spans
                ):
                    issues.append(_error(QUARANTINE, "finding cites quarantined content", i))
        return issues

    def _calculation_provenance(
        self, decision: AgentDecision, case: Case, toolbox: Toolbox
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        referenced: list[float] = []
        for i, finding in enumerate(decision.findings):
            for calc_id in finding.calculation_ids:
                record = toolbox.calculations.get(calc_id)
                if record is None:
                    issues.append(_error(CALCULATION_PROVENANCE, f"unknown {calc_id!r}", i))
                elif record.unit == "usd":
                    referenced.append(record.result)  # only dollar results can be a payable amount
        issues.extend(self._unverified_dates(toolbox, case))
        amount = decision.payable_amount
        if decision.outcome is Outcome.APPROVE and amount is not None:
            quoted = [
                n
                for f in decision.findings
                for c in f.evidence_citations
                for n in _numbers(c.quote)
            ]
            tolerance = self._config.amount_tolerance
            if not any(abs(amount - v) <= tolerance for v in (*referenced, *quoted)):
                issues.append(
                    _error(
                        CALCULATION_PROVENANCE,
                        f"payable_amount {amount} is neither a referenced dollar calculation "
                        "result nor an amount quoted from the application evidence",
                    )
                )
        return issues

    def _unverified_dates(self, toolbox: Toolbox, case: Case) -> list[ValidationIssue]:
        """Warn (never fail) when a calculation date does not appear verbatim in the evidence.

        Date formats vary across documents ("10 April 2026"), so an exact-match failure would
        punish correct decisions; the warning is for human review of the audit trail.
        """
        corpus = "\n".join(d.text for d in case.documents)
        return [
            _warning(
                CALCULATION_PROVENANCE,
                f"{record.calculation_id}: date {moment[:10]} not found verbatim in the evidence",
            )
            for record in toolbox.calculations.values()
            for moment in record.datetimes
            if moment[:10] not in corpus
        ]

    def _output_screen(
        self, decision: AgentDecision, _case: Case, _toolbox: Toolbox
    ) -> list[ValidationIssue]:
        text = "\n".join([decision.rationale, *(f.statement for f in decision.findings)])
        findings = self._screener.heuristic_findings(text, "decision_output")
        return [
            _error(OUTPUT_SCREEN, f"decision text contains instruction-like content: {f.span!r}")
            for f in findings
        ]

    def _quote_issue(
        self, check: str, quote: str, source: str, index: int
    ) -> ValidationIssue | None:
        if len(quote.strip()) < self._config.min_quote_chars:
            return _error(check, f"quote {quote!r} is too short to verify", index)
        score = fuzz.partial_ratio(normalise(quote), normalise(source))
        if score < self._config.citation_min_similarity:
            return _error(
                check, f"quote not found in cited source (similarity {score:.0f}): {quote!r}", index
            )
        return None

    # ------------------------------------------------------------------ claim support

    async def _claim_support(
        self, decision: AgentDecision, case: Case, toolbox: Toolbox, context: CallContext
    ) -> list[ValidationIssue]:
        jobs = [
            self.judge_finding(finding, case, toolbox.calculations, context, i)
            for i, finding in enumerate(decision.findings)
        ]
        jobs.append(
            self._judge(
                [input_text(self._outcome_dossier(decision))],
                f"outcome_support:{case.case_id}",
                context,
                None,
            )
        )
        results = await asyncio.gather(*jobs)
        return [issue for issue in results if issue is not None]

    async def judge_finding(
        self,
        finding: Finding,
        case: Case,
        calculations: Mapping[str, CalculationRecord],
        context: CallContext,
        index: int | None,
    ) -> ValidationIssue | None:
        """Ask the independent judge whether one finding is supported by its own citations."""
        dossier, images = self._finding_dossier(finding, case, calculations)
        content = [input_text(dossier)]
        content.extend(input_image(path, self._image_detail) for path in images)
        return await self._judge(content, f"claim_support:{case.case_id}:f{index}", context, index)

    async def _judge(
        self,
        content: list[dict[str, Any]],
        purpose: str,
        context: CallContext,
        index: int | None,
    ) -> ValidationIssue | None:
        request = ModelRequest(
            model=self._judge_model,
            purpose=purpose,
            instructions=self._judge_prompt,
            input=[{"role": "user", "content": content}],
            text_format=SupportVerdict,
            timeout_s=self._config.judge_timeout_s,
        )
        try:
            response = await self._llm.call(request, context)
        except ModelCallError as exc:
            return _infrastructure_error(f"judge unavailable, treated as unsupported: {exc}", index)
        verdict = response.parsed
        if verdict is None:
            return _infrastructure_error("judge returned no verdict, treated as unsupported", index)
        if not verdict.supported:
            return _error(
                CLAIM_SUPPORT, f"not supported by its citations: {verdict.explanation}", index
            )
        return None

    def _finding_dossier(
        self, finding: Finding, case: Case, calculations: Mapping[str, CalculationRecord]
    ) -> tuple[str, list[Path]]:
        policy_sources: list[str] = []
        images: dict[Path, None] = {}
        for citation in finding.policy_citations:
            section = self._index.section(citation.section_id)
            page = self._index.page(section.document_id, citation.page) if section else None
            body = page.searchable_text() if page else (section.text if section else "")
            has_diagram = page is not None and page.diagram is not None
            if page is not None and has_diagram and self._config.judge_sees_diagram_images:
                images[page.image_path] = None
            policy_sources.append(
                f"<policy section={citation.section_id!r} page={citation.page}"
                f"{' page_image_attached=true' if page and page.image_path in images else ''}>\n"
                f"{body[:_JUDGE_CONTEXT_CHARS]}\n</policy>"
            )
        evidence_sources = [
            f"<untrusted_evidence document={c.document_id!r}>\n"
            f"{_document_text(case, c.document_id)[:_JUDGE_CONTEXT_CHARS]}\n</untrusted_evidence>"
            for c in finding.evidence_citations
        ]
        used = [
            calculations[c].model_dump_json() for c in finding.calculation_ids if c in calculations
        ]
        dossier = (
            f"<claim>\n{finding.statement}\n</claim>\n\n"
            f"<cited_quotes>\n{json.dumps(finding.model_dump(mode='json'), indent=2)}\n"
            "</cited_quotes>\n\n"
            + "\n\n".join(policy_sources + evidence_sources)
            + "\n\n<calculations>\n"
            + ("\n".join(used) or "none")
            + "\n</calculations>"
        )
        return dossier, list(images)

    def _outcome_dossier(self, decision: AgentDecision) -> str:
        statements = "\n".join(f"{i}. {f.statement}" for i, f in enumerate(decision.findings))
        return (
            "<claim>\nGiven ONLY the findings below (assume each is true), the outcome "
            f"{decision.outcome.value} with payable_amount {decision.payable_amount} follows, "
            "and no finding points to a different outcome.\n</claim>\n\n"
            f"<findings>\n{statements}\n</findings>\n\n"
            f"<conflicts_identified>\n{decision.conflicts_identified}\n</conflicts_identified>\n"
            f"<missing_information>\n{decision.missing_information}\n</missing_information>"
        )


def _document_text(case: Case, document_id: str) -> str:
    document = case.document(document_id)
    return document.text if document else ""


def _numbers(text: str) -> Iterable[float]:
    for match in _NUMBER.findall(text):
        try:
            yield float(match.replace(",", ""))
        except ValueError:
            continue


def _error(check: str, message: str, index: int | None = None) -> ValidationIssue:
    return ValidationIssue(check=check, severity="error", message=message, finding_index=index)


def _infrastructure_error(message: str, index: int | None = None) -> ValidationIssue:
    """A verifier could not run. Retrying the agent cannot fix this; re-running the verifier can."""
    return ValidationIssue(
        check=CLAIM_SUPPORT,
        severity="error",
        message=message,
        finding_index=index,
        kind="infrastructure",
    )


def _warning(check: str, message: str, index: int | None = None) -> ValidationIssue:
    return ValidationIssue(check=check, severity="warning", message=message, finding_index=index)
