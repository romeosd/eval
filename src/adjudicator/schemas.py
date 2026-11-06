"""Domain models.

Two families of models live here:

* ``LLMOutput`` subclasses are structured-output schemas returned by models via
  ``responses.parse``. They forbid extra keys and have no defaults so they satisfy
  OpenAI strict JSON-schema mode.
* Everything else is an internal record (policy index, cases, results, manifests).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from adjudicator.calculations import Unit


class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- outcomes


class Outcome(StrEnum):
    APPROVE = "APPROVE"
    DENY = "DENY"
    REFER = "REFER"


class CaseStatus(StrEnum):
    DECIDED = "decided"  # a model decision passed every enabled validation check
    REFERRED = "referred"  # safeguards forced a human referral; no model decision is trusted
    FAILED = "failed"  # infrastructure failure (budget, API, unreadable inputs); no prediction


# --------------------------------------------------------------------------- model outputs


class DiagramTranscription(LLMOutput):
    has_visual_content: bool = Field(description="True if the page contains a diagram/table/image.")
    title: str = Field(description="Title or caption of the visual, or empty string.")
    transcribed_text: str = Field(
        description="Every piece of text visible in the visual, verbatim."
    )
    rules_expressed: list[str] = Field(
        description="Each condition/outcome the visual expresses, as a plain sentence."
    )


class ImageTranscription(LLMOutput):
    visible_text: str = Field(description="All text visible in the image, verbatim.")
    description: str = Field(description="Neutral description of what the image shows.")


class InjectionVerdict(LLMOutput):
    contains_instructions: bool
    suspicious_spans: list[str] = Field(description="Exact substrings that are instruction-like.")
    reason: str


class PolicyCitation(LLMOutput):
    section_id: str = Field(description="Section id exactly as returned by the policy tools.")
    page: int = Field(description="1-based page number the quote appears on.")
    quote: str = Field(description="Verbatim text from that page (or its diagram transcription).")


class EvidenceCitation(LLMOutput):
    document_id: str = Field(description="Application document id, e.g. 'form' or 'receipt_1'.")
    quote: str = Field(description="Verbatim text from that document or its transcription.")


class Finding(LLMOutput):
    statement: str = Field(description="One factual or policy conclusion the decision relies on.")
    policy_citations: list[PolicyCitation]
    evidence_citations: list[EvidenceCitation]
    calculation_ids: list[str] = Field(description="Ids returned by the calculate tool, if any.")


class AgentDecision(LLMOutput):
    case_id: str
    outcome: Outcome
    payable_amount: float | None = Field(description="Amount payable if APPROVE, otherwise null.")
    findings: list[Finding]
    conflicts_identified: list[str] = Field(description="Contradictions found in the evidence.")
    missing_information: list[str] = Field(description="Information needed but not available.")
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(description="Short explanation tying the findings to the outcome.")


class SupportVerdict(LLMOutput):
    supported: bool
    explanation: str


# --------------------------------------------------------------------------- policy index


class PolicySection(Record):
    section_id: str
    document_id: str
    number: str
    title: str
    level: int
    parent_id: str | None
    page_start: int
    page_end: int
    text: str


class PolicyPage(Record):
    document_id: str
    page_number: int
    text: str
    image_path: Path
    diagram: DiagramTranscription | None
    quarantined_spans: list[str]

    def searchable_text(self) -> str:
        if self.diagram is None:
            return self.text
        diagram = self.diagram
        rules = "\n".join(diagram.rules_expressed)
        return f"{self.text}\n[DIAGRAM: {diagram.title}]\n{diagram.transcribed_text}\n{rules}"


class PolicyDocument(Record):
    document_id: str
    source_path: Path
    source_sha256: str
    pages: list[PolicyPage]
    sections: list[PolicySection]


class PolicyIndex(Record):
    cache_key: str
    ingest_version: str
    documents: list[PolicyDocument]

    def section(self, section_id: str) -> PolicySection | None:
        for document in self.documents:
            for section in document.sections:
                if section.section_id == section_id:
                    return section
        return None

    def page(self, document_id: str, page_number: int) -> PolicyPage | None:
        for document in self.documents:
            if document.document_id == document_id:
                for page in document.pages:
                    if page.page_number == page_number:
                        return page
        return None

    def all_sections(self) -> list[PolicySection]:
        return [section for document in self.documents for section in document.sections]

    def all_pages(self) -> list[PolicyPage]:
        return [page for document in self.documents for page in document.pages]


# --------------------------------------------------------------------------- applications


class DocumentKind(StrEnum):
    FORM = "form"
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"


class InjectionFinding(Record):
    source: str
    span: str
    method: Literal["heuristic", "classifier", "classifier_error"]
    reason: str


class ApplicationDocument(Record):
    document_id: str
    kind: DocumentKind
    description: str
    path: Path | None
    load_status: Literal["loaded", "failed"]
    load_error: str | None
    text: str  # sanitised text (or image transcription) the model may see
    image_path: Path | None
    quarantined: bool  # raw image withheld from the model because it carried instructions
    injection_findings: list[InjectionFinding]


class Case(Record):
    case_id: str
    split: str
    form: dict[str, JsonValue]
    documents: list[ApplicationDocument]

    def document(self, document_id: str) -> ApplicationDocument | None:
        return next((d for d in self.documents if d.document_id == document_id), None)

    @property
    def unrecovered_documents(self) -> list[ApplicationDocument]:
        return [d for d in self.documents if d.load_status == "failed"]

    @property
    def injection_findings(self) -> list[InjectionFinding]:
        return [f for d in self.documents for f in d.injection_findings]


# --------------------------------------------------------------------------- tooling + usage


class VerifiedInput(Record):
    """A calculation input whose value and unit were found in the source it cites."""

    value: float
    unit: Unit
    source: Literal["policy", "evidence", "derived"]
    ref: str  # policy section id, application document id, or earlier calculation id
    matched_text: str


class CalculationRecord(Record):
    calculation_id: str
    operation: str
    inputs: list[VerifiedInput]
    datetimes: list[str]
    result: float
    unit: Unit


class UsageRecord(Record):
    model: str
    resolved_model: str  # snapshot id the provider actually served (aliases move over time)
    purpose: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    latency_s: float


# --------------------------------------------------------------------------- validation + results


class ValidationIssue(Record):
    check: str
    severity: Literal["error", "warning"]
    message: str
    finding_index: int | None = None
    # "decision": the candidate is wrong or unsupported; "infrastructure": a verifier could
    # not run (timeout, outage). Retrying the agent cannot fix the latter.
    kind: Literal["decision", "infrastructure"] = "decision"


class ValidationReport(Record):
    attempt: int
    checks_required: list[str]
    checks_run: list[str]
    issues: list[ValidationIssue]
    run: int = 0  # independent consensus run this report belongs to
    passed: bool = False  # always recomputed; a stored value is never trusted

    @model_validator(mode="after")
    def _derive_passed(self) -> ValidationReport:
        """Fail-closed: every required check must have run and none may report an error."""
        all_ran = set(self.checks_required) <= set(self.checks_run)
        self.passed = all_ran and not any(issue.severity == "error" for issue in self.issues)
        return self

    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]


class CaseResult(Record):
    run_id: str
    case_id: str
    status: CaseStatus
    prediction: Outcome | None
    payable_amount: float | None
    decision: AgentDecision | None  # only ever a decision that passed validation
    rejected_candidates: list[AgentDecision]  # kept for audit, never used as the prediction
    validation_history: list[ValidationReport]
    referral_reasons: list[str]
    injection_findings: list[InjectionFinding]
    usage: list[UsageRecord]
    calculations: list[CalculationRecord]
    policy_sections_read: list[str]
    policy_pages_viewed: list[str]  # "document:page"
    error: str | None
    duration_s: float
    cost_usd: float = 0.0  # always recomputed from usage

    @model_validator(mode="after")
    def _derive_cost(self) -> CaseResult:
        self.cost_usd = round(sum(u.cost_usd for u in self.usage), 6)
        return self


class CaseLabel(Record):
    case_id: str
    expected_outcome: Outcome
    expected_payable_amount: float | None
    expected_sections: list[str]  # section numbers a correct decision should rely on, e.g. "5.3"
    notes: str
    synthetic: bool = False
    source_case_id: str | None = None  # variants: the case they were derived from


class RunManifest(Record):
    run_id: str
    split: str
    started_at: str
    finished_at: str | None
    git_sha: str | None
    git_dirty: bool | None  # uncommitted changes at run time: the SHA alone would not reproduce it
    settings_fingerprint: str
    behavior_fingerprint: str  # settings that can change a decision; must match to resume
    resume_count: int
    settings: dict[str, JsonValue]
    prompt_hashes: dict[str, str]
    policy_cache_key: str
    resolved_models: dict[str, list[str]]  # configured model name -> snapshot ids actually served
    case_ids: list[str]
    case_filter: list[str] | None  # set when only a subset of the split was run
    completed_case_ids: list[str]
