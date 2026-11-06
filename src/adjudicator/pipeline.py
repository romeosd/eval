"""Per-case orchestration with an explicit, fail-closed outcome state machine.

    load → screen → gate → k x ([decide → validate] x max_attempts) → consensus
        → DECIDED | REFERRED | FAILED

``consensus_runs`` (k, default 1) runs independent attempt loops; with k > 1 the outcome and
payable amount must agree across all of them, so model variance becomes a referral instead
of a coin flip.

* Only a decision that passed validation is ever stored as ``decision`` or used for
  ``prediction``. Rejected candidates are kept separately for audit.
* Exhausted attempts, unscreenable content and agent turn limits → REFERRED
  (prediction ``REFER``: a human looks at it).
* Budget exhaustion, API failures and unreadable forms → FAILED (no prediction).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import openai

from adjudicator.agent import Adjudicator, AgentError
from adjudicator.application.loader import CaseLoader, CaseLoadError
from adjudicator.config import Settings
from adjudicator.llm.client import BudgetExceededError, CallContext, LLMClient, ModelCallError
from adjudicator.observability import AuditLog
from adjudicator.policy.retrieval import PolicySearchIndex
from adjudicator.schemas import (
    AgentDecision,
    Case,
    CaseResult,
    CaseStatus,
    InjectionFinding,
    Outcome,
    PolicyIndex,
    ValidationReport,
)
from adjudicator.security import InjectionScreener, has_unscreened_content
from adjudicator.tools import Toolbox
from adjudicator.validation import DecisionValidator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Services:
    settings: Settings
    llm: LLMClient
    index: PolicyIndex
    search_index: PolicySearchIndex
    screener: InjectionScreener
    loader: CaseLoader
    adjudicator: Adjudicator
    validator: DecisionValidator


@dataclass
class _Trace:
    """Mutable accumulator for one case; frozen into a ``CaseResult`` at the end."""

    run_id: str
    case_id: str
    context: CallContext
    started: float
    reports: list[ValidationReport]
    rejected: list[AgentDecision]
    injection_findings: list[InjectionFinding]

    def finish(
        self,
        status: CaseStatus,
        *,
        decision: AgentDecision | None = None,
        toolbox: Toolbox | None = None,
        reasons: list[str] | None = None,
        error: str | None = None,
    ) -> CaseResult:
        if status is CaseStatus.DECIDED and decision is None:
            raise ValueError("a DECIDED result requires a validated decision")
        prediction = (
            decision.outcome
            if decision is not None
            else Outcome.REFER
            if status is CaseStatus.REFERRED
            else None
        )
        result = CaseResult(
            run_id=self.run_id,
            case_id=self.case_id,
            status=status,
            prediction=prediction,
            payable_amount=decision.payable_amount if decision else None,
            decision=decision,
            rejected_candidates=self.rejected,
            validation_history=self.reports,
            referral_reasons=reasons or [],
            injection_findings=self.injection_findings,
            usage=self.context.usage,
            calculations=list(toolbox.calculations.values()) if toolbox else [],
            policy_sections_read=sorted(toolbox.sections_read) if toolbox else [],
            policy_pages_viewed=sorted(toolbox.pages_viewed) if toolbox else [],
            error=error,
            duration_s=round(time.perf_counter() - self.started, 2),
        )
        self.context.audit.event(
            "case_finished",
            status=status.value,
            prediction=prediction.value if prediction else None,
            reasons=reasons or [],
            error=error,
            cost_usd=result.cost_usd,
        )
        return result


@dataclass
class _RunOutcome:
    """One independent attempt loop (one of ``consensus_runs``)."""

    decision: AgentDecision | None
    toolbox: Toolbox
    reports: list[ValidationReport]
    rejected: list[AgentDecision]
    reasons: list[str]


class CasePipeline:
    def __init__(self, services: Services, run_id: str, run_dir: Path) -> None:
        self._s = services
        self._run_id = run_id
        self._run_dir = run_dir

    async def run(self, case_dir: Path, split: str) -> CaseResult:
        case_id = case_dir.name
        audit = AuditLog(self._run_dir / "cases" / case_id / "audit.jsonl", scope=case_id)
        trace = _Trace(
            run_id=self._run_id,
            case_id=case_id,
            context=CallContext(audit=audit),
            started=time.perf_counter(),
            reports=[],
            rejected=[],
            injection_findings=[],
        )
        try:
            case = await self._s.loader.load(case_dir, split, trace.context)
            trace.injection_findings = case.injection_findings
            if self._s.settings.security.fail_closed and has_unscreened_content(
                case.injection_findings
            ):
                return trace.finish(
                    CaseStatus.REFERRED,
                    reasons=["application content could not be fully screened for injection"],
                )
            if case.unrecovered_documents:
                # Deterministic gate: the policy forbids deciding on incomplete evidence, so
                # do not spend tokens on an agent run that validation would reject anyway.
                missing = [f"{d.document_id}: {d.load_error}" for d in case.unrecovered_documents]
                return trace.finish(
                    CaseStatus.REFERRED,
                    reasons=[f"required documents could not be loaded ({'; '.join(missing)})"],
                )
            return await self._decide(case, trace)
        except (CaseLoadError, BudgetExceededError, ModelCallError, openai.APIError) as exc:
            logger.error("%s failed: %s", case_id, exc)
            return trace.finish(CaseStatus.FAILED, error=f"{type(exc).__name__}: {exc}")

    async def _decide(self, case: Case, trace: _Trace) -> CaseResult:
        runs = self._s.settings.agent.consensus_runs
        settled = await asyncio.gather(
            *(self._attempt_loop(case, trace.context, run) for run in range(runs)),
            return_exceptions=True,
        )
        for item in settled:  # let every run finish first so no task is left spending budget
            if isinstance(item, BaseException):
                raise item
        outcomes = [item for item in settled if isinstance(item, _RunOutcome)]
        trace.reports = sorted(
            (r for o in outcomes for r in o.reports), key=lambda r: (r.run, r.attempt)
        )
        trace.rejected = [d for o in outcomes for d in o.rejected]

        decided = [o for o in outcomes if o.decision is not None]
        if len(decided) < runs:
            failed = [r for o in outcomes if o.decision is None for r in o.reasons]
            head = (
                failed[0]
                if runs == 1 and failed
                else f"consensus not reached: {len(decided)} of {runs} runs produced a "
                "validated decision"
            )
            return trace.finish(
                CaseStatus.REFERRED,
                toolbox=outcomes[0].toolbox,
                reasons=[head, *dict.fromkeys(failed[1:] if runs == 1 else failed)],
            )
        keys = {_agreement_key(o.decision) for o in decided if o.decision}
        if len(keys) > 1:
            trace.context.audit.event("consensus_disagreement", keys=sorted(keys))
            return trace.finish(
                CaseStatus.REFERRED,
                toolbox=outcomes[0].toolbox,
                reasons=[f"consensus runs disagree: {', '.join(sorted(keys))}"],
            )
        chosen = decided[0]
        return trace.finish(CaseStatus.DECIDED, decision=chosen.decision, toolbox=chosen.toolbox)

    async def _attempt_loop(self, case: Case, context: CallContext, run: int) -> _RunOutcome:
        toolbox = Toolbox(self._s.index, self._s.search_index, case, self._s.settings, context)
        reports: list[ValidationReport] = []
        rejected: list[AgentDecision] = []
        feedback: list[str] = []
        previous: AgentDecision | None = None
        max_attempts = self._s.settings.agent.max_attempts
        for attempt in range(1, max_attempts + 1):
            try:
                candidate = await self._s.adjudicator.decide(
                    case,
                    toolbox,
                    context,
                    run=run,
                    attempt=attempt,
                    feedback=feedback,
                    previous=previous,
                )
            except AgentError as exc:
                return _RunOutcome(None, toolbox, reports, rejected, [str(exc)])
            report = await self._s.validator.validate(
                candidate, case, toolbox, context, attempt=attempt
            )
            for _ in range(self._s.settings.validation.max_revalidations):
                if report.passed or not _only_infrastructure_errors(report):
                    break
                logger.info("%s: verifier unavailable, re-running validation only", case.case_id)
                report = await self._s.validator.validate(
                    candidate, case, toolbox, context, attempt=attempt
                )
            report = report.model_copy(update={"run": run})
            reports.append(report)
            if report.passed:
                return _RunOutcome(candidate, toolbox, reports, rejected, [])
            rejected.append(candidate)
            if _only_infrastructure_errors(report):
                # Never judged wrong, and a new agent run cannot fix a dead verifier.
                reasons = ["verification unavailable", *(i.message for i in report.errors())]
                return _RunOutcome(None, toolbox, reports, rejected, reasons)
            previous = candidate
            feedback = [issue.message for issue in report.errors()]
            logger.info("%s attempt %d rejected: %s", case.case_id, attempt, feedback)
        reasons = [f"no decision passed validation in {max_attempts} attempts", *feedback]
        return _RunOutcome(None, toolbox, reports, rejected, reasons)


def _agreement_key(decision: AgentDecision) -> str:
    amount = "" if decision.payable_amount is None else f"{decision.payable_amount:.2f}"
    return f"{decision.outcome.value}/{amount}"


def _only_infrastructure_errors(report: ValidationReport) -> bool:
    errors = report.errors()
    return bool(errors) and all(issue.kind == "infrastructure" for issue in errors)
