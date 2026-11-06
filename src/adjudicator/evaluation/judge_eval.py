"""Measure the claim-support judge itself.

The judge gates every decision, so its error rates matter as much as the adjudicator's.
For each finding in a finished run we ask the judge about:

* the original finding (a **control**; it already passed validation, so this measures
  consistency and false rejects, with a known selection bias), and
* deliberately corrupted copies that should NOT be supported: a quote from an unrelated
  policy section, a changed amount, or evidence swapped for an unrelated document.

``false_accept_rate`` (corrupted finding judged supported) is the number to watch.
Corruptions are generated deterministically and can occasionally still be supported by
chance, so review a sample of the exported probes by hand before quoting a rate; the
JSON output contains everything needed for that review.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from adjudicator.config import Settings
from adjudicator.llm.client import (
    BudgetTracker,
    CallContext,
    LLMClient,
    ResponsesTransport,
)
from adjudicator.llm.replay import build_transport
from adjudicator.observability import AuditLog
from adjudicator.pipeline import Services
from adjudicator.prompts import PromptLibrary
from adjudicator.runner import load_run, make_services
from adjudicator.schemas import (
    Case,
    CaseResult,
    CaseStatus,
    EvidenceCitation,
    Finding,
    PolicyCitation,
    PolicyIndex,
    Record,
)

_AMOUNT = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
_WORD = re.compile(r"[a-z]{4,}")
CONTROL = "control"


class JudgeProbe(Record):
    case_id: str
    finding_index: int
    kind: str  # "control" or a corruption name
    expected_supported: bool
    judged_supported: bool | None  # None when the judge could not run
    detail: str
    statement: str


class KindStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int
    errors: int
    wrong_verdicts: int  # control judged unsupported, or corruption judged supported


class JudgeEvalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    n_findings: int
    false_reject_rate: float | None  # controls judged unsupported
    false_accept_rate: float | None  # corruptions judged supported
    error_rate: float
    by_kind: dict[str, KindStats]


type Corruption = Callable[[Finding, int, Case, PolicyIndex], Finding | None]


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _wrong_policy_quote(
    finding: Finding, offset: int, _case: Case, index: PolicyIndex
) -> Finding | None:
    cited = {c.section_id for c in finding.policy_citations}
    claim_words = _words(finding.statement)
    candidates = sorted(
        (s for s in index.all_sections() if s.section_id not in cited and len(s.text) > 40),
        key=lambda s: (len(_words(s.text) & claim_words), s.section_id),  # least related first
    )
    if not candidates or not finding.policy_citations:
        return None
    section = candidates[offset % min(len(candidates), 3)]
    swapped = PolicyCitation(
        section_id=section.section_id,
        page=section.page_start,
        quote=" ".join(section.text.split())[:100],
    )
    return finding.model_copy(update={"policy_citations": [swapped]})


def _wrong_amount(
    finding: Finding, _offset: int, _case: Case, _index: PolicyIndex
) -> Finding | None:
    match = _AMOUNT.search(finding.statement)
    if match is None:
        return None
    try:
        value = float(match.group().replace("$", "").replace(",", ""))
    except ValueError:
        return None
    changed = f"${value * 2:,.2f}" if match.group().startswith("$") else f"{value * 2:g}"
    statement = finding.statement[: match.start()] + changed + finding.statement[match.end() :]
    return finding.model_copy(update={"statement": statement})


def _unrelated_evidence(
    finding: Finding, _offset: int, case: Case, _index: PolicyIndex
) -> Finding | None:
    cited = {c.document_id for c in finding.evidence_citations}
    others = [
        d
        for d in case.documents
        if d.document_id not in cited and d.document_id != "form" and d.text.strip()
    ]
    if not others or not finding.evidence_citations:
        return None
    other = others[0]
    quote = " ".join(other.text.split())[:100]
    return finding.model_copy(
        update={
            "evidence_citations": [EvidenceCitation(document_id=other.document_id, quote=quote)]
        }
    )


CORRUPTIONS: dict[str, Corruption] = {
    "wrong_policy_quote": _wrong_policy_quote,
    "wrong_amount": _wrong_amount,
    "unrelated_evidence": _unrelated_evidence,
}


async def run_judge_eval(
    settings: Settings,
    run_dir: Path,
    *,
    max_findings: int = 30,
    transport: ResponsesTransport | None = None,
) -> JudgeEvalSummary:
    manifest, results = load_run(run_dir)
    index = PolicyIndex.model_validate_json((run_dir / "policy_index.json").read_text("utf-8"))
    prompts = PromptLibrary.load(settings.paths.prompts_dir)
    llm = LLMClient(
        build_transport(settings, transport), settings, BudgetTracker(settings.run.budget_usd)
    )
    services = make_services(settings, llm, prompts, index)
    context = CallContext(audit=AuditLog(run_dir / "judge_eval_audit.jsonl", scope="judge_eval"))

    findings = [
        (result, i, finding)
        for result in results
        if result.status is CaseStatus.DECIDED and result.decision is not None
        for i, finding in enumerate(result.decision.findings)
    ][:max_findings]

    semaphore = asyncio.Semaphore(settings.run.concurrency)

    async def probe(
        result: CaseResult, i: int, finding: Finding, kind: str, case: Case
    ) -> JudgeProbe | None:
        if kind == CONTROL:
            candidate: Finding | None = finding
        else:
            candidate = CORRUPTIONS[kind](finding, i, case, index)
        if candidate is None:
            return None
        calculations = {c.calculation_id: c for c in result.calculations}
        async with semaphore:
            issue = await services.validator.judge_finding(
                candidate, case, calculations, context, i
            )
        errored = issue is not None and issue.kind == "infrastructure"
        return JudgeProbe(
            case_id=result.case_id,
            finding_index=i,
            kind=kind,
            expected_supported=kind == CONTROL,
            judged_supported=None if errored else issue is None,
            detail=issue.message if issue else "supported",
            statement=candidate.statement,
        )

    cases = await _load_cases(
        services, manifest.split, {r.case_id for r, _, _ in findings}, context
    )
    probes = [
        p
        for p in await asyncio.gather(
            *(
                probe(result, i, finding, kind, cases[result.case_id])
                for result, i, finding in findings
                for kind in (CONTROL, *CORRUPTIONS)
            )
        )
        if p is not None
    ]
    summary = _summarise(manifest.run_id, len(findings), probes)
    (run_dir / "judge_eval.json").write_text(
        "[" + ",".join(p.model_dump_json() for p in probes) + "]", encoding="utf-8"
    )
    (run_dir / "judge_eval.md").write_text(render(summary), encoding="utf-8")
    return summary


async def _load_cases(
    services: Services, split: str, case_ids: set[str], context: CallContext
) -> dict[str, Case]:
    base = services.settings.paths.cases_dir / split
    loaded = await asyncio.gather(
        *(services.loader.load(base / case_id, split, context) for case_id in sorted(case_ids))
    )
    return {case.case_id: case for case in loaded}


def _summarise(run_id: str, n_findings: int, probes: list[JudgeProbe]) -> JudgeEvalSummary:
    by_kind: dict[str, KindStats] = {}
    for kind in (CONTROL, *CORRUPTIONS):
        subset = [p for p in probes if p.kind == kind]
        errors = sum(p.judged_supported is None for p in subset)
        wrong = sum(
            p.judged_supported is not None and p.judged_supported != p.expected_supported
            for p in subset
        )
        by_kind[kind] = KindStats(n=len(subset), errors=errors, wrong_verdicts=wrong)

    def rate(kinds: list[str]) -> float | None:
        answered = sum(by_kind[k].n - by_kind[k].errors for k in kinds)
        wrong = sum(by_kind[k].wrong_verdicts for k in kinds)
        return wrong / answered if answered else None

    return JudgeEvalSummary(
        run_id=run_id,
        n_findings=n_findings,
        false_reject_rate=rate([CONTROL]),
        false_accept_rate=rate(list(CORRUPTIONS)),
        error_rate=(sum(p.judged_supported is None for p in probes) / len(probes))
        if probes
        else 0.0,
        by_kind=by_kind,
    )


def render(summary: JudgeEvalSummary) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    lines = [
        f"# Judge evaluation: run `{summary.run_id}` ({summary.n_findings} findings)",
        "",
        "- **False accept rate** (corrupted finding judged supported): "
        f"{pct(summary.false_accept_rate)}",
        "- False reject rate (original finding judged unsupported): "
        f"{pct(summary.false_reject_rate)}",
        f"- Judge unavailable (fail-closed): {pct(summary.error_rate)}",
        "",
        "| Probe | n | judge errors | wrong verdicts |",
        "|---|---|---|---|",
        *(f"| {k} | {v.n} | {v.errors} | {v.wrong_verdicts} |" for k, v in summary.by_kind.items()),
        "",
        "> Controls already passed this judge during the run, so the false reject rate is "
        "optimistic. Corruptions are generated automatically; review `judge_eval.json` by hand "
        "before quoting any rate, and add human-labelled findings for an unbiased estimate.",
    ]
    return "\n".join(lines) + "\n"
