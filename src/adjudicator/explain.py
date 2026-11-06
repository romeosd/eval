"""Human-readable trace of one case, for demos, walkthroughs and debugging.

Shows what was decided, on which evidence, which safeguards fired, and the exact
sequence of tool calls — the material a live code review will probe.
"""

from __future__ import annotations

import json
from pathlib import Path

from adjudicator.runner import RESULTS_DIR
from adjudicator.schemas import CaseResult


def explain_case(run_dir: Path, case_id: str) -> str:
    result_path = run_dir / RESULTS_DIR / f"{case_id}.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"No result for {case_id!r} in {run_dir}")
    result = CaseResult.model_validate_json(result_path.read_text("utf-8"))
    lines = [
        f"# {case_id} — {result.status.value.upper()} → {result.prediction or 'no prediction'}",
        f"payable: {result.payable_amount}   cost: ${result.cost_usd:.4f}   "
        f"time: {result.duration_s:.1f}s   attempts: {len(result.validation_history)}",
    ]
    if result.error:
        lines.append(f"error: {result.error}")
    for reason in result.referral_reasons:
        lines.append(f"referral reason: {reason}")
    for finding in result.injection_findings:
        lines.append(f"injection flagged [{finding.method}] in {finding.source}: {finding.span!r}")

    if result.decision is not None:
        decision = result.decision
        lines += ["", f"## Decision ({decision.confidence} confidence)", decision.rationale]
        for i, f in enumerate(decision.findings):
            lines.append(f"{i}. {f.statement}")
            lines += [
                f"   policy {c.section_id} p{c.page}: “{c.quote}”" for c in f.policy_citations
            ]
            lines += [f"   evidence {c.document_id}: “{c.quote}”" for c in f.evidence_citations]
            if f.calculation_ids:
                lines.append(f"   calculations: {', '.join(f.calculation_ids)}")
        lines += [f"conflict: {c}" for c in decision.conflicts_identified]
        lines += [f"missing: {m}" for m in decision.missing_information]

    lines += ["", "## Validation"]
    for report in result.validation_history:
        verdict = "PASS" if report.passed else "FAIL"
        lines.append(f"attempt {report.attempt}: {verdict} ({len(report.checks_run)} checks run)")
        lines += [f"   {i.severity} [{i.check}] {i.message}" for i in report.issues]

    audit = run_dir / "cases" / case_id / "audit.jsonl"
    if audit.is_file():
        lines += ["", "## Tool calls"]
        for raw in audit.read_text("utf-8").splitlines():
            event = json.loads(raw)
            if event.get("kind") == "tool_call":
                flag = " (error)" if event.get("is_error") else ""
                lines.append(f"- {event['tool']}{flag} {event.get('arguments', '')}")
    return "\n".join(lines) + "\n"
