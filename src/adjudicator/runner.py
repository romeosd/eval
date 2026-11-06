"""Run orchestration: one immutable directory per run, with a manifest.

Run directories are never reused: evaluation reads results only from the run it is
pointed at and checks that every case in that run completed, so a failed rerun can
never be scored with stale outputs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from adjudicator.agent import Adjudicator
from adjudicator.application.loader import CaseLoader
from adjudicator.config import Settings
from adjudicator.llm.client import (
    BudgetTracker,
    CallContext,
    LLMClient,
    ResponsesTransport,
)
from adjudicator.llm.replay import build_transport
from adjudicator.observability import AuditLog, configure_logging
from adjudicator.pipeline import CasePipeline, Services
from adjudicator.policy.ingest import PolicyIngestor
from adjudicator.policy.retrieval import PolicySearchIndex
from adjudicator.prompts import PromptLibrary
from adjudicator.schemas import CaseResult, CaseStatus, PolicyIndex, RunManifest, UsageRecord
from adjudicator.security import InjectionScreener
from adjudicator.validation import DecisionValidator

logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
MANIFEST_FILE = "manifest.json"
PREDICTIONS_FILE = "predictions.jsonl"


class RunIntegrityError(RuntimeError):
    pass


async def build_policy_index(
    settings: Settings, llm: LLMClient, prompts: PromptLibrary, audit: AuditLog
) -> tuple[PolicyIndex, CallContext]:
    screener = InjectionScreener(llm, settings, prompts)
    context = CallContext(audit=audit)
    index = await PolicyIngestor(llm, screener, settings, prompts).load_or_build(context)
    return index, context


def make_services(
    settings: Settings, llm: LLMClient, prompts: PromptLibrary, index: PolicyIndex
) -> Services:
    screener = InjectionScreener(llm, settings, prompts)
    return Services(
        settings=settings,
        llm=llm,
        index=index,
        search_index=PolicySearchIndex(index),
        screener=screener,
        loader=CaseLoader(llm, screener, settings, prompts),
        adjudicator=Adjudicator(llm, settings, prompts, index),
        validator=DecisionValidator(llm, settings, prompts, index, screener),
    )


async def execute_run(
    settings: Settings,
    split: str,
    *,
    transport: ResponsesTransport | None = None,
    tag: str | None = None,
    case_filter: list[str] | None = None,
    resume: Path | None = None,
) -> Path:
    """Run every case in a split in a new, immutable run directory.

    With ``resume`` the same directory is continued instead: cases that already have a
    non-failed result are kept untouched, cases without one (or that failed on
    infrastructure) are run, and the budget already spent counts against the limit. Resuming
    is refused unless every setting that can change a decision is unchanged.
    """
    prompts = PromptLibrary.load(settings.paths.prompts_dir)
    prior: RunManifest | None = None
    if resume is None:
        run_id = _new_run_id(split, tag)
        run_dir = settings.paths.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = resume.resolve()
        prior = _read_manifest(run_dir)
        run_id = prior.run_id
        if prior.split != split:
            raise RunIntegrityError(f"Run {run_id} is for split {prior.split!r}, not {split!r}")
    configure_logging(log_file=run_dir / "run.log")
    started_at = prior.started_at if prior else datetime.now(UTC).isoformat()

    llm = LLMClient(
        build_transport(settings, transport), settings, BudgetTracker(settings.run.budget_usd)
    )
    index, ingest_context = await build_policy_index(
        settings, llm, prompts, AuditLog(run_dir / "ingest_audit.jsonl", scope="ingest")
    )
    if prior is None:
        _snapshot_inputs(run_dir, settings, index)
    else:
        _check_resumable(prior, settings, prompts, index, case_filter)
        case_filter = prior.case_filter
    services = make_services(settings, llm, prompts, index)

    case_dirs = _select_cases(services.loader.discover(split), split, case_filter)
    manifest = (
        prior.model_copy(update={"finished_at": None, "resume_count": prior.resume_count + 1})
        if prior
        else _new_manifest(
            settings=settings,
            prompts=prompts,
            index=index,
            run_id=run_id,
            split=split,
            started_at=started_at,
            case_dirs=case_dirs,
            case_filter=case_filter,
        )
    )
    _write_manifest(run_dir, manifest)

    existing = _existing_results(run_dir, run_id)
    kept = {cid: r for cid, r in existing.items() if r.status is not CaseStatus.FAILED}
    pending = [d for d in case_dirs if d.name not in kept]
    llm.budget.record(sum(r.cost_usd for r in existing.values()))  # spend already incurred
    logger.info(
        "Run %s: %d cases (%d kept, %d to run), policy index %s",
        run_id,
        len(case_dirs),
        len(case_dirs) - len(pending),
        len(pending),
        index.cache_key,
    )

    pipeline = CasePipeline(services, run_id, run_dir)
    semaphore = asyncio.Semaphore(settings.run.concurrency)

    async def run_one(case_dir: Path) -> CaseResult:
        async with semaphore:
            result = await pipeline.run(case_dir, split)
        _write_atomic(
            run_dir / RESULTS_DIR / f"{result.case_id}.json", result.model_dump_json(indent=2)
        )
        logger.info(
            "%s -> %s (%s) $%.4f", result.case_id, result.prediction, result.status, result.cost_usd
        )
        return result

    fresh = await asyncio.gather(*(run_one(d) for d in pending))
    results = [*kept.values(), *fresh]

    if sorted(r.case_id for r in results) != sorted(d.name for d in case_dirs):
        raise RunIntegrityError(f"Run {run_id}: completed cases do not match discovered cases")
    _finalize_run(run_dir, manifest, results, ingest_context.usage)
    logger.info("Run %s complete; spend this session $%.4f", run_id, llm.budget.spent_usd)
    return run_dir


def _select_cases(discovered: list[Path], split: str, case_filter: list[str] | None) -> list[Path]:
    if case_filter:
        unknown = sorted(set(case_filter) - {d.name for d in discovered})
        if unknown:
            raise RunIntegrityError(f"Unknown case ids for split {split!r}: {unknown}")
        discovered = [d for d in discovered if d.name in case_filter]
    if not discovered:
        raise RunIntegrityError(f"No cases found for split {split!r}")
    return discovered


def _new_manifest(
    *,
    settings: Settings,
    prompts: PromptLibrary,
    index: PolicyIndex,
    run_id: str,
    split: str,
    started_at: str,
    case_dirs: list[Path],
    case_filter: list[str] | None,
) -> RunManifest:
    root = settings.paths.prompts_dir.parent
    return RunManifest(
        run_id=run_id,
        split=split,
        started_at=started_at,
        finished_at=None,
        git_sha=_git_sha(root),
        git_dirty=_git_dirty(root),
        settings_fingerprint=settings.fingerprint(),
        behavior_fingerprint=settings.behavior_fingerprint(),
        resume_count=0,
        settings=settings.model_dump(mode="json"),
        prompt_hashes=prompts.hashes(),
        policy_cache_key=index.cache_key,
        resolved_models={},
        case_ids=[d.name for d in case_dirs],
        case_filter=sorted(case_filter) if case_filter else None,
        completed_case_ids=[],
    )


def _finalize_run(
    run_dir: Path,
    manifest: RunManifest,
    results: list[CaseResult],
    session_ingest_usage: list[UsageRecord],
) -> None:
    ordered = sorted(results, key=lambda r: r.case_id)
    _write_atomic(
        run_dir / PREDICTIONS_FILE,
        "".join(
            r.model_dump_json(
                include={"run_id", "case_id", "status", "prediction", "payable_amount"}
            )
            + "\n"
            for r in ordered
        ),
    )
    ingest_usage = [*_previous_ingest_usage(run_dir), *session_ingest_usage]
    _write_atomic(
        run_dir / "ingest_usage.json",
        "[" + ",".join(u.model_dump_json() for u in ingest_usage) + "]",
    )
    _write_manifest(
        run_dir,
        manifest.model_copy(
            update={
                "finished_at": datetime.now(UTC).isoformat(),
                "completed_case_ids": [r.case_id for r in ordered],
                "resolved_models": _resolved_models(
                    [*ingest_usage, *(u for r in results for u in r.usage)]
                ),
            }
        ),
    )


def _check_resumable(
    prior: RunManifest,
    settings: Settings,
    prompts: PromptLibrary,
    index: PolicyIndex,
    case_filter: list[str] | None,
) -> None:
    problems: list[str] = []
    if prior.behavior_fingerprint != settings.behavior_fingerprint():
        problems.append("settings that affect decisions changed (models, retrieval, agent, ...)")
    if prior.prompt_hashes != prompts.hashes():
        problems.append("prompt files changed")
    if prior.policy_cache_key != index.cache_key:
        problems.append("the processed policy changed")
    if case_filter and sorted(case_filter) != (prior.case_filter or []):
        problems.append("a different --cases filter was given")
    if problems:
        raise RunIntegrityError(
            f"Cannot resume {prior.run_id}: {'; '.join(problems)}. "
            "Start a new run so every result in it comes from one configuration."
        )


def _existing_results(run_dir: Path, run_id: str) -> dict[str, CaseResult]:
    found: dict[str, CaseResult] = {}
    for path in sorted((run_dir / RESULTS_DIR).glob("*.json")):
        try:
            result = CaseResult.model_validate_json(path.read_text("utf-8"))
        except ValueError:
            logger.warning("Ignoring unreadable result %s; the case will be re-run", path.name)
            continue
        if result.run_id == run_id:
            found[result.case_id] = result
    return found


def _previous_ingest_usage(run_dir: Path) -> list[UsageRecord]:
    path = run_dir / "ingest_usage.json"
    if not path.is_file():
        return []
    return [UsageRecord.model_validate(u) for u in json.loads(path.read_text("utf-8"))]


def _read_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / MANIFEST_FILE).read_text("utf-8"))


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)  # readers never observe a half-written file


def load_run(run_dir: Path) -> tuple[RunManifest, list[CaseResult]]:
    """Load a finished run, refusing incomplete or inconsistent ones."""
    manifest = _read_manifest(run_dir)
    if manifest.finished_at is None:
        raise RunIntegrityError(f"Run {manifest.run_id} did not finish; refusing to score it")
    results = [
        CaseResult.model_validate_json(p.read_text("utf-8"))
        for p in sorted((run_dir / RESULTS_DIR).glob("*.json"))
    ]
    if sorted(r.case_id for r in results) != sorted(manifest.case_ids):
        raise RunIntegrityError(f"Run {manifest.run_id}: result files do not match manifest")
    if any(r.run_id != manifest.run_id for r in results):
        raise RunIntegrityError(f"Run {manifest.run_id}: contains results from another run")
    return manifest, results


def _write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    tmp = run_dir / f"{MANIFEST_FILE}.tmp"
    tmp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(run_dir / MANIFEST_FILE)


def _new_run_id(split: str, tag: str | None) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    suffix = f"-{tag}" if tag else ""
    return f"{stamp}-{split}{suffix}-{secrets.token_hex(3)}"


def _snapshot_inputs(run_dir: Path, settings: Settings, index: PolicyIndex) -> None:
    """Copy what a decision depended on into the run, so it can be reconstructed later.

    The processed policy (sections, page text, diagram transcriptions) and the exact prompt
    files are stored with the run. Page images stay in the cache under the same cache key.
    """
    (run_dir / "policy_index.json").write_text(index.model_dump_json(indent=2), encoding="utf-8")
    shutil.copytree(settings.paths.prompts_dir, run_dir / "prompts")


def _resolved_models(usage: list[UsageRecord]) -> dict[str, list[str]]:
    resolved: dict[str, set[str]] = {}
    for record in usage:
        resolved.setdefault(record.model, set()).add(record.resolved_model)
    return {name: sorted(ids) for name, ids in sorted(resolved.items())}


def _git_dirty(root: Path) -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def _git_sha(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None
