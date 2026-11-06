import asyncio

import httpx2
import openai
import pytest

from adjudicator.runner import RunIntegrityError, execute_run, load_run
from adjudicator.schemas import CaseStatus

from .conftest import ScriptedTransport, respond
from .helpers import referral


def _refer(request):
    return respond(referral(request.purpose.split(":")[1]))


def _flaky(request):
    if request.purpose.split(":")[1] == "dev-003":
        raise openai.APIConnectionError(request=httpx2.Request("POST", "http://example.test"))
    return _refer(request)


@pytest.fixture
def interrupted(settings, transport):
    transport.adjudicate = _flaky
    run_dir = asyncio.run(execute_run(settings, "dev", transport=transport, tag="r"))
    _, results = load_run(run_dir)
    assert {r.case_id: r.status for r in results}["dev-003"] is CaseStatus.FAILED
    return run_dir


def _resume(settings, run_dir, transport=None):
    transport = transport or ScriptedTransport(adjudicate=_refer)
    return transport, asyncio.run(execute_run(settings, "dev", transport=transport, resume=run_dir))


def test_resume_runs_only_the_failed_case_and_keeps_the_rest_untouched(settings, interrupted):
    kept = {
        p.name: p.read_text()
        for p in (interrupted / "results").glob("*.json")
        if "dev-003" not in p.name
    }
    transport, run_dir = _resume(settings, interrupted)

    assert run_dir == interrupted
    assert {p.split(":")[1] for p in transport.purposes("adjudicate")} == {"dev-003"}
    assert not transport.purposes("image_transcription:dev-001")  # nothing re-done elsewhere
    for name, text in kept.items():
        assert (run_dir / "results" / name).read_text() == text

    manifest, results = load_run(run_dir)
    assert manifest.resume_count == 1 and len(results) == 8
    rerun = {r.case_id: r for r in results}["dev-003"]
    assert rerun.status is CaseStatus.DECIDED  # the model chose REFER and it validated
    assert rerun.prediction.value == "REFER"
    assert all(r.run_id == manifest.run_id for r in results)


def test_resuming_a_finished_clean_run_does_nothing(settings, interrupted):
    _, run_dir = _resume(settings, interrupted)
    transport, _ = _resume(settings, run_dir)
    assert not transport.purposes("adjudicate")


def test_resume_refuses_changed_decision_settings(settings, interrupted):
    changed = settings.model_copy(
        update={"agent": settings.agent.model_copy(update={"max_attempts": 2})}
    )
    with pytest.raises(RunIntegrityError, match="settings that affect decisions"):
        _resume(changed, interrupted)


def test_resume_allows_changed_budget_and_concurrency(settings, interrupted):
    run = settings.run.model_copy(update={"concurrency": 1, "budget_usd": 99.0})
    _resume(settings.model_copy(update={"run": run}), interrupted)


def test_resume_refuses_changed_prompts(settings, interrupted):
    prompt = settings.paths.prompts_dir / "adjudicator.md"
    prompt.write_text(prompt.read_text() + "\n- changed", encoding="utf-8")
    with pytest.raises(RunIntegrityError, match="prompt files changed"):
        _resume(settings, interrupted)


def test_budget_already_spent_counts_against_the_resumed_run(settings, interrupted):
    tiny = settings.model_copy(update={"run": settings.run.model_copy(update={"budget_usd": 1e-9})})
    transport, run_dir = _resume(tiny, interrupted)
    _, results = load_run(run_dir)
    failed = {r.case_id: r for r in results}["dev-003"]
    assert failed.status is CaseStatus.FAILED and "budget" in (failed.error or "").lower()
    assert not transport.purposes("adjudicate")


def test_resume_rejects_a_different_split(settings, interrupted, transport):
    with pytest.raises(RunIntegrityError, match="split"):
        asyncio.run(execute_run(settings, "test", transport=transport, resume=interrupted))
