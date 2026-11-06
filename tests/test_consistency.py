import asyncio
import json

import httpx2
import openai
import pytest

from adjudicator.evaluation.consistency import analyse, render
from adjudicator.runner import RunIntegrityError, execute_run

from .conftest import ScriptedTransport, respond
from .helpers import referral


def _refer(request):
    return respond(referral(request.purpose.split(":")[1]))


def _flaky(*case_ids):
    def agent(request):
        if request.purpose.split(":")[1] in case_ids:
            raise openai.APIConnectionError(request=httpx2.Request("POST", "http://example.test"))
        return _refer(request)

    return agent


def _run(settings, agent, split="dev"):
    transport = ScriptedTransport(adjudicate=agent)
    return asyncio.run(execute_run(settings, split, transport=transport))


def test_identical_runs_agree_completely(settings):
    runs = [_run(settings, _refer), _run(settings, _refer)]
    summary = analyse(runs, settings.paths.labels_dir)
    assert summary.mean_agreement == 1.0 and summary.identical_cases == 8
    assert summary.unstable_cases == [] and summary.correctness_flips == []
    assert summary.mean_evidence_overlap == 1.0 and not summary.snapshot_drift


def test_disagreement_and_right_then_wrong_flips_are_reported(settings):
    stable = _run(settings, _refer)
    unstable = _run(settings, _flaky("dev-003", "dev-008"))
    summary = analyse([stable, unstable], settings.paths.labels_dir)
    assert summary.unstable_cases == ["dev-003", "dev-008"]
    assert summary.correctness_flips == ["dev-008"]  # expected REFER: right once, wrong once
    assert summary.mean_agreement == pytest.approx((6 + 0.5 + 0.5) / 8)
    per_case = {c.case_id: c for c in summary.per_case}
    assert per_case["dev-008"].correct_runs == 1 and per_case["dev-001"].agreement == 1.0
    text = render(summary)
    assert "Mean outcome agreement" in text and "dev-008" in text


def test_snapshot_drift_between_runs_is_flagged(settings):
    first, second = _run(settings, _refer), _run(settings, _refer)
    path = second / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["resolved_models"]["gpt-5.5"] = ["gpt-5.5-other-snapshot"]
    path.write_text(json.dumps(manifest))
    assert analyse([first, second], settings.paths.labels_dir).snapshot_drift


def test_needs_two_runs_of_the_same_split(settings):
    only = _run(settings, _refer)
    with pytest.raises(RunIntegrityError, match="at least two"):
        analyse([only])
    other_split = _run(settings, _refer, split="test")
    with pytest.raises(RunIntegrityError, match="same split"):
        analyse([only, other_split])
