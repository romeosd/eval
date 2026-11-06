import asyncio
import json

import pytest

from adjudicator.evaluation.metrics import score_run, write_report
from adjudicator.evaluation.variants import generate_variants
from adjudicator.explain import explain_case
from adjudicator.runner import RunIntegrityError, execute_run, load_run
from adjudicator.schemas import CaseLabel, Outcome

from .conftest import respond
from .helpers import referral


def _refer_everything(request):
    case_id = request.purpose.split(":")[1]
    return respond(referral(case_id))


@pytest.fixture
def run_dir(settings, transport):
    transport.adjudicate = _refer_everything
    return asyncio.run(execute_run(settings, "dev", transport=transport, tag="t"))


def test_run_is_complete_and_scored(settings, run_dir):
    manifest, results = load_run(run_dir)
    assert manifest.case_ids == manifest.completed_case_ids and len(results) == 8
    summary = write_report(run_dir, settings.paths.labels_dir)
    assert summary.n_cases == 8
    assert summary.unsafe_approval_rate == 0.0
    assert summary.decision_accuracy == 2 / 8  # dev-007 and dev-008 are REFER cases
    assert (run_dir / "eval_report.md").is_file()


def test_runs_are_never_reused(settings, transport, run_dir):
    second = asyncio.run(execute_run(settings, "dev", transport=transport, tag="t"))
    assert second != run_dir


def test_missing_result_file_is_rejected(run_dir):
    next((run_dir / "results").glob("*.json")).unlink()
    with pytest.raises(RunIntegrityError):
        load_run(run_dir)


def test_label_case_mismatch_is_rejected(settings, run_dir):
    extra = CaseLabel(
        case_id="dev-999",
        expected_outcome=Outcome.DENY,
        expected_payable_amount=None,
        expected_sections=[],
        notes="",
    )
    (settings.paths.labels_dir / "dev" / "dev-999.json").write_text(extra.model_dump_json())
    with pytest.raises(RunIntegrityError):
        score_run(run_dir, settings.paths.labels_dir)


def test_variants_have_consistent_labels(settings):
    count = generate_variants(settings, "dev", "variants")
    cases = sorted(p.name for p in (settings.paths.cases_dir / "variants").iterdir())
    labels = sorted(p.stem for p in (settings.paths.labels_dir / "variants").glob("*.json"))
    assert count == len(cases) == len(labels) and cases == labels
    late = json.loads(
        (settings.paths.labels_dir / "variants" / "dev-001__late_submission.json").read_text()
    )
    assert late["expected_outcome"] == "DENY" and late["synthetic"] is True
    assert not (
        settings.paths.cases_dir / "variants" / "dev-001__missing_attachment" / "invoice.png"
    ).exists()


def test_partial_run_is_scored_on_its_subset_and_flagged(settings, transport):
    transport.adjudicate = _refer_everything
    run = asyncio.run(
        execute_run(settings, "dev", transport=transport, case_filter=["dev-007", "dev-001"])
    )
    summary, scores = score_run(run, settings.paths.labels_dir)
    assert summary.partial and [s.case_id for s in scores] == ["dev-001", "dev-007"]
    with pytest.raises(RunIntegrityError):
        asyncio.run(execute_run(settings, "dev", transport=transport, case_filter=["nope"]))


def test_explain_renders_trace(run_dir):
    text = explain_case(run_dir, "dev-007")
    assert "REFER" in text and "## Validation" in text


def test_run_is_reconstructable(run_dir, settings):
    manifest, _ = load_run(run_dir)
    assert (run_dir / "policy_index.json").is_file()
    assert (run_dir / "prompts" / "adjudicator.md").read_text() == (
        settings.paths.prompts_dir / "adjudicator.md"
    ).read_text()
    assert manifest.git_dirty is None  # not a git checkout in the test sandbox
    assert manifest.resolved_models["gpt-5.5"] == ["gpt-5.5-snapshot"]
    assert set(manifest.resolved_models) >= {"gpt-5.4-mini", "gpt-5.4-nano"}
