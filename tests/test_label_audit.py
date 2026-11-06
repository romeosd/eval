import json
import shutil

import pytest

from adjudicator.evaluation.label_audit import audit_labels, policy_section_numbers
from adjudicator.evaluation.variants import generate_variants


def _label_path(settings, split, case_id):
    return settings.paths.labels_dir / split / f"{case_id}.json"


def _edit_label(settings, split, stem, **changes):
    path = _label_path(settings, split, stem)
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data))


def test_sample_data_passes_and_reports_class_balance(settings):
    report = audit_labels(settings)
    assert report.errors == []
    assert "dev: 8 cases, APPROVE=3, DENY=3, REFER=2" in report.notes
    assert any("fewer than 2 cases" in n and n.startswith("test") for n in report.notes)


def test_policy_section_numbers_come_from_the_pdf_without_a_model(settings):
    numbers = policy_section_numbers(settings.paths.policy_dir)
    assert {"3.1", "4.3", "5.3", "5.4"} <= numbers


def test_case_without_a_label_and_label_without_a_case(settings):
    _label_path(settings, "dev", "dev-001").unlink()
    shutil.rmtree(settings.paths.cases_dir / "dev" / "dev-002")
    errors = audit_labels(settings).errors
    assert "dev/dev-001: case has no label" in errors
    assert "dev/dev-002: label has no case" in errors


@pytest.mark.parametrize(
    ("case_id", "changes", "message"),
    [
        ("dev-001", {"expected_payable_amount": None}, "APPROVE needs a positive"),
        ("dev-003", {"expected_payable_amount": 10.0}, "must not have an amount"),
        ("dev-003", {"expected_sections": ["9.9"]}, "not in the policy"),
        ("dev-003", {"case_id": "someone-else"}, "does not match its file name"),
        ("dev-003", {"synthetic": True}, "without source_case_id"),
    ],
)
def test_inconsistent_labels_are_reported(settings, case_id, changes, message):
    _edit_label(settings, "dev", case_id, **changes)
    assert any(message in e for e in audit_labels(settings).errors)


def test_identical_form_in_two_splits_is_leakage(settings):
    source = settings.paths.cases_dir / "dev" / "dev-001"
    target = settings.paths.cases_dir / "test" / "test-001"
    shutil.copy(source / "form.json", target / "form.json")
    assert any("possible leakage" in e for e in audit_labels(settings).errors)


def test_the_same_case_id_in_two_splits_is_reported(settings):
    clone = _label_path(settings, "test", "dev-001")
    clone.write_text(_label_path(settings, "dev", "dev-001").read_text())
    shutil.copytree(
        settings.paths.cases_dir / "dev" / "dev-001", settings.paths.cases_dir / "test" / "dev-001"
    )
    assert any("appears in several splits" in e for e in audit_labels(settings).errors)


def test_generated_variants_pass_and_unknown_sources_do_not(settings):
    generate_variants(settings, "dev", "variants")
    assert audit_labels(settings).errors == []
    _edit_label(settings, "variants", "dev-001__late_submission", source_case_id="ghost")
    assert any("unknown source case" in e for e in audit_labels(settings).errors)
