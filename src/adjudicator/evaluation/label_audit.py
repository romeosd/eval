"""Audit the evaluation data itself, deterministically and without any model.

An accuracy number is only as good as its labels and splits. This checks that every case has
exactly one label, that labels are internally consistent and refer to real policy sections,
that variants point at a real source case, and that the same case does not appear in two
splits (leakage between development and held-out data).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from adjudicator.config import Settings
from adjudicator.pdf import extract_text
from adjudicator.policy.sections import parse_sections
from adjudicator.schemas import CaseLabel, Outcome

MIN_PER_CLASS = 2


@dataclass
class AuditReport:
    errors: list[str] = field(default_factory=list[str])
    notes: list[str] = field(default_factory=list[str])


def policy_section_numbers(policy_dir: Path) -> set[str]:
    numbers: set[str] = set()
    for pdf in sorted(policy_dir.glob("*.pdf")):
        numbers.update(s.number for s in parse_sections(pdf.stem, [(1, extract_text(pdf))]))
    return numbers


def _read_labels(directory: Path) -> dict[str, CaseLabel]:
    return {
        p.stem: CaseLabel.model_validate_json(p.read_text("utf-8"))
        for p in sorted(directory.glob("*.json"))
    }


def _label_errors(split: str, stem: str, label: CaseLabel, numbers: set[str]) -> list[str]:
    errors: list[str] = []
    where = f"{split}/{stem}"
    if label.case_id != stem:
        errors.append(f"{where}: label case_id {label.case_id!r} does not match its file name")
    amount = label.expected_payable_amount
    if label.expected_outcome is Outcome.APPROVE and (amount is None or amount <= 0):
        errors.append(f"{where}: APPROVE needs a positive expected_payable_amount")
    if label.expected_outcome is not Outcome.APPROVE and amount is not None:
        errors.append(f"{where}: {label.expected_outcome.value} must not have an amount")
    for section in label.expected_sections:
        if not any(n == section or n.startswith(f"{section}.") for n in numbers):
            errors.append(f"{where}: expected section {section!r} is not in the policy")
    if label.synthetic and not label.source_case_id:
        errors.append(f"{where}: synthetic label without source_case_id")
    return errors


def audit_labels(settings: Settings) -> AuditReport:
    report = AuditReport()
    numbers = policy_section_numbers(settings.paths.policy_dir)
    splits = sorted(p.name for p in settings.paths.labels_dir.iterdir() if p.is_dir())
    all_labels = {split: _read_labels(settings.paths.labels_dir / split) for split in splits}
    known_ids = {stem for labels in all_labels.values() for stem in labels}
    originals: dict[str, list[str]] = {}

    for split, labels in all_labels.items():
        case_ids = {p.parent.name for p in (settings.paths.cases_dir / split).glob("*/form.json")}
        report.errors += [f"{split}/{c}: case has no label" for c in sorted(case_ids - set(labels))]
        report.errors += [f"{split}/{c}: label has no case" for c in sorted(set(labels) - case_ids)]
        for stem, label in labels.items():
            report.errors += _label_errors(split, stem, label, numbers)
            if label.source_case_id and label.source_case_id not in known_ids:
                report.errors.append(
                    f"{split}/{stem}: unknown source case {label.source_case_id!r}"
                )
            if not label.synthetic and stem in case_ids:
                form = settings.paths.cases_dir / split / stem / "form.json"
                digest = hashlib.sha256(form.read_bytes()).hexdigest()
                originals.setdefault(digest, []).append(f"{split}/{stem}")
        counts = Counter(label.expected_outcome for label in labels.values())
        report.notes.append(
            f"{split}: {len(labels)} cases, "
            + ", ".join(f"{o.value}={counts.get(o, 0)}" for o in Outcome)
        )
        if not any(label.synthetic for label in labels.values()):
            thin = [o.value for o in Outcome if counts.get(o, 0) < MIN_PER_CLASS]
            if thin:
                report.notes.append(f"{split}: fewer than {MIN_PER_CLASS} cases for {thin}")

    for stem in sorted({s for labels in all_labels.values() for s in labels}):
        in_splits = [sp for sp, labels in all_labels.items() if stem in labels]
        if len(in_splits) > 1:
            report.errors.append(f"case id {stem!r} appears in several splits: {in_splits}")
    for locations in originals.values():
        if len(locations) > 1:
            report.errors.append(f"identical form in {locations}: possible leakage between splits")
    return report
