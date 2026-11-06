import asyncio

from adjudicator.application.loader import CaseLoader
from adjudicator.llm.client import BudgetTracker, LLMClient
from adjudicator.prompts import PromptLibrary
from adjudicator.security import REDACTION_MARKER, InjectionScreener


def _loader(settings, transport):
    prompts = PromptLibrary.load(settings.paths.prompts_dir)
    llm = LLMClient(transport, settings, BudgetTracker(10))
    return CaseLoader(llm, InjectionScreener(llm, settings, prompts), settings, prompts)


def _load(settings, transport, context, case_id, split="dev"):
    loader = _loader(settings, transport)
    return asyncio.run(loader.load(settings.paths.cases_dir / split / case_id, split, context))


def test_discovery_matches_files(settings, transport):
    assert [p.name for p in _loader(settings, transport).discover("dev")] == [
        f"dev-00{i}" for i in range(1, 9)
    ]


def test_missing_attachment_is_recorded_not_skipped(settings, transport, context):
    case = _load(settings, transport, context, "dev-007")
    [missing] = case.unrecovered_documents
    assert missing.document_id == "receipt_sunglasses"
    assert missing.load_error


def test_injection_in_image_quarantines_raw_image(settings, transport, context):
    case = _load(settings, transport, context, "dev-006")
    letter = case.document("doctor_letter")
    assert letter.quarantined
    assert REDACTION_MARKER in letter.text
    assert "chronic asthma since 2019" in letter.text  # facts survive redaction
    assert case.document("invoice_1").quarantined is False


def test_label_files_never_loaded(settings, transport, context):
    case = _load(settings, transport, context, "dev-001")
    assert all("expected_outcome" not in d.text for d in case.documents)


def test_image_transcriptions_are_cached_across_runs(settings, transport, context):
    _load(settings, transport, context, "dev-002")
    first = len(transport.purposes("image_transcription"))
    _load(settings, transport, context, "dev-002")
    assert first == 2 and len(transport.purposes("image_transcription")) == first
