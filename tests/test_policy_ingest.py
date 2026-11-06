import asyncio

from adjudicator.llm.client import BudgetTracker, CallContext, LLMClient
from adjudicator.observability import AuditLog
from adjudicator.policy.ingest import PolicyIngestor
from adjudicator.policy.retrieval import PolicySearchIndex
from adjudicator.prompts import PromptLibrary
from adjudicator.security import InjectionScreener


def _ingestor(settings, transport):
    prompts = PromptLibrary.load(settings.paths.prompts_dir)
    llm = LLMClient(transport, settings, BudgetTracker(10))
    return PolicyIngestor(llm, InjectionScreener(llm, settings, prompts), settings, prompts)


def _build(settings, transport, tmp_path):
    context = CallContext(audit=AuditLog(tmp_path / "ingest.jsonl", scope="ingest"))
    return asyncio.run(_ingestor(settings, transport).load_or_build(context))


def test_hierarchy_and_page_traceability(settings, transport, tmp_path):
    index = _build(settings, transport, tmp_path)
    theft = index.section("travel-policy:5.3")
    assert theft is not None
    assert (theft.page_start, theft.parent_id) == (4, "travel-policy:5")
    assert index.section("travel-policy:3.1").page_start == 2


def test_only_visual_pages_are_transcribed_and_diagram_is_indexed(settings, transport, tmp_path):
    index = _build(settings, transport, tmp_path)
    assert transport.purposes("diagram_transcription") == [
        "diagram_transcription:policy:travel-policy:p4"
    ]
    page = index.page("travel-policy", 4)
    assert page.diagram is not None and "24 hours" in page.diagram.transcribed_text
    hits = PolicySearchIndex(index).search("police report within 24 hours", 3)
    assert hits[0].kind == "diagram" and hits[0].section_id == "travel-policy:5.3"


def test_cache_reused_then_invalidated_by_prompt_change(settings, transport, tmp_path):
    first = _build(settings, transport, tmp_path)
    calls_after_first = len(transport.requests)
    second = _build(settings, transport, tmp_path)
    assert second.cache_key == first.cache_key
    assert len(transport.requests) == calls_after_first  # no model calls on cache hit

    prompt = settings.paths.prompts_dir / "diagram_transcription.md"
    prompt.write_text(prompt.read_text() + "\n- Extra rule.", encoding="utf-8")
    third = _build(settings, transport, tmp_path)
    assert third.cache_key != first.cache_key
    assert len(transport.requests) > calls_after_first
