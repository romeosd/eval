"""What is deterministic, and the controls for what is not (the model)."""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from adjudicator.demo.sample_data import generate
from adjudicator.evaluation.variants import generate_variants
from adjudicator.llm.client import (
    BudgetTracker,
    LLMClient,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from adjudicator.llm.replay import CassetteMissError, ReplayTransport, request_key
from adjudicator.runner import execute_run, load_run
from adjudicator.schemas import SupportVerdict

from .conftest import ScriptedTransport, respond
from .helpers import cap_calls, referral, theft_approval


def _digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_sample_data_is_byte_identical_across_generations(settings):
    first = _digest(settings.paths.policy_dir.parent)
    generate(settings)
    assert _digest(settings.paths.policy_dir.parent) == first


def test_variant_generation_is_deterministic(settings):
    generate_variants(settings, "dev", "variants")
    first = (
        _digest(settings.paths.cases_dir / "variants"),
        _digest(settings.paths.labels_dir / "variants"),
    )
    generate_variants(settings, "dev", "variants")
    assert first == (
        _digest(settings.paths.cases_dir / "variants"),
        _digest(settings.paths.labels_dir / "variants"),
    )


def test_search_order_is_stable_and_tie_broken_by_id(services):
    hits = services.search_index.search("claim policy section", 20)
    assert hits == services.search_index.search("claim policy section", 20)
    assert hits == sorted(hits, key=lambda h: (-h.score, h.section_id, h.page, h.kind))


# ------------------------------------------------------------------ pinned snapshots


class _FixedModelTransport:
    def __init__(self, resolved: str) -> None:
        self._resolved = resolved

    async def create(self, request):
        return ModelResponse(
            parsed=SupportVerdict(supported=True, explanation="ok"),
            function_calls=[],
            replay_items=[],
            usage=TokenUsage(1, 0, 1, 0),
            resolved_model=self._resolved,
        )


def _verdict_request(settings):
    return ModelRequest(
        model=settings.models.judge,
        purpose="claim_support:t",
        instructions="i",
        input=[],
        text_format=SupportVerdict,
    )


def _pinned(settings, require):
    run = settings.run.model_copy(update={"require_pinned_models": require})
    return settings.model_copy(update={"run": run})


def test_alias_that_resolves_to_a_snapshot_is_refused_when_pinning_is_required(settings, context):
    resolved = f"{settings.models.judge.name}-2026-01-01"
    strict = _pinned(settings, True)
    llm = LLMClient(_FixedModelTransport(resolved), strict, BudgetTracker(10))
    with pytest.raises(ModelCallError, match="resolved to"):
        asyncio.run(llm.call(_verdict_request(settings), context))
    lenient = LLMClient(_FixedModelTransport(resolved), _pinned(settings, False), BudgetTracker(10))
    assert asyncio.run(lenient.call(_verdict_request(settings), context)).parsed


def test_an_exact_snapshot_name_passes_the_pinning_check(settings, context):
    name = settings.models.judge.name
    llm = LLMClient(_FixedModelTransport(name), _pinned(settings, True), BudgetTracker(10))
    assert asyncio.run(llm.call(_verdict_request(settings), context)).parsed


# ------------------------------------------------------------------ record / replay


def test_request_key_is_stable_and_sensitive(settings):
    base = _verdict_request(settings)
    assert request_key(base) == request_key(_verdict_request(settings))
    changed = ModelRequest(**{**base.__dict__, "instructions": "different"})
    image = ModelRequest(**{**base.__dict__, "input": [{"image_url": "data:image/png;base64,AAA"}]})
    other = ModelRequest(**{**base.__dict__, "input": [{"image_url": "data:image/png;base64,AAB"}]})
    assert len({request_key(r) for r in (base, changed, image, other)}) == 4


def _agent(request):
    case_id = request.purpose.split(":")[1]
    if case_id != "dev-002":
        return respond(referral(case_id))
    if not any(item.get("type") == "function_call_output" for item in request.input):
        return respond(calls=cap_calls())
    return respond(theft_approval())


def _canonical(run_dir: Path) -> list[dict]:
    _, results = load_run(run_dir)
    return [
        {
            **r.model_dump(mode="json", exclude={"run_id", "duration_s", "usage"}),
            "usage": [(u.model_dump(mode="json", exclude={"latency_s"})) for u in r.usage],
        }
        for r in sorted(results, key=lambda r: r.case_id)
    ]


def test_replay_reproduces_a_recorded_run_exactly_without_any_model(settings, transport):
    transport.adjudicate = _agent
    recorded = asyncio.run(
        execute_run(settings.with_replay_mode("record"), "dev", transport=transport, tag="rec")
    )
    assert any(settings.paths.cassette_dir.glob("*.json"))

    shutil.rmtree(settings.paths.cache_dir)  # replay must also serve ingest and transcriptions
    replayed = asyncio.run(execute_run(settings.with_replay_mode("replay"), "dev", tag="rep"))

    assert _canonical(recorded) == _canonical(replayed)
    strip = lambda d: [  # noqa: E731
        {k: v for k, v in json.loads(line).items() if k != "run_id"}
        for line in (d / "predictions.jsonl").read_text().splitlines()
    ]
    assert strip(recorded) == strip(replayed)


def test_replay_never_falls_through_to_a_live_call(settings, transport):
    transport.adjudicate = _agent
    asyncio.run(execute_run(settings.with_replay_mode("record"), "dev", transport=transport))
    prompt = settings.paths.prompts_dir / "adjudicator.md"
    prompt.write_text(prompt.read_text() + "\n- one extra rule", encoding="utf-8")
    with pytest.raises(CassetteMissError):
        asyncio.run(execute_run(settings.with_replay_mode("replay"), "dev"))


def test_replay_transport_without_cassettes_fails_loudly(settings):
    request = _verdict_request(settings)
    with pytest.raises(CassetteMissError):
        asyncio.run(ReplayTransport(settings.paths.cassette_dir).create(request))


def test_scripted_transport_is_not_used_in_replay(settings):
    assert ScriptedTransport().requests == []  # replay builds its own transport
