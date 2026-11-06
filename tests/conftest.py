"""Shared fixtures: an isolated project copy and a scripted (offline) model transport."""

from __future__ import annotations

import asyncio
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from adjudicator.agent import Adjudicator
from adjudicator.application.loader import CaseLoader
from adjudicator.config import Settings
from adjudicator.demo.sample_data import generate, sample_cases
from adjudicator.llm.client import (
    BudgetTracker,
    CallContext,
    FunctionCall,
    LLMClient,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from adjudicator.observability import AuditLog
from adjudicator.pipeline import Services
from adjudicator.policy.ingest import PolicyIngestor
from adjudicator.policy.retrieval import PolicySearchIndex
from adjudicator.prompts import PromptLibrary
from adjudicator.schemas import (
    DiagramTranscription,
    ImageTranscription,
    InjectionVerdict,
    SupportVerdict,
)
from adjudicator.security import InjectionScreener
from adjudicator.validation import DecisionValidator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USAGE = TokenUsage(input_tokens=1000, cached_input_tokens=0, output_tokens=200, reasoning_tokens=50)

FLOWCHART_TEXT = (
    "Theft of item reported. Police report obtained within 24 hours of discovering the theft? "
    "No: Decline claim. Was the item attended at the time of theft? No: Decline claim (see 4.2). "
    "Yes: Pay the item value, up to $500 per item and $3,000 in total (see 3.1)."
)

type Responder = Callable[[ModelRequest[Any]], ModelResponse[Any] | None]


def respond(
    parsed: BaseModel | None = None, calls: list[FunctionCall] | None = None
) -> ModelResponse[Any]:
    replay = [
        {"type": "function_call", "call_id": c.call_id, "name": c.name, "arguments": c.arguments}
        for c in calls or []
    ]
    return ModelResponse(
        parsed=parsed, function_calls=calls or [], replay_items=replay, usage=USAGE
    )


def _sample_lines() -> dict[tuple[str, str], list[str]]:
    return {
        (case.case_id, doc.document_id): doc.text_lines
        for case in sample_cases()
        for doc in case.documents
    }


@dataclass
class ScriptedTransport:
    """Offline transport: per-purpose responders, with sensible defaults for helpers."""

    adjudicate: Responder | None = None
    overrides: list[tuple[str, Responder]] = field(default_factory=list[tuple[str, Responder]])
    requests: list[ModelRequest[Any]] = field(default_factory=list[ModelRequest[Any]])
    delay_s: dict[str, float] = field(default_factory=dict[str, float])
    _lines: dict[tuple[str, str], list[str]] = field(default_factory=_sample_lines)

    async def create(self, request: ModelRequest[Any]) -> ModelResponse[Any]:
        response = await self._create(request)
        return replace(response, resolved_model=f"{request.model.name}-snapshot")

    async def _create(self, request: ModelRequest[Any]) -> ModelResponse[Any]:
        self.requests.append(request)
        for prefix, seconds in self.delay_s.items():
            if request.purpose.startswith(prefix):
                await asyncio.sleep(seconds)
        for prefix, responder in self.overrides:
            if request.purpose.startswith(prefix):
                result = responder(request)
                if result is not None:
                    return result
        purpose = request.purpose
        if purpose.startswith("injection_screen"):
            return respond(
                InjectionVerdict(contains_instructions=False, suspicious_spans=[], reason="clean")
            )
        if purpose.startswith("diagram_transcription"):
            return respond(
                DiagramTranscription(
                    has_visual_content=True,
                    title="Figure 5.3 - Theft claim assessment",
                    transcribed_text=FLOWCHART_TEXT,
                    rules_expressed=[
                        "If a police report was not obtained within 24 hours of discovering the "
                        "theft, the claim is declined."
                    ],
                )
            )
        if purpose.startswith("image_transcription"):
            match = re.match(r"image_transcription:([^:]+):(.+)", purpose)
            assert match, purpose
            lines = self._lines.get((match[1].split("__")[0], match[2]), ["[illegible]"])
            return respond(ImageTranscription(visible_text="\n".join(lines), description="scan"))
        if purpose.startswith(("claim_support", "outcome_support")):
            return respond(SupportVerdict(supported=True, explanation="supported"))
        if purpose.startswith("adjudicate") and self.adjudicate is not None:
            result = self.adjudicate(request)
            if result is not None:
                return result
        raise AssertionError(f"no scripted response for {purpose}")

    def purposes(self, prefix: str) -> list[str]:
        return [r.purpose for r in self.requests if r.purpose.startswith(prefix)]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    shutil.copytree(PROJECT_ROOT / "config", tmp_path / "config")
    shutil.copytree(PROJECT_ROOT / "prompts", tmp_path / "prompts")
    return tmp_path


@pytest.fixture
def settings(project: Path) -> Settings:
    loaded = Settings.load(project / "config" / "settings.toml")
    fast = loaded.model_copy(
        update={"loading": loaded.loading.model_copy(update={"retry_backoff_s": 0.0})}
    )
    generate(fast)
    return fast


@pytest.fixture
def transport() -> ScriptedTransport:
    return ScriptedTransport()


@pytest.fixture
def context(tmp_path: Path) -> CallContext:
    return CallContext(audit=AuditLog(tmp_path / "audit.jsonl", scope="test"))


def build_services(settings: Settings, transport: ScriptedTransport) -> Services:
    prompts = PromptLibrary.load(settings.paths.prompts_dir)
    llm = LLMClient(transport, settings, BudgetTracker(settings.run.budget_usd))
    screener = InjectionScreener(llm, settings, prompts)
    audit = AuditLog(settings.paths.cache_dir / "ingest.jsonl", scope="ingest")
    index = asyncio.run(
        PolicyIngestor(llm, screener, settings, prompts).load_or_build(CallContext(audit=audit))
    )
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


@pytest.fixture
def services(settings: Settings, transport: ScriptedTransport) -> Services:
    return build_services(settings, transport)
