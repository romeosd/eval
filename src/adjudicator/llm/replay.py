"""Record and replay model calls, so everything except the model can be proven deterministic.

* ``record``: every response is stored under a hash of the complete request (model, effort,
  instructions, input including image bytes, tools, output schema, purpose).
* ``replay``: only recorded responses are served. There is no API key, no spend, and a
  request that was not recorded is an error, never a silent fall-through to the network.

Replay reproduces a run exactly, so a change to validators, scoring or orchestration can be
regression-tested against *real* model outputs. It cannot measure model variance: to do
that, run repeats with replay ``off``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from adjudicator.config import Settings
from adjudicator.llm.client import (
    FunctionCall,
    ModelRequest,
    ModelResponse,
    OpenAITransport,
    ResponsesTransport,
    TokenUsage,
)


class CassetteMissError(RuntimeError):
    """Replay was asked for a request that was never recorded."""


def request_key(request: ModelRequest[Any]) -> str:
    payload = {
        "model": request.model.model_dump(),
        "purpose": request.purpose,
        "instructions": request.instructions,
        "input": request.input,
        "tools": request.tools,
        "text_format": request.text_format.model_json_schema(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _dump(request: ModelRequest[Any], response: ModelResponse[Any]) -> dict[str, Any]:
    return {
        "purpose": request.purpose,
        "model": request.model.name,
        "response": {
            "parsed": response.parsed.model_dump(mode="json") if response.parsed else None,
            "function_calls": [asdict(c) for c in response.function_calls],
            "replay_items": response.replay_items,
            "usage": asdict(response.usage),
            "refusal": response.refusal,
            "resolved_model": response.resolved_model,
        },
    }


def _load[T: BaseModel](request: ModelRequest[T], data: dict[str, Any]) -> ModelResponse[T]:
    body = data["response"]
    return ModelResponse(
        parsed=request.text_format.model_validate(body["parsed"]) if body["parsed"] else None,
        function_calls=[FunctionCall(**c) for c in body["function_calls"]],
        replay_items=body["replay_items"],
        usage=TokenUsage(**body["usage"]),
        refusal=body["refusal"],
        resolved_model=body["resolved_model"],
    )


class RecordingTransport:
    def __init__(self, inner: ResponsesTransport, directory: Path) -> None:
        self._inner = inner
        self._directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    async def create[T: BaseModel](self, request: ModelRequest[T]) -> ModelResponse[T]:
        response = await self._inner.create(request)
        path = self._directory / f"{request_key(request)}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_dump(request, response), ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written cassette
        return response


class ReplayTransport:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    async def create[T: BaseModel](self, request: ModelRequest[T]) -> ModelResponse[T]:
        path = self._directory / f"{request_key(request)}.json"
        if not path.is_file():
            raise CassetteMissError(
                f"{request.purpose}: no recorded response for this exact request in "
                f"{self._directory} (a prompt, schema, input or setting changed?)"
            )
        return _load(request, json.loads(path.read_text(encoding="utf-8")))


def build_transport(
    settings: Settings, base: ResponsesTransport | None = None
) -> ResponsesTransport:
    """Transport for the configured replay mode. ``base`` replaces the live OpenAI transport."""
    match settings.replay.mode:
        case "replay":
            return ReplayTransport(settings.paths.cassette_dir)
        case "record":
            return RecordingTransport(
                base or OpenAITransport(settings), settings.paths.cassette_dir
            )
        case "off":
            return base or OpenAITransport(settings)
