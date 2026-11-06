"""Exercise the real OpenAITransport against the SDK with HTTP mocked out.

Verifies request shape (strict structured output, reasoning replay settings) and
that output items are replayed without SDK-only parse artefacts.
"""

import asyncio
import json

import httpx2
from openai import AsyncOpenAI

from adjudicator.llm.client import ModelRequest, OpenAITransport
from adjudicator.schemas import AgentDecision
from adjudicator.tools import CalculateArgs, strict_schema

USAGE = {
    "input_tokens": 100,
    "input_tokens_details": {"cached_tokens": 10},
    "output_tokens": 20,
    "output_tokens_details": {"reasoning_tokens": 5},
    "total_tokens": 120,
}
BASE = {
    "object": "response",
    "created_at": 1,
    "model": "gpt-5.5",
    "status": "completed",
    "usage": USAGE,
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}
DECISION = {
    "case_id": "c1",
    "outcome": "REFER",
    "payable_amount": None,
    "findings": [],
    "conflicts_identified": [],
    "missing_information": [],
    "confidence": "low",
    "rationale": "r",
}
TOOL_TURN = {
    **BASE,
    "id": "r1",
    "output": [
        {"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": "abc"},
        {
            "type": "function_call",
            "id": "fc1",
            "call_id": "call_1",
            "name": "calculate",
            "arguments": '{"operation": "sum", "inputs": [], "datetimes": []}',
            "status": "completed",
        },
    ],
}
FINAL_TURN = {
    **BASE,
    "id": "r2",
    "output": [
        {
            "type": "message",
            "id": "m1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": json.dumps(DECISION), "annotations": []}],
        }
    ],
}


def test_tool_loop_round_trip(settings, monkeypatch):
    monkeypatch.setenv(settings.openai.api_key_env, "sk-test")
    responses = iter([TOOL_TURN, FINAL_TURN])
    bodies: list[dict] = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=next(responses))

    transport = OpenAITransport(settings)
    transport._client = AsyncOpenAI(
        api_key="x", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    )
    tool = {
        "type": "function",
        "name": "calculate",
        "description": "d",
        "parameters": strict_schema(CalculateArgs),
        "strict": True,
    }

    def request(items):
        return ModelRequest(
            model=settings.models.adjudicator,
            purpose="p",
            instructions="i",
            input=items,
            text_format=AgentDecision,
            tools=[tool],
        )

    first = asyncio.run(transport.create(request([{"role": "user", "content": "hi"}])))
    assert first.function_calls[0].name == "calculate"
    assert first.usage.cached_input_tokens == 10
    items = [
        {"role": "user", "content": "hi"},
        *first.replay_items,
        {"type": "function_call_output", "call_id": "call_1", "output": "3"},
    ]
    second = asyncio.run(transport.create(request(items)))
    assert isinstance(second.parsed, AgentDecision)
    assert "parsed" not in json.dumps(second.replay_items)

    sent = bodies[1]
    assert sent["store"] is False and sent["include"] == ["reasoning.encrypted_content"]
    assert sent["text"]["format"]["strict"] is True
    assert [i.get("type", i.get("role")) for i in sent["input"]] == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
