"""Typed gateway over the OpenAI Responses API.

``LLMClient`` owns cross-cutting concerns (budget guard, timeouts, cost accounting,
audit events). The network call itself sits behind the ``ResponsesTransport``
protocol so the gateway can be tested with a scripted transport.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel

from adjudicator.config import ModelPrice, ModelSpec, Settings
from adjudicator.observability import AuditLog
from adjudicator.schemas import UsageRecord

type InputItem = dict[str, Any]
type ToolDefinition = dict[str, Any]


class BudgetExceededError(RuntimeError):
    """Raised before a model call when the run budget is already spent."""


class ModelCallError(RuntimeError):
    """The model returned no usable output (refusal, empty output, or timeout)."""


@dataclass(frozen=True)
class FunctionCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int


@dataclass(frozen=True)
class ModelRequest[T: BaseModel]:
    model: ModelSpec
    purpose: str
    instructions: str
    input: list[InputItem]
    text_format: type[T]
    tools: list[ToolDefinition] = field(default_factory=list[ToolDefinition])
    timeout_s: float | None = None


@dataclass(frozen=True)
class ModelResponse[T: BaseModel]:
    parsed: T | None
    function_calls: list[FunctionCall]
    replay_items: list[InputItem]  # output items to append to the next request's input
    usage: TokenUsage
    refusal: str | None = None
    resolved_model: str = ""


class ResponsesTransport(Protocol):
    async def create[T: BaseModel](self, request: ModelRequest[T]) -> ModelResponse[T]: ...


@dataclass
class CallContext:
    """Per-case sink for audit events and usage records."""

    audit: AuditLog
    usage: list[UsageRecord] = field(default_factory=list[UsageRecord])


class BudgetTracker:
    def __init__(self, limit_usd: float) -> None:
        self._limit = limit_usd
        self._spent = 0.0
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        return self._spent

    def ensure_available(self) -> None:
        if self._spent >= self._limit:
            raise BudgetExceededError(
                f"Run budget of ${self._limit:.2f} exhausted (spent ${self._spent:.4f})."
            )

    def record(self, cost: float) -> None:
        with self._lock:
            self._spent += cost


def cost_usd(usage: TokenUsage, price: ModelPrice) -> float:
    uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
    return (
        uncached * price.input
        + usage.cached_input_tokens * price.cached_input
        + usage.output_tokens * price.output
    ) / 1_000_000


class LLMClient:
    def __init__(
        self, transport: ResponsesTransport, settings: Settings, budget: BudgetTracker
    ) -> None:
        self._transport = transport
        self._pricing = settings.pricing
        self._require_pinned = settings.run.require_pinned_models
        self._budget = budget

    @property
    def budget(self) -> BudgetTracker:
        return self._budget

    async def call[T: BaseModel](
        self, request: ModelRequest[T], context: CallContext
    ) -> ModelResponse[T]:
        self._budget.ensure_available()
        started = time.perf_counter()
        try:
            if request.timeout_s is None:
                response = await self._transport.create(request)
            else:
                response = await asyncio.wait_for(
                    self._transport.create(request), timeout=request.timeout_s
                )
        except TimeoutError as exc:
            context.audit.event("llm_timeout", model=request.model.name, purpose=request.purpose)
            raise ModelCallError(
                f"{request.purpose}: timed out after {request.timeout_s}s"
            ) from exc
        latency = time.perf_counter() - started

        record = UsageRecord(
            model=request.model.name,
            resolved_model=response.resolved_model or request.model.name,
            purpose=request.purpose,
            input_tokens=response.usage.input_tokens,
            cached_input_tokens=response.usage.cached_input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            cost_usd=cost_usd(response.usage, self._pricing[request.model.name]),
            latency_s=round(latency, 3),
        )
        self._budget.record(record.cost_usd)
        context.usage.append(record)
        context.audit.event(
            "llm_call",
            model=record.model,
            purpose=record.purpose,
            reasoning_effort=request.model.reasoning_effort,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            reasoning_tokens=record.reasoning_tokens,
            cost_usd=record.cost_usd,
            latency_s=record.latency_s,
            function_calls=[call.name for call in response.function_calls],
            refusal=response.refusal,
        )
        if (
            self._require_pinned
            and response.resolved_model
            and response.resolved_model != request.model.name
        ):
            raise ModelCallError(
                f"{request.purpose}: '{request.model.name}' resolved to "
                f"'{response.resolved_model}'; configure that dated snapshot name to pin it"
            )
        if response.refusal is not None:
            raise ModelCallError(f"{request.purpose}: model refused: {response.refusal}")
        if response.parsed is None and not response.function_calls:
            raise ModelCallError(f"{request.purpose}: no structured output or tool calls returned")
        return response


class OpenAITransport:
    """Production transport using ``client.responses.parse`` (structured outputs)."""

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.api_key(),
            timeout=settings.openai.request_timeout_s,
            max_retries=settings.openai.max_retries,
        )
        self._store = settings.openai.store

    async def create[T: BaseModel](self, request: ModelRequest[T]) -> ModelResponse[T]:
        kwargs: dict[str, Any] = {
            "model": request.model.name,
            "instructions": request.instructions,
            "input": request.input,
            "text_format": request.text_format,
            "store": self._store,
        }
        if request.tools:
            kwargs["tools"] = request.tools
        if request.model.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": request.model.reasoning_effort}
            if not self._store:
                # Stateless multi-turn tool use needs reasoning items replayed verbatim.
                kwargs["include"] = ["reasoning.encrypted_content"]

        response = await self._client.responses.parse(**kwargs)

        function_calls: list[FunctionCall] = []
        refusal: str | None = None
        for item in response.output:
            if item.type == "function_call":
                function_calls.append(
                    FunctionCall(call_id=item.call_id, name=item.name, arguments=item.arguments)
                )
            elif item.type == "message":
                for content in item.content:
                    if content.type == "refusal":
                        refusal = content.refusal

        usage = response.usage
        token_usage = TokenUsage(
            input_tokens=usage.input_tokens if usage else 0,
            cached_input_tokens=usage.input_tokens_details.cached_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens if usage else 0,
        )
        replay = [
            _strip_parsed(item.model_dump(mode="json", exclude_none=True))
            for item in response.output
        ]
        return ModelResponse(
            parsed=response.output_parsed,
            function_calls=function_calls,
            replay_items=replay,
            usage=token_usage,
            refusal=refusal,
            resolved_model=str(response.model),
        )


_SDK_ONLY_KEYS = frozenset({"parsed", "parsed_arguments"})


def _strip_parsed(value: Any) -> Any:
    """Remove SDK-side parse artefacts so output items can be replayed as input."""
    if isinstance(value, dict):
        return {k: _strip_parsed(v) for k, v in value.items() if k not in _SDK_ONLY_KEYS}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_strip_parsed(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value
