"""The adjudication agent: a tool-calling loop that ends in a structured decision.

The model navigates policy and application evidence through tools and must finish
with an ``AgentDecision`` (structured output). Conversation state is the list of
Responses API items — never a flattened text transcript — so reasoning items and
tool calls are replayed exactly.
"""

from __future__ import annotations

import json

from adjudicator.config import Settings
from adjudicator.llm.client import CallContext, InputItem, LLMClient, ModelRequest
from adjudicator.policy.retrieval import render_full_policy
from adjudicator.prompts import PromptLibrary, PromptName
from adjudicator.schemas import AgentDecision, Case, PolicyIndex
from adjudicator.tools import Toolbox


class AgentError(RuntimeError):
    """The agent did not produce a decision within its limits."""


class Adjudicator:
    def __init__(
        self, llm: LLMClient, settings: Settings, prompts: PromptLibrary, index: PolicyIndex
    ) -> None:
        self._llm = llm
        self._settings = settings
        instructions = prompts.get(PromptName.ADJUDICATOR).text
        if settings.retrieval.mode == "full_context":
            instructions += (
                "\n\n# Policy reference (complete, screened)\n"
                "The policy search tools are disabled; the full policy is below. "
                "Use view_policy_page for diagram pages.\n\n" + render_full_policy(index)
            )
        self._instructions = instructions

    async def decide(
        self,
        case: Case,
        toolbox: Toolbox,
        context: CallContext,
        *,
        run: int,
        attempt: int,
        feedback: list[str],
        previous: AgentDecision | None,
    ) -> AgentDecision:
        items: list[InputItem] = [
            {"role": "user", "content": _case_brief(case, attempt, feedback, previous)}
        ]
        tools = toolbox.definitions()
        detail = self._settings.retrieval.image_detail
        for turn in range(1, self._settings.agent.max_tool_turns + 1):
            response = await self._llm.call(
                ModelRequest(
                    model=self._settings.models.adjudicator,
                    purpose=f"adjudicate:{case.case_id}:run{run}:attempt{attempt}:turn{turn}",
                    instructions=self._instructions,
                    input=items,
                    text_format=AgentDecision,
                    tools=tools,
                ),
                context,
            )
            items.extend(response.replay_items)
            if response.function_calls:
                for call in response.function_calls:
                    result = toolbox.execute(call)
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": result.as_output(detail),
                        }
                    )
                continue
            if response.parsed is not None:
                context.audit.event("decision_candidate", attempt=attempt, turns=turn)
                return response.parsed
        raise AgentError(
            f"{case.case_id}: no decision after {self._settings.agent.max_tool_turns} tool turns"
        )


def _case_brief(
    case: Case, attempt: int, feedback: list[str], previous: AgentDecision | None
) -> str:
    documents = "\n".join(
        f"- {d.document_id} ({d.kind.value}, {d.load_status}"
        f"{', QUARANTINED' if d.quarantined else ''}): {d.description}"
        for d in case.documents
    )
    form = next(d.text for d in case.documents if d.document_id == "form")
    brief = (
        f"Case id: {case.case_id}\n\nApplication documents:\n{documents}\n\n"
        "<untrusted_application_form>\n"
        f"{form}\n"
        "</untrusted_application_form>"
    )
    if attempt > 1 and previous is not None:
        problems = "\n".join(f"- {line}" for line in feedback)
        brief += (
            f"\n\nThis is attempt {attempt}. Your previous decision was REJECTED by automated "
            f"validation for these reasons:\n{problems}\n\nPrevious decision:\n"
            f"{json.dumps(previous.model_dump(mode='json'), indent=2)}\n\n"
            "Re-check the evidence with the tools and produce a corrected decision. If the "
            "problems cannot be resolved from the available evidence, return REFER."
        )
    return brief
