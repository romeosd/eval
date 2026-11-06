"""Prompt-injection screening for every piece of untrusted content.

Defence is layered and enforced in code rather than by prompt instructions alone:

1. A deterministic heuristic screen (regexes) — cheap, always available, auditable.
2. An LLM classifier with a structured verdict — catches paraphrased instructions.
3. Redaction: flagged spans are replaced before content reaches the decision model,
   and quarantined spans are remembered so validation can reject any decision that
   cites them.

The same screener runs over application text, application image transcriptions,
policy page text and policy diagram transcriptions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rapidfuzz import fuzz

from adjudicator.config import SecurityConfig, Settings
from adjudicator.llm.client import CallContext, LLMClient, ModelCallError, ModelRequest
from adjudicator.prompts import PromptLibrary, PromptName
from adjudicator.schemas import InjectionFinding, InjectionVerdict

REDACTION_MARKER = "[REDACTED: instruction-like content removed by screener]"

DEFAULT_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+instructions",
    r"disregard\s+.{0,40}?(?:instructions|policy|rules|guidelines)",
    r"\b(?:system|developer)\s+(?:prompt|message|note|override)\b",
    r"\bnote\s+to\s+(?:the\s+)?(?:ai|assistant|model|assessor\s+ai|automated\s+assessor)\b",
    r"\byou\s+(?:are|must)\s+now\b",
    r"\b(?:as\s+an?\s+)?(?:ai|llm|language\s+model)\b.{0,30}\b(?:must|should|will)\b",
    r"\b(?:approve|accept)\s+(?:this|the)\s+claim\s+in\s+full\b",
    r"\bdo\s+not\s+(?:verify|check|validate|mention|flag)\b",
    r"<\s*/?\s*(?:system|instructions?)\s*>",
)

_CLASSIFIER_MAX_CHARS = 12_000


class InjectionScreener:
    def __init__(self, llm: LLMClient, settings: Settings, prompts: PromptLibrary) -> None:
        self._llm = llm
        self._config: SecurityConfig = settings.security
        self._model = settings.models.injection_classifier
        self._prompt = prompts.get(PromptName.INJECTION_CLASSIFIER).text
        self._patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL)
            for p in (*DEFAULT_PATTERNS, *self._config.extra_patterns)
        ]

    def heuristic_findings(self, text: str, source: str) -> list[InjectionFinding]:
        findings: list[InjectionFinding] = []
        for pattern in self._patterns:
            for match in pattern.finditer(text):
                findings.append(
                    InjectionFinding(
                        source=source,
                        span=_expand_to_sentence(text, match.start(), match.end()),
                        method="heuristic",
                        reason=f"matched pattern {pattern.pattern!r}",
                    )
                )
        return findings

    async def screen(self, text: str, source: str, context: CallContext) -> list[InjectionFinding]:
        if not text.strip():
            return []
        findings: list[InjectionFinding] = []
        if self._config.heuristic_screen:
            findings.extend(self.heuristic_findings(text, source))
        if self._config.classifier_screen:
            findings.extend(await self._classify(text, source, context))
        if findings:
            context.audit.event(
                "injection_flagged",
                source=source,
                findings=[f.model_dump(mode="json") for f in findings],
            )
        return findings

    async def _classify(
        self, text: str, source: str, context: CallContext
    ) -> list[InjectionFinding]:
        request = ModelRequest(
            model=self._model,
            purpose=f"injection_screen:{source}",
            instructions=self._prompt,
            input=[
                {
                    "role": "user",
                    "content": f"<untrusted_content source={source!r}>\n"
                    f"{text[:_CLASSIFIER_MAX_CHARS]}\n</untrusted_content>",
                }
            ],
            text_format=InjectionVerdict,
        )
        try:
            response = await self._llm.call(request, context)
        except ModelCallError as exc:
            if not self._config.fail_closed:
                return []
            return [
                InjectionFinding(source=source, span="", method="classifier_error", reason=str(exc))
            ]
        verdict = response.parsed
        if verdict is None or not verdict.contains_instructions:
            return []
        located = [_locate(span, text) for span in verdict.suspicious_spans if span.strip()]
        # An unlocatable span cannot be redacted; "" marks the content as unscreened.
        spans = [span if span is not None else "" for span in located] or [""]
        return [
            InjectionFinding(source=source, span=span, method="classifier", reason=verdict.reason)
            for span in spans
        ]


def redact(text: str, findings: Sequence[InjectionFinding]) -> str:
    redacted = text
    for span in sorted({f.span for f in findings if f.span}, key=len, reverse=True):
        redacted = redacted.replace(span, REDACTION_MARKER)
    return redacted


def has_unscreened_content(findings: Sequence[InjectionFinding]) -> bool:
    """True if the classifier failed or flagged content without locating the span."""
    return any(f.method == "classifier_error" or not f.span for f in findings)


def _locate(span: str, text: str, min_score: float = 90) -> str | None:
    """Map a classifier-reported span to the exact substring of ``text`` it refers to."""
    if span in text:
        return span
    alignment = fuzz.partial_ratio_alignment(span, text)
    if alignment is None or alignment.score < min_score:
        return None
    return text[alignment.dest_start : alignment.dest_end]


def _expand_to_sentence(text: str, start: int, end: int) -> str:
    """Grow a regex match to the surrounding sentence/line so redaction removes the payload."""
    boundaries = ".!?\n"
    left = max((text.rfind(ch, 0, start) for ch in boundaries), default=-1) + 1
    right_candidates = [i for ch in boundaries if (i := text.find(ch, end)) != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return text[left:right].strip()
