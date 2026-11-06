"""Lexical (BM25) search over policy sections and diagram transcriptions.

Deliberately not embedding-based: policy wording is precise and the corpus is small,
so exact-term ranking plus agentic navigation (``get_policy_section``) is cheaper,
deterministic and easier to audit. ``retrieval.mode = "full_context"`` bypasses
search entirely so the two strategies can be compared in evaluation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from adjudicator.schemas import PolicyIndex

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
_K1 = 1.5
_B = 0.75


@dataclass(frozen=True)
class SearchHit:
    section_id: str
    title: str
    page: int
    kind: str  # "section" | "diagram"
    score: float
    snippet: str


@dataclass(frozen=True)
class _Entry:
    section_id: str
    title: str
    page: int
    kind: str
    text: str
    terms: Counter[str]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class PolicySearchIndex:
    def __init__(self, index: PolicyIndex) -> None:
        entries: list[_Entry] = []
        for section in index.all_sections():
            body = f"{section.number} {section.title}\n{section.text}"
            entries.append(
                _Entry(
                    section.section_id,
                    section.title,
                    section.page_start,
                    "section",
                    body,
                    Counter(tokenize(body)),
                )
            )
        for page in index.all_pages():
            if page.diagram is None or not page.diagram.has_visual_content:
                continue
            owner = next(
                (
                    s
                    for s in index.all_sections()
                    if s.document_id == page.document_id
                    and s.page_start <= page.page_number <= s.page_end
                ),
                None,
            )
            body = page.searchable_text()
            entries.append(
                _Entry(
                    owner.section_id if owner else f"{page.document_id}:page-{page.page_number}",
                    f"Diagram: {page.diagram.title}",
                    page.page_number,
                    "diagram",
                    body,
                    Counter(tokenize(body)),
                )
            )
        self._entries = entries
        lengths = [sum(e.terms.values()) for e in entries]
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
        doc_freq: Counter[str] = Counter()
        for entry in entries:
            doc_freq.update(entry.terms.keys())
        n = len(entries)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_terms = tokenize(query)
        scored: list[tuple[float, _Entry]] = []
        for entry in self._entries:
            length = sum(entry.terms.values())
            score = 0.0
            for term in query_terms:
                tf = entry.terms.get(term, 0)
                if tf:
                    norm = tf + _K1 * (1 - _B + _B * length / self._avg_len)
                    score += self._idf.get(term, 0.0) * tf * (_K1 + 1) / norm
            if score > 0:
                scored.append((score, entry))
        # Ties break on section id and page, never on insertion order, so results are stable.
        scored.sort(key=lambda pair: (-pair[0], pair[1].section_id, pair[1].page, pair[1].kind))
        return [
            SearchHit(
                section_id=e.section_id,
                title=e.title,
                page=e.page,
                kind=e.kind,
                score=round(s, 3),
                snippet=e.text[:300],
            )
            for s, e in scored[:top_k]
        ]


def render_full_policy(index: PolicyIndex) -> str:
    """Whole policy as one reference block (for ``retrieval.mode = "full_context"``)."""
    parts: list[str] = []
    for section in index.all_sections():
        parts.append(
            f"<section id={section.section_id!r} pages={section.page_start}-{section.page_end}>\n"
            f"{section.number} {section.title}\n{section.text}\n</section>"
        )
    for page in index.all_pages():
        if page.diagram is not None and page.diagram.has_visual_content:
            parts.append(
                f"<diagram document={page.document_id!r} page={page.page_number}>\n"
                f"{page.searchable_text()}\n</diagram>"
            )
    return "\n\n".join(parts)
