"""Recover the numbered section hierarchy of a policy document from page text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from adjudicator.schemas import PolicySection

HEADING = re.compile(r"^(?P<number>\d+(?:\.\d+)*)\.?\s+(?P<title>[A-Z][^\n]{1,80})$")
PREAMBLE_NUMBER = "0"


@dataclass
class _OpenSection:
    number: str
    title: str
    page_start: int
    page_end: int
    lines: list[str] = field(default_factory=list[str])


def parse_sections(document_id: str, pages: list[tuple[int, str]]) -> list[PolicySection]:
    """Split pages into sections at numbered headings, keeping page spans and parents."""
    open_sections: list[_OpenSection] = []
    current = _OpenSection(number=PREAMBLE_NUMBER, title="Preamble", page_start=1, page_end=1)
    for page_number, text in pages:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = HEADING.match(line)
            if match:
                open_sections.append(current)
                current = _OpenSection(
                    number=match["number"],
                    title=match["title"].strip(),
                    page_start=page_number,
                    page_end=page_number,
                )
            else:
                current.lines.append(line)
                current.page_end = page_number
    open_sections.append(current)

    sections: list[PolicySection] = []
    for section in open_sections:
        if section.number == PREAMBLE_NUMBER and not section.lines:
            continue
        sections.append(
            PolicySection(
                section_id=f"{document_id}:{section.number}",
                document_id=document_id,
                number=section.number,
                title=section.title,
                level=section.number.count(".") + 1,
                parent_id=_parent_id(document_id, section.number),
                page_start=section.page_start,
                page_end=section.page_end,
                text="\n".join(section.lines),
            )
        )
    return sections


def _parent_id(document_id: str, number: str) -> str | None:
    if "." not in number:
        return None
    return f"{document_id}:{number.rsplit('.', 1)[0]}"
