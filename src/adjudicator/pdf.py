"""Typed boundary around PyMuPDF (which ships without complete type information)."""

# pyright: reportUnknownArgumentType=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class RenderedPage:
    number: int  # 1-based
    text: str
    image_path: Path
    has_visual_content: bool


def render_pages(
    pdf_path: Path, image_dir: Path, *, dpi: int, min_drawings_for_visual: int
) -> list[RenderedPage]:
    """Extract each page's text layer, render it to PNG, and flag visual content."""
    image_dir.mkdir(parents=True, exist_ok=True)
    pages: list[RenderedPage] = []
    with pymupdf.open(pdf_path) as document:
        for index in range(document.page_count):
            page = document.load_page(index)
            image_path = image_dir / f"page-{index + 1:03d}.png"
            page.get_pixmap(dpi=dpi).save(image_path)
            visual = bool(page.get_images()) or len(page.get_drawings()) >= min_drawings_for_visual
            pages.append(RenderedPage(index + 1, str(page.get_text("text")), image_path, visual))
    return pages


def extract_text(pdf_path: Path) -> str:
    with pymupdf.open(pdf_path) as document:
        return "\n".join(
            str(document.load_page(i).get_text("text")) for i in range(document.page_count)
        )
