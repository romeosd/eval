"""Helpers for passing images to the Responses API."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from adjudicator.config import ImageDetail


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def input_image(path: Path, detail: ImageDetail) -> dict[str, Any]:
    return {"type": "input_image", "image_url": data_url(path), "detail": detail}


def input_text(text: str) -> dict[str, Any]:
    return {"type": "input_text", "text": text}
