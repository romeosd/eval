"""Versioned prompt templates loaded from ``prompts/*.md``.

Prompts are files, not string literals, so they can be diffed and reviewed; each
prompt's content hash is recorded in run manifests and in the policy cache key.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class PromptName(StrEnum):
    ADJUDICATOR = "adjudicator"
    CLAIM_SUPPORT_JUDGE = "claim_support_judge"
    DIAGRAM_TRANSCRIPTION = "diagram_transcription"
    IMAGE_TRANSCRIPTION = "image_transcription"
    INJECTION_CLASSIFIER = "injection_classifier"


@dataclass(frozen=True)
class Prompt:
    name: PromptName
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:16]


class PromptLibrary:
    def __init__(self, prompts: dict[PromptName, Prompt]) -> None:
        self._prompts = prompts

    @classmethod
    def load(cls, directory: Path) -> PromptLibrary:
        prompts: dict[PromptName, Prompt] = {}
        for name in PromptName:
            path = directory / f"{name.value}.md"
            if not path.is_file():
                raise FileNotFoundError(f"Missing prompt file: {path}")
            prompts[name] = Prompt(name=name, text=path.read_text(encoding="utf-8").strip())
        return cls(prompts)

    def get(self, name: PromptName) -> Prompt:
        return self._prompts[name]

    def hashes(self) -> dict[str, str]:
        return {name.value: prompt.sha256 for name, prompt in self._prompts.items()}
