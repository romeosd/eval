"""Typed configuration loaded from ``config/settings.toml``.

All tunable values live in the TOML file; this module only defines their shape and
resolves paths. Secrets come from the environment (or a ``.env`` file), never the TOML.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
ImageDetail = Literal["low", "high", "auto"]
RetrievalMode = Literal["tools", "full_context"]

DEFAULT_CONFIG_PATH = Path("config/settings.toml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Section):
    policy_dir: Path
    cases_dir: Path
    labels_dir: Path
    cache_dir: Path
    cassette_dir: Path
    runs_dir: Path
    prompts_dir: Path

    def resolved(self, root: Path) -> PathsConfig:
        return PathsConfig(
            **{name: (root / value).resolve() for name, value in self.model_dump().items()}
        )


class OpenAIConfig(_Section):
    api_key_env: str
    request_timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)
    store: bool


class ModelSpec(_Section):
    name: str
    reasoning_effort: ReasoningEffort | None = None


class ModelsConfig(_Section):
    adjudicator: ModelSpec
    judge: ModelSpec
    vision: ModelSpec
    injection_classifier: ModelSpec


class RetrievalConfig(_Section):
    mode: RetrievalMode
    search_top_k: int = Field(ge=1, le=20)
    image_detail: ImageDetail


class AgentConfig(_Section):
    max_tool_turns: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    consensus_runs: int = Field(ge=1, le=5)


class ValidationConfig(_Section):
    claim_support_check: bool
    judge_sees_diagram_images: bool
    max_revalidations: int = Field(ge=0)
    citation_min_similarity: float = Field(ge=0, le=100)
    judge_timeout_s: float = Field(gt=0)
    amount_tolerance: float = Field(ge=0)
    min_quote_chars: int = Field(ge=1)


class AuditConfig(_Section):
    tool_args_chars: int = Field(ge=100)
    tool_output_chars: int = Field(ge=100)


class SecurityConfig(_Section):
    heuristic_screen: bool
    classifier_screen: bool
    fail_closed: bool
    extra_patterns: tuple[str, ...] = ()


class LoadingConfig(_Section):
    max_load_retries: int = Field(ge=1)
    retry_backoff_s: float = Field(ge=0)


class RunConfig(_Section):
    concurrency: int = Field(ge=1)
    budget_usd: float = Field(gt=0)
    require_pinned_models: bool  # refuse an alias that resolves to a different snapshot id


ReplayMode = Literal["off", "record", "replay"]


class ReplayConfig(_Section):
    mode: ReplayMode


class EvalConfig(_Section):
    held_out_split: str


class IngestConfig(_Section):
    min_drawings_for_visual: int = Field(ge=1)
    render_dpi: int = Field(ge=72, le=300)


class ModelPrice(_Section):
    """USD per one million tokens."""

    input: float = Field(ge=0)
    cached_input: float = Field(ge=0)
    output: float = Field(ge=0)


class Settings(_Section):
    paths: PathsConfig
    openai: OpenAIConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    agent: AgentConfig
    validation: ValidationConfig
    audit: AuditConfig
    security: SecurityConfig
    loading: LoadingConfig
    run: RunConfig
    replay: ReplayConfig
    eval: EvalConfig
    ingest: IngestConfig
    pricing: dict[str, ModelPrice]

    @model_validator(mode="after")
    def _every_model_is_priced(self) -> Settings:
        missing = {spec.name for spec in self.model_specs()} - set(self.pricing)
        if missing:
            raise ValueError(
                f"No [pricing] entry for model(s) {sorted(missing)}; the budget guard needs one."
            )
        return self

    def model_specs(self) -> list[ModelSpec]:
        m = self.models
        return [m.adjudicator, m.judge, m.vision, m.injection_classifier]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> Settings:
        config_path = path.resolve()
        root = config_path.parent.parent
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        settings = cls.model_validate(raw)
        _load_dotenv(root / ".env")
        return settings.model_copy(update={"paths": settings.paths.resolved(root)})

    def fingerprint(self) -> str:
        """Stable hash of every setting; recorded in each run manifest."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]

    def api_key(self) -> str:
        key = os.environ.get(self.openai.api_key_env)
        if not key:
            raise RuntimeError(
                f"Environment variable {self.openai.api_key_env} is not set "
                "(export it or add it to .env in the project root)."
            )
        return key

    def behavior_fingerprint(self) -> str:
        """Hash of every setting that can change a decision (not paths, budget or logging).

        Resuming a run is only valid when this is unchanged: mixing behaviours in one run
        would make its scores meaningless.
        """
        behaviour = self.model_dump(
            mode="json",
            include={"models", "retrieval", "agent", "validation", "security", "loading", "ingest"},
        )
        return hashlib.sha256(json.dumps(behaviour, sort_keys=True).encode()).hexdigest()[:16]

    def with_replay_mode(self, mode: ReplayMode) -> Settings:
        return self.model_copy(update={"replay": ReplayConfig(mode=mode)})

    def with_adjudicator(self, spec: ModelSpec) -> Settings:
        return self.model_copy(
            update={"models": self.models.model_copy(update={"adjudicator": spec})}
        )

    def with_retrieval_mode(self, mode: RetrievalMode) -> Settings:
        return self.model_copy(
            update={"retrieval": self.retrieval.model_copy(update={"mode": mode})}
        )


def _load_dotenv(path: Path) -> None:
    """Minimal .env support: KEY=VALUE lines; existing environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
