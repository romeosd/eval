"""Load a case folder into a screened, model-ready ``Case``.

Layout: ``<cases_dir>/<split>/<case_id>/form.json`` plus the attachment files the form
references. Labels live elsewhere (``labels_dir``) and are never read here.

Every attachment read is retried; anything that still cannot be loaded (or
transcribed) is recorded as ``load_status="failed"`` rather than silently skipped,
so validation can force a referral instead of deciding on incomplete evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from adjudicator.config import Settings
from adjudicator.images import input_image
from adjudicator.llm.client import CallContext, LLMClient, ModelCallError, ModelRequest
from adjudicator.pdf import extract_text
from adjudicator.prompts import PromptLibrary, PromptName
from adjudicator.schemas import ApplicationDocument, Case, DocumentKind, ImageTranscription
from adjudicator.security import InjectionScreener, redact

FORM_FILE = "form.json"
FORM_DOCUMENT_ID = "form"
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
TEXT_SUFFIXES = frozenset({".txt", ".md"})

logger = logging.getLogger(__name__)


_FORM_ADAPTER: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


class CaseLoadError(RuntimeError):
    pass


class AttachmentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    file: str
    description: str


_ATTACHMENTS_ADAPTER = TypeAdapter(list[AttachmentRef])


class CaseLoader:
    def __init__(
        self,
        llm: LLMClient,
        screener: InjectionScreener,
        settings: Settings,
        prompts: PromptLibrary,
    ) -> None:
        self._llm = llm
        self._screener = screener
        self._settings = settings
        self._transcription_prompt = prompts.get(PromptName.IMAGE_TRANSCRIPTION)
        self._transcription_cache = settings.paths.cache_dir / "transcriptions"

    def discover(self, split: str) -> list[Path]:
        split_dir = self._settings.paths.cases_dir / split
        if not split_dir.is_dir():
            raise CaseLoadError(f"Split directory not found: {split_dir}")
        return sorted(p.parent for p in split_dir.glob(f"*/{FORM_FILE}"))

    async def load(self, case_dir: Path, split: str, context: CallContext) -> Case:
        form, attachments = self._read_form(case_dir)
        case_id = case_dir.name
        form_text = json.dumps(form, indent=2, ensure_ascii=False)
        form_findings = await self._screener.screen(form_text, f"{case_id}:form", context)
        documents = [
            ApplicationDocument(
                document_id=FORM_DOCUMENT_ID,
                kind=DocumentKind.FORM,
                description="Structured application form",
                path=case_dir / FORM_FILE,
                load_status="loaded",
                load_error=None,
                text=redact(form_text, form_findings),
                image_path=None,
                quarantined=False,
                injection_findings=form_findings,
            )
        ]
        documents.extend(
            await asyncio.gather(
                *(self._load_attachment(case_dir, ref, context) for ref in attachments)
            )
        )
        for document in documents:
            context.audit.event(
                "document_loaded",
                document_id=document.document_id,
                status=document.load_status,
                error=document.load_error,
                quarantined=document.quarantined,
            )
        return Case(case_id=case_id, split=split, form=form, documents=documents)

    def _read_form(self, case_dir: Path) -> tuple[dict[str, JsonValue], list[AttachmentRef]]:
        path = case_dir / FORM_FILE
        try:
            form = _FORM_ADAPTER.validate_json(path.read_bytes())
            refs = _ATTACHMENTS_ADAPTER.validate_python(form.get("attachments") or [])
        except OSError as exc:
            raise CaseLoadError(f"{path}: unreadable form: {exc}") from exc
        except ValidationError as exc:
            raise CaseLoadError(f"{path}: invalid form: {exc}") from exc
        return form, refs

    async def _load_attachment(
        self, case_dir: Path, ref: AttachmentRef, context: CallContext
    ) -> ApplicationDocument:
        path = case_dir / ref.file
        suffix = path.suffix.lower()
        kind = (
            DocumentKind.IMAGE
            if suffix in IMAGE_SUFFIXES
            else DocumentKind.PDF
            if suffix == ".pdf"
            else DocumentKind.TEXT
        )
        try:
            await self._read_with_retry(path, context)
            if kind is DocumentKind.IMAGE:
                return await self._image_document(case_dir.name, ref, path, context)
            text = _pdf_text(path) if kind is DocumentKind.PDF else path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ModelCallError, CaseLoadError) as exc:
            logger.warning("%s/%s could not be loaded: %s", case_dir.name, ref.document_id, exc)
            return _failed(ref, kind, path, str(exc))

        findings = await self._screener.screen(text, f"{case_dir.name}:{ref.document_id}", context)
        return ApplicationDocument(
            document_id=ref.document_id,
            kind=kind,
            description=ref.description,
            path=path,
            load_status="loaded",
            load_error=None,
            text=redact(text, findings),
            image_path=None,
            quarantined=False,
            injection_findings=findings,
        )

    def _transcription_key(self, path: Path) -> str:
        material = json.dumps(
            {
                "file": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model": self._settings.models.vision.model_dump(),
                "prompt": self._transcription_prompt.sha256,
                "detail": self._settings.retrieval.image_detail,
            },
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()[:24]

    async def _transcribe(
        self, case_id: str, ref: AttachmentRef, path: Path, context: CallContext
    ) -> ImageTranscription:
        """Vision transcription, cached by image bytes + model + prompt so reruns cost nothing."""
        cache_file = self._transcription_cache / f"{self._transcription_key(path)}.json"
        if cache_file.is_file():
            context.audit.event("transcription_cache_hit", document_id=ref.document_id)
            return ImageTranscription.model_validate_json(cache_file.read_text("utf-8"))
        request = ModelRequest(
            model=self._settings.models.vision,
            purpose=f"image_transcription:{case_id}:{ref.document_id}",
            instructions=self._transcription_prompt.text,
            input=[
                {
                    "role": "user",
                    "content": [input_image(path, self._settings.retrieval.image_detail)],
                }
            ],
            text_format=ImageTranscription,
        )
        response = await self._llm.call(request, context)
        if response.parsed is None:
            raise CaseLoadError("image transcription returned no output")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(response.parsed.model_dump_json(), "utf-8")
        tmp.replace(cache_file)
        return response.parsed

    async def _image_document(
        self, case_id: str, ref: AttachmentRef, path: Path, context: CallContext
    ) -> ApplicationDocument:
        transcription = await self._transcribe(case_id, ref, path, context)
        transcript = (
            f"[Image transcription]\nVisible text:\n{transcription.visible_text}\n"
            f"Description: {transcription.description}"
        )
        findings = await self._screener.screen(transcript, f"{case_id}:{ref.document_id}", context)
        return ApplicationDocument(
            document_id=ref.document_id,
            kind=DocumentKind.IMAGE,
            description=ref.description,
            path=path,
            load_status="loaded",
            load_error=None,
            text=redact(transcript, findings),
            image_path=path,
            # The raw pixels still carry the payload, so a flagged image is withheld.
            quarantined=bool(findings),
            injection_findings=findings,
        )

    async def _read_with_retry(self, path: Path, context: CallContext) -> None:
        attempts = self._settings.loading.max_load_retries
        for attempt in range(1, attempts + 1):
            try:
                if path.stat().st_size == 0:
                    raise CaseLoadError(f"{path.name} is empty")
                return
            except FileNotFoundError:
                context.audit.event("load_retry", file=path.name, attempt=attempt)
                if attempt == attempts:
                    raise
                await asyncio.sleep(self._settings.loading.retry_backoff_s * attempt)


def _pdf_text(path: Path) -> str:
    text = extract_text(path)
    if not text.strip():
        raise CaseLoadError(f"{path.name} has no text layer (scanned PDFs: supply as images)")
    return text


def _failed(ref: AttachmentRef, kind: DocumentKind, path: Path, error: str) -> ApplicationDocument:
    return ApplicationDocument(
        document_id=ref.document_id,
        kind=kind,
        description=ref.description,
        path=path,
        load_status="failed",
        load_error=error,
        text="",
        image_path=None,
        quarantined=False,
        injection_findings=[],
    )
