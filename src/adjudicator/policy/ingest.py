"""Build (or reuse) the processed policy index.

Each policy PDF is processed page by page: native text is extracted, the page is
rendered to an image, and pages containing visual content (embedded images or
vector drawings) are transcribed by a vision model so diagrams are never silently
dropped. All extracted text — including diagram transcriptions — is screened for
injected instructions and redacted before indexing.

The index is cached under a key that hashes every input that could change it: the
source bytes, the vision model, the transcription and screening prompts, the
security/ingest settings and the ingest code version. Any change triggers a rebuild.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

from adjudicator.config import Settings
from adjudicator.images import input_image, input_text
from adjudicator.llm.client import CallContext, LLMClient, ModelRequest
from adjudicator.pdf import RenderedPage, render_pages
from adjudicator.policy.sections import parse_sections
from adjudicator.prompts import PromptLibrary, PromptName
from adjudicator.schemas import (
    DiagramTranscription,
    InjectionFinding,
    PolicyDocument,
    PolicyIndex,
    PolicyPage,
)
from adjudicator.security import InjectionScreener, has_unscreened_content, redact

INGEST_VERSION = "1"
logger = logging.getLogger(__name__)


class PolicyIngestError(RuntimeError):
    pass


class PolicyIngestor:
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
        self._prompts = prompts
        self._cache_root = settings.paths.cache_dir / "policy"

    def source_files(self) -> list[Path]:
        files = sorted(self._settings.paths.policy_dir.glob("*.pdf"))
        if not files:
            raise PolicyIngestError(f"No policy PDFs found in {self._settings.paths.policy_dir}")
        return files

    def cache_key(self) -> str:
        material = {
            "ingest_version": INGEST_VERSION,
            "sources": [(p.name, _sha256(p)) for p in self.source_files()],
            "vision_model": self._settings.models.vision.model_dump(),
            "classifier_model": self._settings.models.injection_classifier.model_dump(),
            "prompts": {
                n.value: self._prompts.get(n).sha256
                for n in (PromptName.DIAGRAM_TRANSCRIPTION, PromptName.INJECTION_CLASSIFIER)
            },
            "security": self._settings.security.model_dump(),
            "ingest": self._settings.ingest.model_dump(),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:20]

    async def load_or_build(self, context: CallContext) -> PolicyIndex:
        key = self.cache_key()
        cache_file = self._cache_root / f"index-{key}.json"
        if cache_file.is_file():
            index = PolicyIndex.model_validate_json(cache_file.read_text(encoding="utf-8"))
            if index.cache_key == key and all(p.image_path.is_file() for p in index.all_pages()):
                logger.info("Reusing processed policy index %s", key)
                context.audit.event("policy_cache_hit", cache_key=key)
                return index
            logger.warning("Policy cache %s is inconsistent; rebuilding", key)

        logger.info("Building policy index %s", key)
        context.audit.event("policy_cache_miss", cache_key=key)
        documents = [await self._build_document(p, key, context) for p in self.source_files()]
        index = PolicyIndex(cache_key=key, ingest_version=INGEST_VERSION, documents=documents)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(index.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(cache_file)  # atomic: a crashed build never leaves a half-written cache
        return index

    async def _build_document(
        self, pdf_path: Path, key: str, context: CallContext
    ) -> PolicyDocument:
        document_id = _slug(pdf_path.stem)
        rendered = render_pages(
            pdf_path,
            self._cache_root / "pages" / key / document_id,
            dpi=self._settings.ingest.render_dpi,
            min_drawings_for_visual=self._settings.ingest.min_drawings_for_visual,
        )
        pages = await asyncio.gather(
            *(self._process_page(document_id, page, context) for page in rendered)
        )
        sections = parse_sections(document_id, [(p.page_number, p.text) for p in pages])
        return PolicyDocument(
            document_id=document_id,
            source_path=pdf_path,
            source_sha256=_sha256(pdf_path),
            pages=list(pages),
            sections=sections,
        )

    async def _process_page(
        self, document_id: str, page: RenderedPage, context: CallContext
    ) -> PolicyPage:
        source = f"policy:{document_id}:p{page.number}"
        text = page.text
        diagram = (
            await self._transcribe(text, page.image_path, source, context)
            if page.has_visual_content
            else None
        )

        findings: list[InjectionFinding] = await self._screener.screen(text, source, context)
        if diagram is not None:
            diagram_text = "\n".join(
                [diagram.title, diagram.transcribed_text, *diagram.rules_expressed]
            )
            diagram_findings = await self._screener.screen(
                diagram_text, f"{source}:diagram", context
            )
            findings.extend(diagram_findings)
            diagram = DiagramTranscription(
                has_visual_content=diagram.has_visual_content,
                title=redact(diagram.title, diagram_findings),
                transcribed_text=redact(diagram.transcribed_text, diagram_findings),
                rules_expressed=[redact(r, diagram_findings) for r in diagram.rules_expressed],
            )
        if has_unscreened_content(findings):
            # Policy content is the reference standard; if it cannot be screened we stop.
            raise PolicyIngestError(f"{source}: content could not be screened for injection")
        return PolicyPage(
            document_id=document_id,
            page_number=page.number,
            text=redact(text, findings),
            image_path=page.image_path,
            diagram=diagram,
            quarantined_spans=sorted({f.span for f in findings if f.span}),
        )

    async def _transcribe(
        self, page_text: str, image_path: Path, source: str, context: CallContext
    ) -> DiagramTranscription:
        request = ModelRequest(
            model=self._settings.models.vision,
            purpose=f"diagram_transcription:{source}",
            instructions=self._prompts.get(PromptName.DIAGRAM_TRANSCRIPTION).text,
            input=[
                {
                    "role": "user",
                    "content": [
                        input_text(
                            "Native text layer of this page (may omit text inside images):\n"
                            f"<page_text>\n{page_text}\n</page_text>"
                        ),
                        input_image(image_path, self._settings.retrieval.image_detail),
                    ],
                }
            ],
            text_format=DiagramTranscription,
        )
        response = await self._llm.call(request, context)
        if response.parsed is None:
            raise PolicyIngestError(f"{source}: diagram transcription returned no output")
        return response.parsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
