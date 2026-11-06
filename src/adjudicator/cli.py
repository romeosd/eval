"""Command-line entry point: ``adjudicator <command>`` (see ``--help``)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import get_args

from pydantic import TypeAdapter, ValidationError

from adjudicator.config import (
    DEFAULT_CONFIG_PATH,
    ModelSpec,
    ReasoningEffort,
    ReplayMode,
    RetrievalMode,
    Settings,
)
from adjudicator.demo.sample_data import generate
from adjudicator.evaluation.compare import Configuration, compare
from adjudicator.evaluation.consistency import analyse
from adjudicator.evaluation.consistency import render as render_consistency
from adjudicator.evaluation.judge_eval import run_judge_eval
from adjudicator.evaluation.label_audit import audit_labels
from adjudicator.evaluation.metrics import render_markdown, score_run, write_report
from adjudicator.evaluation.variants import generate_variants
from adjudicator.explain import explain_case
from adjudicator.llm.client import BudgetTracker, LLMClient
from adjudicator.llm.replay import build_transport
from adjudicator.observability import AuditLog, configure_logging
from adjudicator.prompts import PromptLibrary
from adjudicator.runner import RunIntegrityError, build_policy_index, execute_run
from adjudicator.schemas import RunManifest

logger = logging.getLogger("adjudicator.cli")
_RETRIEVAL_MODES: TypeAdapter[list[RetrievalMode]] = TypeAdapter(list[RetrievalMode])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adjudicator", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--replay",
        choices=get_args(ReplayMode),
        default=None,
        help="Override [replay] mode: record model calls, or replay them offline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("make-sample-data", help="Generate the sample policy, cases and labels.")
    sub.add_parser("ingest", help="Build (or reuse) the processed policy index and summarise it.")

    run = sub.add_parser("run", help="Adjudicate every case in a split.")
    run.add_argument(
        "--split", default=None, help="Split to run (taken from the run when resuming)"
    )
    run.add_argument(
        "--resume", type=Path, default=None, help="Continue an interrupted run directory"
    )
    run.add_argument("--tag", default=None)
    run.add_argument(
        "--cases", default=None, help="Comma list of case ids (cheap iteration; marked partial)"
    )
    run.add_argument("--no-eval", action="store_true", help="Skip scoring after the run.")

    evaluate = sub.add_parser("eval", help="Score a finished run against its labels.")
    evaluate.add_argument("run_dir", type=Path)

    audit = sub.add_parser(
        "audit-labels", help="Check labels, splits and leakage (no model calls)."
    )
    audit.add_argument("--quiet", action="store_true")

    consistency = sub.add_parser("consistency", help="Run-to-run agreement across repeated runs.")
    consistency.add_argument("run_dirs", type=Path, nargs="+")

    judge = sub.add_parser("judge-eval", help="Measure the claim-support judge on a finished run.")
    judge.add_argument("run_dir", type=Path)
    judge.add_argument("--max-findings", type=int, default=30)

    show = sub.add_parser("show", help="Explain one case from a run (for demos and debugging).")
    show.add_argument("run_dir", type=Path)
    show.add_argument("case_id")

    variants = sub.add_parser("make-variants", help="Generate labelled unseen-data variants.")
    variants.add_argument("--source", default="dev")
    variants.add_argument("--target", default="variants")

    comparison = sub.add_parser("compare", help="Compare models x retrieval modes with repeats.")
    comparison.add_argument("--split", required=True)
    comparison.add_argument(
        "--models",
        required=True,
        help="Comma list of name[:effort], e.g. gpt-5.5:medium,gpt-5.4-mini:low",
    )
    comparison.add_argument("--retrieval", default="tools", help="Comma list: tools,full_context")
    comparison.add_argument("--repeats", type=int, default=1)
    return parser


def _manifest_split(resume: Path | None) -> str:
    if resume is None:
        raise SystemExit("Give --split, or --resume <run directory>")
    return RunManifest.model_validate_json((resume / "manifest.json").read_text("utf-8")).split


def _model_spec(token: str) -> ModelSpec:
    name, _, effort = token.partition(":")
    if effort and effort not in get_args(ReasoningEffort):
        raise SystemExit(f"Unknown reasoning effort {effort!r} in {token!r}")
    return ModelSpec.model_validate({"name": name, "reasoning_effort": effort or None})


def _retrieval_modes(value: str) -> list[RetrievalMode]:
    modes = [m.strip() for m in value.split(",") if m.strip()]
    try:
        return _RETRIEVAL_MODES.validate_python(modes)
    except ValidationError as exc:
        raise SystemExit(f"Unknown retrieval mode in {value!r}: {exc}") from exc


async def _ingest(settings: Settings) -> None:
    prompts = PromptLibrary.load(settings.paths.prompts_dir)
    llm = LLMClient(build_transport(settings), settings, BudgetTracker(settings.run.budget_usd))
    audit = AuditLog(settings.paths.cache_dir / "ingest_audit.jsonl", scope="ingest")
    index, context = await build_policy_index(settings, llm, prompts, audit)
    for document in index.documents:
        pages, sections = len(document.pages), len(document.sections)
        print(f"{document.document_id}: {pages} pages, {sections} sections")
        for section in document.sections:
            span = f"p{section.page_start}-{section.page_end}"
            print(f"  {section.section_id:<28} {span}  {section.title}")
        for page in document.pages:
            if page.diagram and page.diagram.has_visual_content:
                print(f"  diagram on p{page.page_number}: {page.diagram.title}")
                for rule in page.diagram.rules_expressed:
                    print(f"    - {rule}")
    print(f"cache key {index.cache_key}; ingest cost ${sum(u.cost_usd for u in context.usage):.4f}")


type Handler = Callable[[Settings, argparse.Namespace], int]


def _cmd_make_sample_data(settings: Settings, _args: argparse.Namespace) -> int:
    policy, count = generate(settings)
    print(f"Wrote {policy} and {count} labelled cases")
    return 0


def _cmd_ingest(settings: Settings, _args: argparse.Namespace) -> int:
    asyncio.run(_ingest(settings))
    return 0


def _cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    case_filter = args.cases.split(",") if args.cases else None
    split = args.split or _manifest_split(args.resume)
    run_dir = asyncio.run(
        execute_run(settings, split, tag=args.tag, case_filter=case_filter, resume=args.resume)
    )
    print(f"Run directory: {run_dir}")
    if not args.no_eval:
        write_report(run_dir, settings.paths.labels_dir)
        print((run_dir / "eval_report.md").read_text("utf-8"))
    return 0


def _cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    summary, scores = score_run(args.run_dir, settings.paths.labels_dir)
    write_report(args.run_dir, settings.paths.labels_dir)
    print(render_markdown(summary, scores))
    return 0


def _cmd_audit_labels(settings: Settings, _args: argparse.Namespace) -> int:
    report = audit_labels(settings)
    print("\n".join(report.notes))
    for error in report.errors:
        print(f"ERROR {error}")
    return 2 if report.errors else 0


def _cmd_consistency(settings: Settings, args: argparse.Namespace) -> int:
    print(render_consistency(analyse(args.run_dirs, settings.paths.labels_dir)))
    return 0


def _cmd_judge_eval(settings: Settings, args: argparse.Namespace) -> int:
    asyncio.run(run_judge_eval(settings, args.run_dir, max_findings=args.max_findings))
    print((args.run_dir / "judge_eval.md").read_text("utf-8"))
    return 0


def _cmd_show(_settings: Settings, args: argparse.Namespace) -> int:
    print(explain_case(args.run_dir, args.case_id))
    return 0


def _cmd_make_variants(settings: Settings, args: argparse.Namespace) -> int:
    count = generate_variants(settings, args.source, args.target)
    print(f"Wrote {count} variants to split {args.target!r}")
    return 0


def _cmd_compare(settings: Settings, args: argparse.Namespace) -> int:
    configurations = [
        Configuration(model=_model_spec(token), retrieval=mode)
        for token in args.models.split(",")
        for mode in _retrieval_modes(args.retrieval)
    ]
    out = asyncio.run(compare(settings, args.split, configurations, args.repeats))
    print(out.read_text("utf-8"))
    return 0


HANDLERS: dict[str, Handler] = {
    "make-sample-data": _cmd_make_sample_data,
    "ingest": _cmd_ingest,
    "run": _cmd_run,
    "eval": _cmd_eval,
    "audit-labels": _cmd_audit_labels,
    "consistency": _cmd_consistency,
    "judge-eval": _cmd_judge_eval,
    "show": _cmd_show,
    "make-variants": _cmd_make_variants,
    "compare": _cmd_compare,
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(args.log_level)
    settings = Settings.load(args.config)
    if args.replay:
        settings = settings.with_replay_mode(args.replay)
    try:
        return HANDLERS[args.command](settings, args)
    except RunIntegrityError as exc:
        logger.error("Integrity check failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
