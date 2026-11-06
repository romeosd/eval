"""Compare models and retrieval strategies on the same split, with repeats.

Each configuration x repeat is a normal, immutable run. The report never ranks two
configurations on accuracy alone: it adds an exact paired test on the cases where they
disagree and the run-to-run agreement of each configuration, because with small samples
most differences are noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev

from adjudicator.config import ModelSpec, RetrievalMode, Settings
from adjudicator.evaluation.consistency import analyse
from adjudicator.evaluation.metrics import score_run, write_report
from adjudicator.evaluation.results import CaseScore, EvalSummary
from adjudicator.evaluation.stats import paired
from adjudicator.llm.client import ResponsesTransport
from adjudicator.runner import execute_run


@dataclass(frozen=True)
class Configuration:
    model: ModelSpec
    retrieval: RetrievalMode

    @property
    def label(self) -> str:
        effort = self.model.reasoning_effort or "default"
        return f"{self.model.name}@{effort}/{self.retrieval}"

    @property
    def tag(self) -> str:
        return self.label.replace("/", "-").replace("@", "-").replace(".", "")


type Scored = list[tuple[EvalSummary, list[CaseScore]]]


async def compare(
    settings: Settings,
    split: str,
    configurations: list[Configuration],
    repeats: int,
    transport: ResponsesTransport | None = None,
) -> Path:
    for configuration in configurations:
        if configuration.model.name not in settings.pricing:
            raise ValueError(f"Add a [pricing] entry for {configuration.model.name!r} first")
    rows: list[tuple[Configuration, list[Path]]] = []
    for configuration in configurations:
        variant = settings.with_adjudicator(configuration.model).with_retrieval_mode(
            configuration.retrieval
        )
        run_dirs: list[Path] = []
        for _ in range(repeats):
            run_dir = await execute_run(variant, split, transport=transport, tag=configuration.tag)
            write_report(run_dir, settings.paths.labels_dir)
            run_dirs.append(run_dir)
        rows.append((configuration, run_dirs))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    out = settings.paths.runs_dir / f"compare-{split}-{stamp}.md"
    out.write_text(render_comparison(split, repeats, rows, settings.paths.labels_dir), "utf-8")
    return out


def _majority_correct(scored: Scored) -> dict[str, bool]:
    """A case counts as correct for a configuration if most of its repeats got it right."""
    cases = [s.case_id for s in scored[0][1]]
    return {
        c: sum(next(s for s in scores if s.case_id == c).outcome_correct for _, scores in scored)
        > len(scored) / 2
        for c in cases
    }


def render_comparison(
    split: str,
    repeats: int,
    rows: list[tuple[Configuration, list[Path]]],
    labels_dir: Path,
) -> str:
    scored = [(c, [score_run(d, labels_dir) for d in dirs]) for c, dirs in rows]

    def stat(values: list[float]) -> str:
        return (
            f"{mean(values):.0%} +/- {pstdev(values):.0%}"
            if len(values) > 1
            else f"{values[0]:.0%}"
        )

    lines = [
        f"# Comparison on `{split}` ({repeats} repeat(s) per configuration)",
        "",
        "| Configuration | Accuracy | Unsafe approvals | Forced referrals | Section recall | "
        "Cost / case | Latency / case | Runs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (configuration, dirs), (_, runs) in zip(rows, scored, strict=True):
        summaries = [s for s, _ in runs]
        recalls = [s.mean_section_recall for s in summaries if s.mean_section_recall is not None]
        lines.append(
            f"| {configuration.label} "
            f"| {stat([s.decision_accuracy for s in summaries])} "
            f"| {stat([s.unsafe_approval_rate for s in summaries])} "
            f"| {stat([s.forced_referral_rate for s in summaries])} "
            f"| {stat(recalls) if recalls else 'n/a'} "
            f"| ${mean(s.mean_cost_usd for s in summaries):.4f} "
            f"| {mean(s.mean_duration_s for s in summaries):.1f}s "
            f"| {', '.join(d.name for d in dirs)} |"
        )
    n = scored[0][1][0][0].n_cases
    lines += [
        "",
        f"> n = {n} cases per run. Repeats of one configuration are not independent samples, so "
        "the +/- shows run-to-run variance only. Do not rank configurations on these numbers "
        "alone; use the paired test below.",
        "",
    ]
    if repeats > 1:
        lines += ["## Agreement across repeats", ""]
        lines += ["| Configuration | Mean agreement | Unstable cases | Right-and-wrong cases |"]
        lines += ["|---|---|---|---|"]
        for (configuration, dirs), _ in zip(rows, scored, strict=True):
            c = analyse(dirs, labels_dir)
            lines.append(
                f"| {configuration.label} | {c.mean_agreement:.0%} | "
                f"{', '.join(c.unstable_cases) or 'none'} | "
                f"{', '.join(c.correctness_flips) or 'none'} |"
            )
        lines.append("")
    if len(rows) > 1:
        base_config, base_runs = scored[0]
        base = _majority_correct(base_runs)
        lines += [
            f"## Paired comparison against `{base_config.label}`",
            "",
            "| Configuration | Both right | Only baseline | Only this | Neither | Exact p |",
            "|---|---|---|---|---|---|",
        ]
        for configuration, runs in scored[1:]:
            result = paired(base, _majority_correct(runs))
            lines.append(
                f"| {configuration.label} | {result.both} | {result.only_a} | {result.only_b} | "
                f"{result.neither} | {result.p_value:.2f} |"
            )
        lines += ["", "A p-value above 0.05 means the difference could easily be chance.", ""]
    return "\n".join(lines) + "\n"
