import asyncio

from adjudicator.config import ModelSpec
from adjudicator.evaluation.compare import Configuration, compare

from .conftest import ScriptedTransport, respond
from .helpers import referral


def _refer(request):
    return respond(referral(request.purpose.split(":")[1]))


def test_comparison_reports_paired_tests_and_run_to_run_agreement(settings):
    configurations = [
        Configuration(ModelSpec(name="gpt-5.5", reasoning_effort="medium"), "tools"),
        Configuration(ModelSpec(name="gpt-5.4-mini", reasoning_effort="low"), "full_context"),
    ]
    transport = ScriptedTransport(adjudicate=_refer)
    out = asyncio.run(compare(settings, "dev", configurations, 2, transport=transport))
    text = out.read_text()

    assert "n = 8 cases per run" in text
    assert "Agreement across repeats" in text
    assert "Paired comparison against `gpt-5.5@medium/tools`" in text
    assert "Exact p" in text and "could easily be chance" in text
    assert text.count("gpt-5.4-mini@low/full_context") >= 3
    runs = [p for p in settings.paths.runs_dir.iterdir() if p.is_dir()]
    assert len(runs) == 4  # 2 configurations x 2 repeats, each an immutable run


def test_single_configuration_single_repeat_has_no_pairwise_sections(settings):
    configurations = [Configuration(ModelSpec(name="gpt-5.5", reasoning_effort="medium"), "tools")]
    text = asyncio.run(
        compare(settings, "dev", configurations, 1, transport=ScriptedTransport(adjudicate=_refer))
    ).read_text()
    assert "Paired comparison" not in text and "Agreement across repeats" not in text
