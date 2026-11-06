import json

import pytest

from adjudicator.llm.client import FunctionCall
from adjudicator.provenance import extract_quantities, match_quantity

from .conftest import build_services
from .helpers import cap_calculation, load_case, toolbox_for


def _call(toolbox, name, **arguments):
    return toolbox.execute(FunctionCall(call_id="x", name=name, arguments=json.dumps(arguments)))


def _read(toolbox, section):
    assert not _call(toolbox, "get_policy_section", section_id=f"travel-policy:{section}").is_error


def _calc(toolbox, value, unit, source, ref, *, operation="sum"):
    inputs = [{"value": value, "unit": unit, "source": source, "ref": ref}]
    return _call(toolbox, "calculate", operation=operation, inputs=inputs, datetimes=[])


@pytest.fixture
def toolbox(services, context):
    return toolbox_for(services, load_case(services, context, "dev-002"), context)


# ------------------------------------------------------------------ quantity extraction


def test_extracts_money_durations_words_and_percent():
    text = "Over $5,000 or 2,500 AUD. Within 24 hours; twenty-four hour notice; 60 days; 15%."
    found = {(q.value, q.unit) for q in extract_quantities(text)}
    assert {(5000, "usd"), (2500, "usd"), (24, "hours"), (60, "days"), (15, "percent")} <= found


def test_unit_must_match_not_just_the_number():
    text = "Claims over $5,000 need documents. Theft must be reported within 24 hours."
    assert match_quantity(text, 5000, "usd")
    assert match_quantity(text, 24, "hours")
    assert match_quantity(text, 5000, "hours") is None  # the number exists, the unit does not
    assert match_quantity(text, 24, "usd") is None


# ------------------------------------------------------------------ policy inputs


def test_policy_value_with_matching_unit_is_accepted_and_recorded(toolbox):
    _read(toolbox, "3.1")
    result = _calc(toolbox, 500, "usd", "policy", "travel-policy:3.1")
    assert not result.is_error
    [verified] = toolbox.calculations["calc-1"].inputs
    assert verified.source == "policy" and "$500" in verified.matched_text


def test_hallucinated_threshold_is_rejected_with_a_pointer_to_where_it_exists(toolbox):
    _read(toolbox, "4.3")
    result = _calc(toolbox, 500, "usd", "policy", "travel-policy:4.3")
    assert result.is_error
    assert "travel-policy:3.1" in str(result.payload["error"])
    assert not toolbox.calculations  # nothing unverified ever reaches the ledger


def test_wrong_unit_for_a_real_number_is_rejected(toolbox):
    _read(toolbox, "3.1")
    assert _calc(toolbox, 500, "hours", "policy", "travel-policy:3.1").is_error


def test_policy_section_must_be_read_first(toolbox):
    result = _calc(toolbox, 500, "usd", "policy", "travel-policy:3.1")
    assert result.is_error and "get_policy_section" in str(result.payload["error"])


def test_diagram_numbers_are_available_from_the_section_that_spans_the_page(toolbox):
    _read(toolbox, "5.3")
    assert not _calc(toolbox, 24, "hours", "policy", "travel-policy:5.3").is_error


def test_full_context_mode_does_not_require_explicit_reads(settings, transport, context):
    services = build_services(settings.with_retrieval_mode("full_context"), transport)
    box = toolbox_for(services, load_case(services, context, "dev-002"), context)
    assert not _calc(box, 500, "usd", "policy", "travel-policy:3.1").is_error


def test_unknown_section_is_rejected(toolbox):
    assert _calc(toolbox, 1, "count", "policy", "travel-policy:99.9").is_error


# ------------------------------------------------------------------ evidence and derived


def test_evidence_amount_must_appear_in_the_document(toolbox):
    assert not _calc(toolbox, 1200, "usd", "evidence", "receipt_laptop").is_error
    assert _calc(toolbox, 1300, "usd", "evidence", "receipt_laptop").is_error
    assert _calc(toolbox, 1200, "hours", "evidence", "receipt_laptop").is_error


def test_form_fields_count_as_evidence_with_money_keys_as_dollars(toolbox):
    assert not _calc(toolbox, 1200, "usd", "evidence", "form").is_error


def test_unloaded_or_unknown_document_is_rejected(toolbox):
    assert _calc(toolbox, 1200, "usd", "evidence", "no_such_document").is_error


def test_derived_inputs_must_equal_an_earlier_result_and_unit(toolbox):
    cap_calculation(toolbox)  # calc-1 = 500 usd
    assert not _calc(toolbox, 500, "usd", "derived", "calc-1").is_error
    assert _calc(toolbox, 450, "usd", "derived", "calc-1").is_error
    assert _calc(toolbox, 500, "hours", "derived", "calc-1").is_error
    assert _calc(toolbox, 500, "usd", "derived", "calc-9").is_error
