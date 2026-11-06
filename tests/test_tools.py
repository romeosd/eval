import json

from adjudicator.llm.client import FunctionCall
from adjudicator.tools import CalculateArgs, strict_schema

from .conftest import build_services
from .helpers import load_case, toolbox_for


def _call(toolbox, name, **arguments):
    return toolbox.execute(FunctionCall(call_id="x", name=name, arguments=json.dumps(arguments)))


def test_every_tool_schema_is_strict(services, context):
    toolbox = toolbox_for(services, load_case(services, context, "dev-001"), context)
    for definition in toolbox.definitions():
        params = definition["parameters"]
        assert definition["strict"] is True
        assert params["additionalProperties"] is False
        assert set(params["required"]) == set(params["properties"])


def test_invalid_arguments_return_error_payload(services, context):
    toolbox = toolbox_for(services, load_case(services, context, "dev-001"), context)
    result = toolbox.execute(
        FunctionCall(call_id="x", name="calculate", arguments='{"operation": "pow"}')
    )
    assert result.is_error and "invalid arguments" in str(result.payload["error"])
    assert _call(toolbox, "no_such_tool").is_error


def test_quarantined_image_is_withheld(services, context):
    toolbox = toolbox_for(services, load_case(services, context, "dev-006"), context)
    withheld = _call(toolbox, "view_application_image", document_id="doctor_letter")
    assert withheld.is_error and withheld.image_path is None
    clean = _call(toolbox, "view_application_image", document_id="invoice_1")
    assert clean.image_path is not None
    output = clean.as_output("high")
    assert isinstance(output, list) and output[1]["type"] == "input_image"


def test_policy_page_returns_diagram_and_image(services, context):
    toolbox = toolbox_for(services, load_case(services, context, "dev-003"), context)
    page = _call(toolbox, "view_policy_page", document_id="travel-policy", page=4)
    assert page.image_path is not None and page.payload["diagram"] is not None


def test_full_context_mode_disables_search_tools(settings, transport, context):
    services = build_services(settings.with_retrieval_mode("full_context"), transport)
    names = {
        d["name"]
        for d in toolbox_for(
            services, load_case(services, context, "dev-001"), context
        ).definitions()
    }
    assert "search_policy" not in names and "view_policy_page" in names


def test_list_policy_sections_flags_diagram_sections(services, context):
    toolbox = toolbox_for(services, load_case(services, context, "dev-001"), context)
    result = _call(toolbox, "list_policy_sections")
    sections = {s["section_id"]: s for s in result.payload["sections"]}
    assert sections["travel-policy:5.3"]["has_diagram"] is True
    assert sections["travel-policy:3.1"]["has_diagram"] is False
    assert len(sections) == len(services.index.all_sections())


def test_full_context_mode_has_no_section_listing(settings, transport, context):
    services = build_services(settings.with_retrieval_mode("full_context"), transport)
    toolbox = toolbox_for(services, load_case(services, context, "dev-001"), context)
    assert "list_policy_sections" not in {d["name"] for d in toolbox.definitions()}


def test_tool_schemas_have_no_ref_with_sibling_keywords():
    """OpenAI strict mode rejects {"$ref": ..., "description": ...} (400 Bad Request)."""

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                assert set(node) == {"$ref"}, node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    schema = strict_schema(CalculateArgs)
    walk(schema)
    unit = schema["$defs"]["NumberInput"]["properties"]["unit"]
    assert "usd" in unit["enum"] and unit["description"]
