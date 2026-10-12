import pytest
from google.genai import types

from agent.gemini_tools import build_gemini_tools
from agent.tool_registry import TOOL_REGISTRY


EXPECTED_TOOL_NAMES = {
    "profiler",
    "filter",
    "sort",
    "statistics",
    "correlation",
    "bar_chart",
    "line_chart",
    "scatter_plot",
    "histogram",
    "anomaly",
}


@pytest.fixture
def declarations():
    # Building declarations only constructs local SDK objects; it makes no API calls.
    tools = build_gemini_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], types.Tool)
    return {
        declaration.name: declaration
        for declaration in tools[0].function_declarations
    }


def test_declarations_include_all_registry_tools(declarations):
    assert set(declarations) == EXPECTED_TOOL_NAMES
    assert set(declarations) == set(TOOL_REGISTRY)


@pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOL_NAMES))
def test_function_declarations_have_name_description_and_parameters(
    declarations, tool_name
):
    declaration = declarations[tool_name]
    dumped = declaration.model_dump(exclude_none=True)

    assert declaration.name
    assert declaration.description
    assert declaration.parameters_json_schema
    assert dumped["name"] == tool_name
    assert dumped["description"].strip()
    assert dumped["parameters_json_schema"]["type"] == "OBJECT"


def test_filter_requires_column_operator_and_string_value(declarations):
    schema = declarations["filter"].parameters_json_schema

    assert set(schema["required"]) == {"column", "operator", "value"}
    assert schema["properties"]["value"]["type"] == "STRING"


def test_bar_chart_requires_only_category_and_value_columns(declarations):
    schema = declarations["bar_chart"].parameters_json_schema

    assert set(schema["required"]) == {"category_column", "value_column"}
    assert "aggregation" in schema["properties"]
    assert "aggregation" not in schema["required"]


def test_sort_optional_arguments_are_not_required(declarations):
    schema = declarations["sort"].parameters_json_schema

    assert "descending" in schema["properties"]
    assert "limit" in schema["properties"]
    assert "descending" not in schema["required"]
    assert "limit" not in schema["required"]


def test_anomaly_contamination_is_optional(declarations):
    schema = declarations["anomaly"].parameters_json_schema

    assert "columns" in schema["required"]
    assert "contamination" in schema["properties"]
    assert "contamination" not in schema["required"]


@pytest.mark.parametrize(
    "tool_name",
    ["bar_chart", "line_chart", "scatter_plot", "histogram"],
)
def test_chart_output_path_is_not_exposed(declarations, tool_name):
    schema = declarations[tool_name].parameters_json_schema

    assert "output_path" not in schema["properties"]


def test_json_schema_argument_types(declarations):
    assert (
        declarations["filter"].parameters_json_schema["properties"]["value"]["type"]
        == "STRING"
    )
    assert (
        declarations["sort"].parameters_json_schema["properties"]["descending"]["type"]
        == "BOOLEAN"
    )
    assert (
        declarations["sort"].parameters_json_schema["properties"]["limit"]["type"]
        == "INTEGER"
    )
    assert (
        declarations["anomaly"].parameters_json_schema["properties"]["contamination"]["type"]
        == "NUMBER"
    )
    columns_schema = declarations["anomaly"].parameters_json_schema["properties"][
        "columns"
    ]
    assert columns_schema["type"] == "ARRAY"
    assert columns_schema["items"]["type"] == "STRING"
