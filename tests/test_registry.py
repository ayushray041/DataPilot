import inspect

import pandas as pd
import pytest

from agent.tool_registry import TOOL_REGISTRY, build_tool_context
from tools.anomaly import detect_anomalies
from tools.profiler import profile_dataset
from tools.query import filter_data, sort_data
from tools.statistics import calculate_correlation, calculate_statistics
from tools.visualization import (
    create_bar_chart,
    create_histogram,
    create_line_chart,
    create_scatter_plot,
)


EXPECTED_FUNCTIONS = {
    "profiler": profile_dataset,
    "filter": filter_data,
    "sort": sort_data,
    "statistics": calculate_statistics,
    "correlation": calculate_correlation,
    "bar_chart": create_bar_chart,
    "line_chart": create_line_chart,
    "scatter_plot": create_scatter_plot,
    "histogram": create_histogram,
    "anomaly": detect_anomalies,
}


def test_registry_has_exactly_expected_tools():
    assert set(TOOL_REGISTRY) == set(EXPECTED_FUNCTIONS)


@pytest.mark.parametrize("tool_name", EXPECTED_FUNCTIONS)
def test_registry_entries_have_valid_metadata(tool_name):
    entry = TOOL_REGISTRY[tool_name]

    assert entry["function"] is EXPECTED_FUNCTIONS[tool_name]
    assert callable(entry["function"])
    assert isinstance(entry["description"], str)
    assert entry["description"].strip()
    assert isinstance(entry["arguments"], dict)
    assert all(
        isinstance(name, str) and isinstance(description, str) and description.strip()
        for name, description in entry["arguments"].items()
    )
    assert isinstance(entry["required"], list)
    assert all(
        isinstance(name, str) and name in entry["arguments"]
        for name in entry["required"]
    )


@pytest.mark.parametrize(
    ("tool_name", "expected_arguments", "expected_required"),
    [
        ("profiler", set(), set()),
        ("filter", {"column", "operator", "value"}, {"column", "operator", "value"}),
        ("sort", {"column", "descending", "limit"}, {"column"}),
        ("statistics", {"column"}, {"column"}),
        ("correlation", {"column1", "column2"}, {"column1", "column2"}),
        (
            "bar_chart",
            {"category_column", "value_column", "aggregation"},
            {"category_column", "value_column"},
        ),
        ("line_chart", {"x_column", "y_column"}, {"x_column", "y_column"}),
        ("scatter_plot", {"x_column", "y_column"}, {"x_column", "y_column"}),
        ("histogram", {"column"}, {"column"}),
        ("anomaly", {"columns", "contamination"}, {"columns"}),
    ],
)
def test_tool_arguments_match_public_function_signatures(
    tool_name, expected_arguments, expected_required
):
    entry = TOOL_REGISTRY[tool_name]
    signature = inspect.signature(entry["function"])
    expected_parameters = set(signature.parameters) - {"df", "output_path"}

    assert set(entry["arguments"]) == expected_arguments
    assert set(entry["arguments"]) == expected_parameters
    assert set(entry["required"]) == expected_required
    assert "df" not in entry["arguments"]
    assert "output_path" not in entry["arguments"]

    for name, parameter in signature.parameters.items():
        if name in ("df", "output_path"):
            continue
        should_be_optional = parameter.default is not inspect.Parameter.empty
        assert (name not in entry["required"]) is should_be_optional


def test_filter_value_description_allows_text_and_numeric_comparisons():
    description = TOOL_REGISTRY["filter"]["arguments"]["value"].lower()

    assert "text" in description
    assert "numeric" in description


def test_build_tool_context_reports_dataset_columns_and_dtypes():
    df = pd.DataFrame(
        {
            "Name": ["Ada", "Grace"],
            "Score": [91, 87],
            "Passed": [True, False],
        }
    )

    context = build_tool_context(df)

    assert context["columns"] == ["Name", "Score", "Passed"]
    assert context["data_types"] == {
        "Name": "str",
        "Score": "int64",
        "Passed": "bool",
    }
    assert set(context["tools"]) == set(EXPECTED_FUNCTIONS)