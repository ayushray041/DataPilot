from pathlib import Path

import pandas as pd
import pytest

from tools.visualization import (
    create_bar_chart,
    create_histogram,
    create_line_chart,
    create_scatter_plot,
)


@pytest.fixture
def chart_data():
    return pd.DataFrame(
        {
            "Branch": ["CSE", "ECE", "CSE", "ME"],
            "Salary": [60000.0, 40000.0, 80000.0, 70000.0],
            "CGPA": [8.4, 7.2, 9.1, 8.0],
        }
    )


def assert_saved_png(path):
    image_path = Path(path)

    assert image_path.exists()
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
    assert image_path.suffix.lower() == ".png"
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("aggregation", ["mean", "sum", "count", "min", "max", "none"])
def test_create_bar_chart_saves_png(chart_data, tmp_path, aggregation):
    output_path = tmp_path / aggregation / "bar_chart.png"

    saved_path = create_bar_chart(
        chart_data,
        "Branch",
        "Salary",
        output_path=output_path,
        aggregation=aggregation,
    )

    assert saved_path == str(output_path)
    assert_saved_png(saved_path)


@pytest.mark.parametrize(
    ("chart_function", "columns", "filename"),
    [
        (create_line_chart, ("CGPA", "Salary"), "line_chart.png"),
        (create_scatter_plot, ("CGPA", "Salary"), "scatter_plot.png"),
        (create_histogram, ("Salary",), "histogram.png"),
    ],
)
def test_other_charts_save_png(chart_data, tmp_path, chart_function, columns, filename):
    output_path = tmp_path / "charts" / filename

    saved_path = chart_function(chart_data, *columns, output_path=output_path)

    assert saved_path == str(output_path)
    assert_saved_png(saved_path)


@pytest.mark.parametrize(
    ("chart_function", "columns"),
    [
        (create_bar_chart, ("Missing", "Salary")),
        (create_bar_chart, ("Branch", "Missing")),
        (create_line_chart, ("Missing", "Salary")),
        (create_scatter_plot, ("CGPA", "Missing")),
        (create_histogram, ("Missing",)),
    ],
)
def test_charts_reject_invalid_columns(
    chart_data, tmp_path, chart_function, columns
):
    with pytest.raises(ValueError, match="does not exist"):
        chart_function(
            chart_data,
            *columns,
            output_path=tmp_path / "invalid_column.png",
        )


@pytest.mark.parametrize(
    ("chart_function", "columns"),
    [
        (create_bar_chart, ("Branch", "Branch")),
        (create_line_chart, ("CGPA", "Branch")),
        (create_scatter_plot, ("Branch", "Salary")),
        (create_histogram, ("Branch",)),
    ],
)
def test_charts_reject_non_numeric_values(
    chart_data, tmp_path, chart_function, columns
):
    with pytest.raises(ValueError, match="numeric"):
        chart_function(
            chart_data,
            *columns,
            output_path=tmp_path / "non_numeric.png",
        )


@pytest.mark.parametrize(
    ("chart_function", "columns"),
    [
        (create_bar_chart, ("Branch", "Salary")),
        (create_line_chart, ("CGPA", "Salary")),
        (create_scatter_plot, ("CGPA", "Salary")),
        (create_histogram, ("Salary",)),
    ],
)
def test_charts_reject_empty_valid_data(
    tmp_path, chart_function, columns
):
    empty_data = pd.DataFrame(
        {
            "Branch": [None, None],
            "CGPA": pd.Series([float("nan"), float("nan")], dtype="float64"),
            "Salary": pd.Series([float("nan"), float("nan")], dtype="float64"),
        }
    )

    with pytest.raises(ValueError, match="No valid data"):
        chart_function(
            empty_data,
            *columns,
            output_path=tmp_path / "empty_data.png",
        )


def test_bar_chart_rejects_invalid_aggregation(chart_data, tmp_path):
    with pytest.raises(ValueError, match="aggregation must be one of"):
        create_bar_chart(
            chart_data,
            "Branch",
            "Salary",
            output_path=tmp_path / "invalid_aggregation.png",
            aggregation="median",
        )