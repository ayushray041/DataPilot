from pathlib import Path

import matplotlib

# Select a headless backend before importing pyplot for tests and servers.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


def create_bar_chart(
    df: pd.DataFrame,
    category_column,
    value_column,
    output_path="outputs/bar_chart.png",
    aggregation="mean",
) -> str:
    """Create and save a bar chart from two DataFrame columns.

    ``aggregation`` may be ``mean``, ``sum``, ``count``, ``min``, ``max``, or
    ``none``. The ``none`` option plots each valid row separately. Missing
    categories and values are excluded.

    Args:
        df: DataFrame containing the chart data.
        category_column: Column used for the chart categories.
        value_column: Column used for values or counts.
        output_path: Destination image path.
        aggregation: How to combine values within each category.

    Returns:
        The saved chart path as a string.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If a required column is invalid, the aggregation is
            unsupported, the values are not numeric where required, or no
            valid rows remain.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    for column in (category_column, value_column):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} does not exist in the DataFrame")
        if not isinstance(df[column], pd.Series):
            raise ValueError(f"Column {column!r} must identify exactly one column")

    allowed_aggregations = ("mean", "sum", "count", "min", "max", "none")
    if aggregation not in allowed_aggregations:
        raise ValueError(
            "aggregation must be one of: mean, sum, count, min, max, or none"
        )

    values = df[value_column]
    if aggregation != "count" and (
        not is_numeric_dtype(values.dtype)
        or is_bool_dtype(values.dtype)
        or is_complex_dtype(values.dtype)
    ):
        raise ValueError(
            f"Column {value_column!r} must contain numeric values for "
            f"{aggregation!r} aggregation"
        )

    # Work from a filtered copy so chart creation never changes the input.
    valid_rows = df[[category_column, value_column]].dropna().copy()
    if valid_rows.empty:
        raise ValueError("No valid data remains for the bar chart")

    figure, axes = plt.subplots()
    try:
        if aggregation == "none":
            positions = range(len(valid_rows))
            axes.bar(positions, valid_rows[value_column])
            axes.set_xticks(list(positions))
            axes.set_xticklabels(valid_rows[category_column].astype(str))
        else:
            if aggregation != "count" and (
                not is_numeric_dtype(valid_rows[value_column].dtype)
                or is_bool_dtype(valid_rows[value_column].dtype)
                or is_complex_dtype(valid_rows[value_column].dtype)
            ):
                raise ValueError(
                    f"Column {value_column!r} must contain numeric values "
                    f"for {aggregation!r} aggregation"
                )
            grouped = valid_rows.groupby(
                category_column, sort=False, observed=True
            )[value_column]
            if aggregation == "count":
                chart_data = grouped.count()
            elif aggregation == "mean":
                chart_data = grouped.mean()
            elif aggregation == "sum":
                chart_data = grouped.sum()
            elif aggregation == "min":
                chart_data = grouped.min()
            else:
                chart_data = grouped.max()
            axes.bar(chart_data.index.astype(str), chart_data.values)

        axes.set_xlabel(str(category_column))
        axes.set_ylabel(
            "Count" if aggregation == "count" else str(value_column)
        )
        axes.set_title(f"{value_column} by {category_column}")
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path)
    finally:
        plt.close(figure)

    return str(saved_path)


def create_line_chart(
    df: pd.DataFrame,
    x_column,
    y_column,
    output_path="outputs/line_chart.png",
) -> str:
    """Create and save a line chart from an x column and numeric y column.

    Rows missing either selected value are excluded. The remaining rows stay
    in their original order.

    Args:
        df: DataFrame containing the chart data.
        x_column: Column providing x-axis values.
        y_column: Numeric column providing y-axis values.
        output_path: Destination image path.

    Returns:
        The saved chart path as a string.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If a required column is invalid, the y column is not
            numeric, or no valid rows remain.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    for column in (x_column, y_column):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} does not exist in the DataFrame")
        if not isinstance(df[column], pd.Series):
            raise ValueError(f"Column {column!r} must identify exactly one column")
    y_values = df[y_column]
    if (
        not is_numeric_dtype(y_values.dtype)
        or is_bool_dtype(y_values.dtype)
        or is_complex_dtype(y_values.dtype)
    ):
        raise ValueError(f"Column {y_column!r} must contain numeric values")

    valid_rows = df[[x_column, y_column]].dropna()
    if valid_rows.empty:
        raise ValueError("No valid data remains for the line chart")

    figure, axes = plt.subplots()
    try:
        axes.plot(valid_rows[x_column], valid_rows[y_column], marker="o")
        axes.set_xlabel(str(x_column))
        axes.set_ylabel(str(y_column))
        axes.set_title(f"{y_column} by {x_column}")
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path)
    finally:
        plt.close(figure)

    return str(saved_path)


def create_scatter_plot(
    df: pd.DataFrame,
    x_column,
    y_column,
    output_path="outputs/scatter_plot.png",
) -> str:
    """Create and save a scatter plot using two numeric columns.

    Rows missing either selected value are excluded.

    Args:
        df: DataFrame containing the chart data.
        x_column: Numeric column providing x-axis values.
        y_column: Numeric column providing y-axis values.
        output_path: Destination image path.

    Returns:
        The saved chart path as a string.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If a required column is invalid, either column is not
            numeric, or no valid rows remain.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    for column in (x_column, y_column):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} does not exist in the DataFrame")
        if not isinstance(df[column], pd.Series):
            raise ValueError(f"Column {column!r} must identify exactly one column")
        values = df[column]
        if (
            not is_numeric_dtype(values.dtype)
            or is_bool_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
        ):
            raise ValueError(f"Column {column!r} must contain numeric values")

    valid_rows = df[[x_column, y_column]].dropna()
    if valid_rows.empty:
        raise ValueError("No valid data remains for the scatter plot")

    figure, axes = plt.subplots()
    try:
        axes.scatter(valid_rows[x_column], valid_rows[y_column])
        axes.set_xlabel(str(x_column))
        axes.set_ylabel(str(y_column))
        axes.set_title(f"{y_column} vs. {x_column}")
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path)
    finally:
        plt.close(figure)

    return str(saved_path)


def create_histogram(
    df: pd.DataFrame,
    column,
    output_path="outputs/histogram.png",
) -> str:
    """Create and save a histogram for one numeric DataFrame column.

    Missing values are excluded.

    Args:
        df: DataFrame containing the chart data.
        column: Numeric column to display.
        output_path: Destination image path.

    Returns:
        The saved chart path as a string.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If the column is invalid, is not numeric, or has no valid
            values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column {column!r} does not exist in the DataFrame")
    values = df[column]
    if not isinstance(values, pd.Series):
        raise ValueError(f"Column {column!r} must identify exactly one column")
    if (
        not is_numeric_dtype(values.dtype)
        or is_bool_dtype(values.dtype)
        or is_complex_dtype(values.dtype)
    ):
        raise ValueError(f"Column {column!r} must contain numeric values")

    valid_values = values.dropna()
    if valid_values.empty:
        raise ValueError("No valid data remains for the histogram")

    figure, axes = plt.subplots()
    try:
        axes.hist(valid_values)
        axes.set_xlabel(str(column))
        axes.set_ylabel("Frequency")
        axes.set_title(f"Distribution of {column}")
        saved_path = Path(output_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(saved_path)
    finally:
        plt.close(figure)

    return str(saved_path)