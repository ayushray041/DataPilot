"""Descriptions and callable functions for DataPilot's dataset tools."""

import pandas as pd

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


TOOL_REGISTRY = {
    "profiler": {
        "function": profile_dataset,
        "description": (
            "Summarize the dataset's shape, columns, data types, missing "
            "values, distinct values, and duplicate rows."
        ),
        "arguments": {},
        "required": [],
    },
    "filter": {
        "function": filter_data,
        "description": (
            "Filter rows by comparing a column with a value. Values can be "
            "categorical text, such as CSE, or numeric."
        ),
        "arguments": {
            "column": "Name of the column to filter.",
            "operator": "Comparison operator: ==, !=, >, <, >=, or <=.",
            "value": "Text or numeric value to compare against.",
        },
        "required": ["column", "operator", "value"],
    },
    "sort": {
        "function": sort_data,
        "description": (
            "Sort rows by a column, placing missing values last, and "
            "optionally return only the first requested number of rows."
        ),
        "arguments": {
            "column": "Name of the column to sort by.",
            "descending": "Whether to sort from highest to lowest; defaults to true.",
            "limit": "Optional positive maximum number of rows to return.",
        },
        "required": ["column"],
    },
    "statistics": {
        "function": calculate_statistics,
        "description": (
            "Calculate count, mean, median, minimum, maximum, standard "
            "deviation, and quartiles for one numeric column."
        ),
        "arguments": {
            "column": "Name of the numeric column to summarize.",
        },
        "required": ["column"],
    },
    "correlation": {
        "function": calculate_correlation,
        "description": (
            "Calculate Pearson correlation for two numeric columns and "
            "report the number of complete row pairs used."
        ),
        "arguments": {
            "column1": "Name of the first numeric column.",
            "column2": "Name of the second numeric column.",
        },
        "required": ["column1", "column2"],
    },
    "bar_chart": {
        "function": create_bar_chart,
        "description": (
            "Create a bar chart of a value column grouped by a category "
            "column, using mean aggregation by default."
        ),
        "arguments": {
            "category_column": "Name of the column that defines bar categories.",
            "value_column": "Name of the column containing values to plot.",
            "aggregation": (
                "Optional aggregation: mean, sum, count, min, max, or none; "
                "defaults to mean."
            ),
        },
        "required": ["category_column", "value_column"],
    },
    "line_chart": {
        "function": create_line_chart,
        "description": "Create a line chart using an x column and a numeric y column.",
        "arguments": {
            "x_column": "Name of the column for the horizontal axis.",
            "y_column": "Name of the numeric column for the vertical axis.",
        },
        "required": ["x_column", "y_column"],
    },
    "scatter_plot": {
        "function": create_scatter_plot,
        "description": "Create a scatter plot comparing two numeric columns.",
        "arguments": {
            "x_column": "Name of the numeric column for the horizontal axis.",
            "y_column": "Name of the numeric column for the vertical axis.",
        },
        "required": ["x_column", "y_column"],
    },
    "histogram": {
        "function": create_histogram,
        "description": "Create a histogram showing the distribution of one numeric column.",
        "arguments": {
            "column": "Name of the numeric column to display.",
        },
        "required": ["column"],
    },
    "anomaly": {
        "function": detect_anomalies,
        "description": (
            "Flag unusual observations in selected numeric columns. "
            "An unusual observation is not necessarily an error."
        ),
        "arguments": {
            "columns": "Non-empty list of numeric column names to analyze.",
            "contamination": (
                "Optional expected proportion of anomalies, greater than 0 "
                "and at most 0.5; defaults to 0.05."
            ),
        },
        "required": ["columns"],
    },
}


def build_tool_context(df: pd.DataFrame) -> dict:
    """Build dataset and tool information for a DataPilot conversation.

    Column names and data types help a beginner choose valid arguments. The
    tool descriptions and argument guides explain what each available action
    does without sending the DataFrame itself as a model-provided argument.

    Args:
        df: The current dataset.

    Returns:
        A dictionary containing dataset columns, their data types, and tool
        descriptions, arguments, and required argument names.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Match the profiler's user-facing column-name format.
    columns = [str(column) for column in df.columns]
    data_types = {
        str(column): str(df.iloc[:, position].dtype)
        for position, column in enumerate(df.columns)
    }
    available_tools = {
        name: {
            "description": tool["description"],
            "arguments": tool["arguments"],
            "required": tool["required"],
        }
        for name, tool in TOOL_REGISTRY.items()
    }

    return {
        "columns": columns,
        "data_types": data_types,
        "tools": available_tools,
    }