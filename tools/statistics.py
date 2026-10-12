import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


def calculate_statistics(df: pd.DataFrame, column) -> dict:
    """Calculate descriptive statistics for one numeric DataFrame column.

    Missing values are ignored. The returned dictionary contains the count,
    mean, median, minimum, maximum, sample standard deviation, and 25th, 50th,
    and 75th percentiles.

    Args:
        df: DataFrame containing the column to summarize.
        column: Name of the numeric column to summarize.

    Returns:
        A dictionary of statistics using Python ``int`` and ``float`` values.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If the column is missing, not numeric, or has no valid
            values after missing data is removed.
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
        raise ValueError(f"Column {column!r} must contain real numeric values")

    values = values.dropna()
    if values.empty:
        raise ValueError(f"Column {column!r} has no valid numeric values")

    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
        "25th_percentile": float(values.quantile(0.25)),
        "50th_percentile": float(values.quantile(0.50)),
        "75th_percentile": float(values.quantile(0.75)),
    }


def calculate_correlation(
    df: pd.DataFrame, column1, column2
) -> dict:
    """Calculate the Pearson correlation between two numeric columns.

    Rows with a missing value in either selected column are ignored.

    Args:
        df: DataFrame containing both columns.
        column1: Name of the first numeric column.
        column2: Name of the second numeric column.

    Returns:
        A dictionary containing the column names, Pearson correlation
        coefficient as a Python ``float``, and number of complete pairs as
        a Python ``int``.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If a column is missing or not numeric, fewer than two
            complete row pairs remain, either column has no variation, or
            the correlation cannot be calculated.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    selected_columns = []
    for column in (column1, column2):
        if column not in df.columns:
            raise ValueError(f"Column {column!r} does not exist in the DataFrame")

        values = df[column]
        if not isinstance(values, pd.Series):
            raise ValueError(
                f"Column {column!r} must identify exactly one column"
            )
        if (
            not is_numeric_dtype(values.dtype)
            or is_bool_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
        ):
            raise ValueError(
                f"Column {column!r} must contain real numeric values"
            )
        selected_columns.append(values)

    # Drop whole rows so both values in each pair come from the same observation.
    complete_rows = pd.concat(selected_columns, axis=1).dropna()
    if len(complete_rows) < 2:
        raise ValueError(
            "At least two complete numeric row pairs are required to calculate correlation"
        )
    if complete_rows.iloc[:, 0].nunique() < 2:
        raise ValueError(f"Column {column1!r} has no variation")
    if complete_rows.iloc[:, 1].nunique() < 2:
        raise ValueError(f"Column {column2!r} has no variation")
    if complete_rows.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError(
            "Correlation cannot be calculated with infinite column values"
        )

    correlation = float(
        complete_rows.iloc[:, 0].corr(complete_rows.iloc[:, 1])
    )
    if not math.isfinite(correlation):
        raise ValueError(
            "Correlation cannot be calculated from the selected column values"
        )

    return {
        "column1": column1,
        "column2": column2,
        "correlation": correlation,
        "pairs_used": int(len(complete_rows)),
    }