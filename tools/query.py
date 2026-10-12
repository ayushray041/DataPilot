import numbers

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def filter_data(df: pd.DataFrame, column, operator, value) -> pd.DataFrame:
    """Filter rows using a comparison on one DataFrame column.

    Numeric columns accept numeric-looking string values, such as ``"42"``,
    which are converted before comparison. Values for text columns are kept
    as text.

    Args:
        df: DataFrame to filter.
        column: Name of the column to compare.
        operator: One of ``==``, ``!=``, ``>``, ``<``, ``>=``, or ``<=``.
        value: Value to compare against.

    Returns:
        A filtered DataFrame; the input DataFrame is not modified.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
        ValueError: If the column or operator is invalid, or a value cannot
            be compared with the selected column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column {column!r} does not exist in the DataFrame")
    if operator not in ("==", "!=", ">", "<", ">=", "<="):
        raise ValueError(
            f"Unsupported operator {operator!r}; use ==, !=, >, <, >=, or <="
        )

    selected = df[column]
    if not isinstance(selected, pd.Series):
        raise ValueError(f"Column {column!r} must identify exactly one column")

    # Only convert string operands for numeric columns; text values like CSE
    # should be compared exactly as supplied.
    if (
        isinstance(value, str)
        and is_numeric_dtype(selected.dtype)
        and not is_bool_dtype(selected.dtype)
    ):
        try:
            value = pd.to_numeric(value, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Value {value!r} must be numeric for column {column!r}"
            ) from error

    try:
        if operator == "==":
            matches = selected == value
        elif operator == "!=":
            matches = selected != value
        elif operator == ">":
            matches = selected > value
        elif operator == "<":
            matches = selected < value
        elif operator == ">=":
            matches = selected >= value
        else:
            matches = selected <= value
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Value {value!r} cannot be compared with column {column!r}"
        ) from error

    return df.loc[matches].copy()


def sort_data(
    df: pd.DataFrame,
    column,
    descending: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """Sort a DataFrame by one column and optionally limit the returned rows.

    Missing values are always placed last, regardless of sort direction.

    Args:
        df: DataFrame to sort.
        column: Name of the column to sort by.
        descending: If true, sort from highest to lowest.
        limit: Optional positive integer number of rows to return.

    Returns:
        A sorted DataFrame; the input DataFrame is not modified.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame, ``descending`` is not
            a boolean, or ``limit`` is not an integer.
        ValueError: If the column does not exist or ``limit`` is not positive.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if column not in df.columns:
        raise ValueError(f"Column {column!r} does not exist in the DataFrame")
    if not isinstance(descending, bool):
        raise TypeError("descending must be a boolean")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, numbers.Integral):
            raise TypeError("limit must be a positive integer")
        if limit <= 0:
            raise ValueError("limit must be a positive integer")

    if not isinstance(df[column], pd.Series):
        raise ValueError(f"Column {column!r} must identify exactly one column")

    # Pandas returns a new sorted DataFrame; na_position keeps nulls at the end.
    sorted_df = df.sort_values(
        by=column,
        ascending=not descending,
        na_position="last",
    )
    if limit is not None:
        sorted_df = sorted_df.head(int(limit))
    return sorted_df