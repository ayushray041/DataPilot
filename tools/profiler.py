import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a summary of a pandas DataFrame without changing it.

    The summary includes the DataFrame's shape, column names and data types,
    missing and distinct value counts for each column, and the number of
    duplicate rows. Empty DataFrames are supported.

    Args:
        df: The pandas DataFrame to profile.

    Returns:
        A dictionary containing basic information about the dataset.

    Raises:
        TypeError: If ``df`` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    # Convert labels to strings so the returned names and dictionary keys
    # consistently match the documented output format.
    column_names = [str(column) for column in df.columns]
    data_types = {}
    missing_values = {}
    unique_values = {}

    # Use positional access so columns are profiled correctly even if labels
    # are duplicated.
    for position, column_name in enumerate(column_names):
        column = df.iloc[:, position]
        data_types[column_name] = str(column.dtype)
        missing_values[column_name] = int(column.isna().sum())
        unique_values[column_name] = int(column.nunique(dropna=True))

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": column_names,
        "data_types": data_types,
        "missing_values": missing_values,
        "unique_values": unique_values,
        "duplicate_rows": int(df.duplicated().sum()),
    }