import math
import numbers

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df: pd.DataFrame,
    columns: list,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Flag unusual observations in selected numeric DataFrame columns.

    Anomaly detection flags unusual observations, which are not necessarily
    errors. Rows missing a value in any selected column are not used to fit
    the model; they remain in the returned DataFrame with missing anomaly
    indicators and scores.

    Args:
        df: DataFrame containing the observations to check.
        columns: Non-empty list of numeric column names to use.
        contamination: Expected proportion of anomalies, greater than 0 and
            at most 0.5.

    Returns:
        A copy of ``df`` with ``is_anomaly`` (nullable boolean) and
        ``anomaly_score`` (numeric) columns added.

    Raises:
        TypeError: If ``df`` is not a DataFrame, ``columns`` is not a list,
            or ``contamination`` is not numeric.
        ValueError: If columns are empty, missing, duplicated, or non-numeric,
            contamination is outside its allowed range, or too few complete
            rows are available to fit the model.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if not isinstance(columns, list):
        raise TypeError("columns must be a list of column names")
    if not columns:
        raise ValueError("columns must contain at least one column name")
    try:
        has_duplicate_columns = len(set(columns)) != len(columns)
    except TypeError as error:
        raise TypeError("columns must contain hashable column names") from error
    if has_duplicate_columns:
        raise ValueError("columns must not contain duplicate column names")

    if isinstance(contamination, bool) or not isinstance(
        contamination, numbers.Real
    ):
        raise TypeError("contamination must be a numeric value")
    contamination = float(contamination)
    if not math.isfinite(contamination) or not 0 < contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5")

    for column in columns:
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

    # Fit only rows with a value in every selected feature.
    complete_mask = df[columns].notna().all(axis=1)
    complete_data = df.loc[complete_mask, columns]
    if len(complete_data) < 2:
        raise ValueError(
            "At least two complete rows are required to fit the anomaly detector"
        )
    feature_values = complete_data.to_numpy(dtype=float)
    if not bool(np.isfinite(feature_values).all()):
        raise ValueError(
            "Selected columns must contain finite numeric values in complete rows"
        )

    model = IsolationForest(random_state=42, contamination=contamination)
    model.fit(feature_values)
    predictions = model.predict(feature_values)
    scores = model.decision_function(feature_values)

    result = df.copy()
    anomaly_flags = pd.array([pd.NA] * len(df), dtype="boolean")
    anomaly_flags[complete_mask.to_numpy()] = predictions == -1
    anomaly_scores = [float("nan")] * len(df)
    for position, score in zip(complete_mask.to_numpy().nonzero()[0], scores):
        anomaly_scores[position] = float(score)

    # Assign extension/array values positionally to also handle duplicate indexes.
    result["is_anomaly"] = anomaly_flags
    result["anomaly_score"] = anomaly_scores
    return result