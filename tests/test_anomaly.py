import numpy as np
import pandas as pd
import pytest

from tools.anomaly import detect_anomalies


def test_detects_extreme_outlier_without_modifying_input():
    normal_values = np.linspace(9.9, 10.1, 30)
    df = pd.DataFrame({"measurement": np.append(normal_values, 1000.0)})
    original = df.copy(deep=True)

    result = detect_anomalies(df, ["measurement"], contamination=0.05)

    assert len(result) == len(df)
    assert "is_anomaly" in result.columns
    assert "anomaly_score" in result.columns
    assert result["is_anomaly"].notna().all()
    assert result["is_anomaly"].dtype == pd.BooleanDtype()
    assert result.loc[30, "is_anomaly"]
    assert np.isfinite(result["anomaly_score"]).all()
    pd.testing.assert_frame_equal(df, original)


def test_missing_rows_are_preserved_but_excluded_from_model():
    df = pd.DataFrame(
        {
            "first": [1.0, 1.1, 0.9, 1.05, 100.0, np.nan],
            "second": [2.0, 2.1, 1.9, 2.05, 100.0, 2.0],
        }
    )
    original = df.copy(deep=True)

    result = detect_anomalies(
        df,
        ["first", "second"],
        contamination=0.2,
    )

    assert len(result) == len(df)
    assert pd.isna(result.loc[5, "is_anomaly"])
    assert pd.isna(result.loc[5, "anomaly_score"])
    assert result["is_anomaly"].iloc[:5].notna().all()
    assert result["anomaly_score"].iloc[:5].notna().all()
    assert result["is_anomaly"].iloc[4]
    pd.testing.assert_frame_equal(df, original)


def test_detect_anomalies_rejects_empty_columns_list():
    df = pd.DataFrame({"value": [1.0, 2.0]})

    with pytest.raises(ValueError, match="at least one column"):
        detect_anomalies(df, [])


def test_detect_anomalies_rejects_missing_column():
    df = pd.DataFrame({"value": [1.0, 2.0]})

    with pytest.raises(ValueError, match="does not exist"):
        detect_anomalies(df, ["missing"])


def test_detect_anomalies_rejects_non_numeric_column():
    df = pd.DataFrame({"value": ["one", "two"]})

    with pytest.raises(ValueError, match="numeric"):
        detect_anomalies(df, ["value"])


@pytest.mark.parametrize("contamination", [0, 0.6])
def test_detect_anomalies_rejects_invalid_contamination(contamination):
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="contamination"):
        detect_anomalies(df, ["value"], contamination=contamination)


def test_detect_anomalies_requires_at_least_two_complete_rows():
    df = pd.DataFrame({"value": [1.0, np.nan, np.nan]})

    with pytest.raises(ValueError, match="At least two complete rows"):
        detect_anomalies(df, ["value"])