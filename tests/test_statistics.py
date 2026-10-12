import pandas as pd
import pytest

from tools.statistics import calculate_correlation, calculate_statistics


@pytest.fixture
def employee_data():
    return pd.DataFrame(
        {
            "CGPA": [8.4, 7.2, 9.1, 8.0, 6.9],
            "Salary": [60000, 40000, 80000, 70000, 35000],
            "Branch": ["CSE", "ECE", "CSE", "ME", "ECE"],
        }
    )


def test_calculate_statistics_returns_expected_salary_summary(employee_data):
    result = calculate_statistics(employee_data, "Salary")

    assert result["count"] == 5
    assert result["mean"] == 57000
    assert result["median"] == 60000
    assert result["minimum"] == 35000
    assert result["maximum"] == 80000
    assert result["25th_percentile"] == 40000
    assert result["50th_percentile"] == 60000
    assert result["75th_percentile"] == 70000


def test_calculate_statistics_ignores_missing_values(employee_data):
    employee_data.loc[1, "Salary"] = None

    result = calculate_statistics(employee_data, "Salary")

    assert result["count"] == 4
    assert result["mean"] == 61250
    assert result["median"] == 65000
    assert result["minimum"] == 35000
    assert result["maximum"] == 80000


def test_calculate_statistics_rejects_invalid_column(employee_data):
    with pytest.raises(ValueError, match="does not exist"):
        calculate_statistics(employee_data, "Bonus")


def test_calculate_statistics_rejects_non_numeric_column(employee_data):
    with pytest.raises(ValueError, match="real numeric"):
        calculate_statistics(employee_data, "Branch")


def test_calculate_statistics_rejects_all_missing_numeric_column():
    df = pd.DataFrame({"Salary": [None, float("nan")]})

    with pytest.raises(ValueError, match="no valid numeric values"):
        calculate_statistics(df, "Salary")


def test_calculate_correlation_matches_pandas(employee_data):
    result = calculate_correlation(employee_data, "CGPA", "Salary")

    expected_correlation = employee_data["CGPA"].corr(employee_data["Salary"])
    assert result["column1"] == "CGPA"
    assert result["column2"] == "Salary"
    assert result["correlation"] == pytest.approx(expected_correlation)
    assert result["pairs_used"] == 5


def test_calculate_correlation_ignores_rows_with_missing_pairs(employee_data):
    employee_data.loc[1, "CGPA"] = None
    employee_data.loc[3, "Salary"] = None
    complete_pairs = employee_data[["CGPA", "Salary"]].dropna()

    result = calculate_correlation(employee_data, "CGPA", "Salary")

    assert result["correlation"] == pytest.approx(
        complete_pairs["CGPA"].corr(complete_pairs["Salary"])
    )
    assert result["pairs_used"] == 3


@pytest.mark.parametrize(
    ("column1", "column2", "missing_column"),
    [
        ("GPA", "Salary", "GPA"),
        ("CGPA", "Bonus", "Bonus"),
    ],
)
def test_calculate_correlation_rejects_invalid_columns(
    employee_data, column1, column2, missing_column
):
    with pytest.raises(ValueError, match=f"Column '{missing_column}'"):
        calculate_correlation(employee_data, column1, column2)


def test_calculate_correlation_rejects_non_numeric_column(employee_data):
    with pytest.raises(ValueError, match="real numeric"):
        calculate_correlation(employee_data, "CGPA", "Branch")


@pytest.mark.parametrize(
    ("column1", "column2"),
    [
        ("Constant", "Salary"),
        ("CGPA", "Constant"),
    ],
)
def test_calculate_correlation_rejects_constant_columns(
    employee_data, column1, column2
):
    employee_data["Constant"] = [1, 1, 1, 1, 1]

    with pytest.raises(ValueError, match="no variation"):
        calculate_correlation(employee_data, column1, column2)