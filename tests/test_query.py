import pandas as pd
import pytest

from tools.query import filter_data, sort_data


@pytest.fixture
def student_data():
    return pd.DataFrame(
        {
            "Name": ["Asha", "Ben", "Chitra", "Dev", "Esha"],
            "CGPA": [7, 8, 9, 6, 8],
            "Salary": [70000, 80000, None, 60000, 90000],
            "Branch": ["CSE", "ECE", "CSE", "ME", "CSE"],
        }
    )


def test_filter_numeric_value_as_number_and_string(student_data):
    expected = student_data.iloc[[1, 4]]

    pd.testing.assert_frame_equal(
        filter_data(student_data, "CGPA", "==", 8),
        expected,
    )
    pd.testing.assert_frame_equal(
        filter_data(student_data, "CGPA", "==", "8"),
        expected,
    )


def test_filter_text_value(student_data):
    pd.testing.assert_frame_equal(
        filter_data(student_data, "Branch", "==", "CSE"),
        student_data.iloc[[0, 2, 4]],
    )


@pytest.mark.parametrize(
    ("operator", "value", "expected_indices"),
    [
        ("==", 8, [1, 4]),
        ("!=", 8, [0, 2, 3]),
        (">", 8, [2]),
        ("<", 8, [0, 3]),
        (">=", 8, [1, 2, 4]),
        ("<=", 8, [0, 1, 3, 4]),
    ],
)
def test_filter_supports_each_comparison_operator(
    student_data, operator, value, expected_indices
):
    pd.testing.assert_frame_equal(
        filter_data(student_data, "CGPA", operator, value),
        student_data.iloc[expected_indices],
    )


def test_filter_rejects_invalid_column_and_operator(student_data):
    with pytest.raises(ValueError, match="does not exist"):
        filter_data(student_data, "GPA", "==", 8)

    with pytest.raises(ValueError, match="Unsupported operator"):
        filter_data(student_data, "CGPA", "contains", 8)


def test_filter_and_sort_do_not_modify_original_dataframe(student_data):
    original = student_data.copy(deep=True)

    filter_data(student_data, "CGPA", ">=", 8)
    sort_data(student_data, "Salary", descending=False)

    pd.testing.assert_frame_equal(student_data, original)


def test_sort_ascending_places_missing_salary_last(student_data):
    result = sort_data(student_data, "Salary", descending=False)

    assert result["Name"].tolist() == ["Dev", "Asha", "Ben", "Esha", "Chitra"]
    assert result["Salary"].iloc[:-1].tolist() == [60000.0, 70000.0, 80000.0, 90000.0]
    assert pd.isna(result["Salary"].iloc[-1])


def test_sort_descending_places_missing_salary_last(student_data):
    result = sort_data(student_data, "Salary", descending=True)

    assert result["Name"].tolist() == ["Esha", "Ben", "Asha", "Dev", "Chitra"]
    assert result["Salary"].iloc[:-1].tolist() == [90000.0, 80000.0, 70000.0, 60000.0]
    assert pd.isna(result["Salary"].iloc[-1])


def test_sort_applies_limit_after_sorting(student_data):
    result = sort_data(student_data, "Salary", descending=True, limit=2)

    assert result["Name"].tolist() == ["Esha", "Ben"]
    assert result["Salary"].tolist() == [90000.0, 80000.0]


def test_sort_rejects_invalid_column(student_data):
    with pytest.raises(ValueError, match="does not exist"):
        sort_data(student_data, "GPA")


@pytest.mark.parametrize("limit", [0, -1])
def test_sort_rejects_non_positive_limits(student_data, limit):
    with pytest.raises(ValueError, match="positive integer"):
        sort_data(student_data, "Salary", limit=limit)


@pytest.mark.parametrize("limit", [1.5, True])
def test_sort_rejects_non_integer_limits(student_data, limit):
    with pytest.raises(TypeError, match="positive integer"):
        sort_data(student_data, "Salary", limit=limit)