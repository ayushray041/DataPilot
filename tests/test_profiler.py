import pandas as pd
import pytest

from tools.profiler import profile_dataset


def test_profile_dataset_with_normal_data():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "age": [36, 40],
        }
    )

    result = profile_dataset(df)

    assert result == {
        "rows": 2,
        "columns": 2,
        "column_names": ["id", "age"],
        "data_types": {"id": "int64", "age": "int64"},
        "missing_values": {"id": 0, "age": 0},
        "unique_values": {"id": 2, "age": 2},
        "duplicate_rows": 0,
    }


def test_profile_dataset_counts_missing_values():
    df = pd.DataFrame(
        {
            "score": [10.0, None, 10.0],
            "comment": ["good", None, "good"],
        }
    )

    result = profile_dataset(df)

    assert result["missing_values"] == {"score": 1, "comment": 1}
    assert result["unique_values"] == {"score": 1, "comment": 1}


def test_profile_dataset_counts_duplicate_rows():
    df = pd.DataFrame(
        {
            "item": ["pen", "book", "pen"],
            "quantity": [2, 1, 2],
        }
    )

    result = profile_dataset(df)

    assert result["duplicate_rows"] == 1


def test_profile_dataset_with_empty_dataframe():
    df = pd.DataFrame(columns=["name", "age"])

    result = profile_dataset(df)

    assert result == {
        "rows": 0,
        "columns": 2,
        "column_names": ["name", "age"],
        "data_types": {"name": "object", "age": "object"},
        "missing_values": {"name": 0, "age": 0},
        "unique_values": {"name": 0, "age": 0},
        "duplicate_rows": 0,
    }


def test_profile_dataset_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        profile_dataset([1, 2, 3])
