"""
Lightweight tests for the pure-Python data-processing functions.
Run with: python -m pytest tests/ -v
(or: python tests/test_core.py to run without pytest)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from modules.data_profiler import detect_column_types, compute_data_health
from modules.data_cleaner import clean_dataset
from modules.analytics import (
    detect_metric_columns,
    calculate_kpis,
    compute_grouped_totals,
    compute_trend,
    compute_correlation_matrix,
)
from utils.helpers import format_number, format_compact_number, percent_change, safe_divide


def make_sample_df():
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=6, freq="MS"),
        "Product": ["Laptop", "Mouse", "Laptop", "Mouse", "Laptop", "Mouse"],
        "Region": ["Mumbai", "Pune", "Mumbai", "Pune", "Mumbai", "Pune"],
        "Revenue": [1000.0, 200.0, 1200.0, 250.0, 900.0, 400.0],
        "Cost": [700.0, 100.0, 800.0, 120.0, 650.0, 150.0],
    })


def test_detect_column_types():
    df = make_sample_df()
    profile = detect_column_types(df)
    assert "Revenue" in profile.numeric_columns
    assert "Cost" in profile.numeric_columns
    assert "Product" in profile.categorical_columns
    assert "Region" in profile.categorical_columns
    assert "Date" in profile.date_columns
    print("test_detect_column_types passed")


def test_data_health_no_issues():
    df = make_sample_df()
    health = compute_data_health(df)
    assert health.row_count == 6
    assert health.missing_cells == 0
    assert health.duplicate_rows == 0
    assert health.quality_score == 100.0
    print("test_data_health_no_issues passed")


def test_data_health_with_missing_and_duplicates():
    df = make_sample_df()
    df.loc[0, "Revenue"] = np.nan
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)  # add one duplicate row
    health = compute_data_health(df)
    assert health.missing_cells == 1
    assert health.duplicate_rows == 1
    assert 0 <= health.quality_score < 100
    print("test_data_health_with_missing_and_duplicates passed")


def test_clean_dataset_removes_empty_rows_and_trims():
    df = make_sample_df()
    df.loc[6] = [pd.NaT, None, None, None, None]
    df.loc[0, "Product"] = "  Laptop  "
    result = clean_dataset(df, remove_duplicates=False)
    assert len(result.dataframe) == 6  # empty row removed
    assert result.dataframe.loc[0, "Product"] == "Laptop"
    assert len(result.operations_log) > 0
    print("test_clean_dataset_removes_empty_rows_and_trims passed")


def test_clean_dataset_never_mutates_original():
    df = make_sample_df()
    original_copy = df.copy(deep=True)
    clean_dataset(df, remove_duplicates=False)
    pd.testing.assert_frame_equal(df, original_copy)
    print("test_clean_dataset_never_mutates_original passed")


def test_detect_metric_columns():
    numeric_cols = ["Revenue", "Cost", "Profit_Margin", "Unit_Price", "Units_Sold"]
    metrics = detect_metric_columns(numeric_cols)
    assert metrics.revenue == "Revenue"
    assert metrics.cost == "Cost"
    assert metrics.price == "Unit_Price"
    assert metrics.units == "Units_Sold"
    print("test_detect_metric_columns passed")


def test_calculate_kpis_with_revenue_and_cost():
    df = make_sample_df()
    metrics = detect_metric_columns(["Revenue", "Cost"])
    kpis = calculate_kpis(df, metrics)
    labels = {k.label: k.value for k in kpis}
    assert labels["Total Revenue"] == 3950.0
    assert labels["Total Cost"] == 2520.0
    assert abs(labels["Total Profit (derived)"] - 1430.0) < 0.001
    print("test_calculate_kpis_with_revenue_and_cost passed")


def test_calculate_kpis_missing_columns_does_not_crash():
    df = pd.DataFrame({"Notes": ["a", "b", "c"]})
    metrics = detect_metric_columns([])
    kpis = calculate_kpis(df, metrics)
    # Should still return at least "Number of Transactions" without crashing
    assert any(k.label == "Number of Transactions" for k in kpis)
    print("test_calculate_kpis_missing_columns_does_not_crash passed")


def test_compute_grouped_totals():
    df = make_sample_df()
    grouped = compute_grouped_totals(df, "Product", "Revenue")
    assert grouped["Laptop"] == 3100.0
    assert grouped["Mouse"] == 850.0
    print("test_compute_grouped_totals passed")


def test_compute_trend():
    df = make_sample_df()
    trend = compute_trend(df, "Date", "Revenue")
    assert trend is not None
    assert trend.percent_change_value is not None
    print("test_compute_trend passed")


def test_correlation_matrix_shape():
    df = make_sample_df()
    corr = compute_correlation_matrix(df, ["Revenue", "Cost"])
    assert corr.shape == (2, 2)
    print("test_correlation_matrix_shape passed")


def test_helpers_safe_divide_and_percent_change():
    assert safe_divide(10, 0) is None
    assert safe_divide(10, 2) == 5
    assert percent_change(120, 100) == 20.0
    assert percent_change(50, 0) is None
    print("test_helpers_safe_divide_and_percent_change passed")


def test_helpers_formatting():
    assert format_number(None) == "N/A"
    assert format_number(1234.5, decimals=1) == "1,234.5"
    assert format_compact_number(12_400_000) == "12.40M"
    print("test_helpers_formatting passed")


def test_empty_dataframe_does_not_crash():
    df = pd.DataFrame()
    health = compute_data_health(df)
    assert health.row_count == 0
    profile = detect_column_types(df)
    assert profile.numeric_columns == []
    print("test_empty_dataframe_does_not_crash passed")


if __name__ == "__main__":
    test_functions = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for fn in test_functions:
        fn()
    print(f"\nAll {len(test_functions)} tests passed.")
