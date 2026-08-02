"""
data_profiler.py

Detects column types (numeric, categorical, date, boolean, identifier) and
computes data-health statistics (missing values, duplicates, quality score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.constants import (
    DATE_KEYWORDS,
    ID_KEYWORDS,
    MISSING_WEIGHT,
    DUPLICATE_WEIGHT,
)
from utils.helpers import normalize_column_name


@dataclass
class ColumnProfile:
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    boolean_columns: List[str] = field(default_factory=list)
    identifier_columns: List[str] = field(default_factory=list)


@dataclass
class DataHealth:
    row_count: int
    column_count: int
    missing_cells: int
    missing_percentage: float
    duplicate_rows: int
    duplicate_percentage: float
    missing_by_column: Dict[str, int]
    quality_score: float


def _looks_like_identifier(series: pd.Series, column_name: str) -> bool:
    """Heuristic: near-unique values + id-like column name => identifier column."""
    normalized = normalize_column_name(column_name)
    name_hint = any(keyword in normalized for keyword in ID_KEYWORDS)
    if len(series) == 0:
        return False
    uniqueness_ratio = series.nunique(dropna=True) / max(len(series), 1)
    return name_hint and uniqueness_ratio > 0.9


def _looks_like_date_by_name(column_name: str) -> bool:
    normalized = normalize_column_name(column_name)
    return any(keyword in normalized for keyword in DATE_KEYWORDS)


def _try_parse_dates(series: pd.Series) -> pd.Series:
    """Attempt to parse a column as dates. Returns parsed series (may be all-NaT on failure)."""
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(series, errors="coerce")
        except Exception:
            return pd.Series([pd.NaT] * len(series))


def detect_column_types(df: pd.DataFrame) -> ColumnProfile:
    """Classify each column into numeric / categorical / date / boolean / identifier."""
    profile = ColumnProfile()

    for column in df.columns:
        series = df[column]

        # Boolean check first (before numeric, since bool is a numpy numeric subtype)
        if pd.api.types.is_bool_dtype(series):
            profile.boolean_columns.append(column)
            continue

        # Already-parsed datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            profile.date_columns.append(column)
            continue

        # Numeric dtype
        if pd.api.types.is_numeric_dtype(series):
            if _looks_like_identifier(series, column):
                profile.identifier_columns.append(column)
            else:
                profile.numeric_columns.append(column)
            continue

        # Object / string columns: could be dates, identifiers, or categories
        non_null = series.dropna()
        if non_null.empty:
            profile.categorical_columns.append(column)
            continue

        # Try date parsing if the name hints at a date, or a sample parses cleanly
        should_try_date = _looks_like_date_by_name(column)
        if not should_try_date:
            sample = non_null.head(20)
            parsed_sample = _try_parse_dates(sample)
            if parsed_sample.notna().mean() > 0.8:
                should_try_date = True

        if should_try_date:
            parsed_full = _try_parse_dates(non_null)
            if parsed_full.notna().mean() > 0.7:
                profile.date_columns.append(column)
                continue

        if _looks_like_identifier(series, column):
            profile.identifier_columns.append(column)
            continue

        profile.categorical_columns.append(column)

    return profile


def get_parsed_date_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Return a properly parsed datetime series for a detected date column."""
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        return df[column]
    return _try_parse_dates(df[column])


def compute_data_health(df: pd.DataFrame) -> DataHealth:
    """Compute missing values, duplicates, and an honest data quality score."""
    row_count = len(df)
    column_count = len(df.columns)
    total_cells = max(row_count * column_count, 1)

    missing_by_column = df.isna().sum().to_dict()
    missing_cells = int(sum(missing_by_column.values()))
    missing_percentage = round((missing_cells / total_cells) * 100, 2)

    duplicate_rows = int(df.duplicated().sum())
    duplicate_percentage = round((duplicate_rows / max(row_count, 1)) * 100, 2)

    # Quality score: 100 minus weighted penalties for missing data and duplicates.
    # This is a simple, transparent formula (not an AI-generated score):
    #   penalty = (missing_ratio * MISSING_WEIGHT) + (duplicate_ratio * DUPLICATE_WEIGHT)
    #   score   = max(0, 100 - penalty)
    missing_ratio = missing_cells / total_cells
    duplicate_ratio = duplicate_rows / max(row_count, 1)
    penalty = (missing_ratio * MISSING_WEIGHT) + (duplicate_ratio * DUPLICATE_WEIGHT)
    quality_score = round(max(0.0, 100.0 - penalty), 1)

    return DataHealth(
        row_count=row_count,
        column_count=column_count,
        missing_cells=missing_cells,
        missing_percentage=missing_percentage,
        duplicate_rows=duplicate_rows,
        duplicate_percentage=duplicate_percentage,
        missing_by_column={k: int(v) for k, v in missing_by_column.items()},
        quality_score=quality_score,
    )
