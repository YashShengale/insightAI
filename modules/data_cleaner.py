"""
data_cleaner.py

Performs only SAFE, non-destructive automatic cleaning. The original
uploaded DataFrame is never mutated in place — a cleaned copy is returned
alongside a human-readable log of what was changed, so the user always
knows what happened to their data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from modules.data_profiler import get_parsed_date_series, detect_column_types


@dataclass
class CleaningResult:
    dataframe: pd.DataFrame
    operations_log: List[str] = field(default_factory=list)


def clean_dataset(df: pd.DataFrame, remove_duplicates: bool = False) -> CleaningResult:
    """
    Apply safe cleaning steps:
      1. Strip whitespace from column names.
      2. Drop fully-empty rows.
      3. Drop fully-empty columns.
      4. Trim whitespace inside string/object cell values.
      5. Optionally remove exact duplicate rows (user-controlled, off by default).
      6. Parse detected date columns into real datetime dtype.

    Business values (numeric data) are never modified.
    """
    log: List[str] = []
    cleaned = df.copy(deep=True)

    # 1. Normalize column names (whitespace only — never rename semantics)
    original_columns = list(cleaned.columns)
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    if list(cleaned.columns) != original_columns:
        log.append("Trimmed leading/trailing whitespace from column names.")

    # 2. Drop fully empty rows
    rows_before = len(cleaned)
    cleaned = cleaned.dropna(how="all")
    rows_removed = rows_before - len(cleaned)
    if rows_removed > 0:
        log.append(f"Removed {rows_removed} completely empty row(s).")

    # 3. Drop fully empty columns
    cols_before = set(cleaned.columns)
    cleaned = cleaned.dropna(axis=1, how="all")
    cols_removed = cols_before - set(cleaned.columns)
    if cols_removed:
        log.append(f"Removed {len(cols_removed)} completely empty column(s): {', '.join(sorted(cols_removed))}.")

    # 4. Trim whitespace in string/object columns only
    trimmed_any = False
    try:
        string_like_columns = cleaned.select_dtypes(include=["object", "str"]).columns
    except TypeError:
        # Older Pandas versions don't recognize "str" as a dtype name
        string_like_columns = cleaned.select_dtypes(include=["object"]).columns

    for column in string_like_columns:
        try:
            new_series = cleaned[column].apply(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            if not new_series.equals(cleaned[column]):
                trimmed_any = True
            cleaned[column] = new_series
        except Exception:
            continue
    if trimmed_any:
        log.append("Trimmed extra whitespace inside text cells.")

    # 5. Optional duplicate removal (explicit opt-in, since removing rows
    #    is a more consequential change than the steps above)
    if remove_duplicates:
        dup_count = int(cleaned.duplicated().sum())
        if dup_count > 0:
            cleaned = cleaned.drop_duplicates()
            log.append(f"Removed {dup_count} exact duplicate row(s).")
        else:
            log.append("No exact duplicate rows found.")

    # 6. Parse detected date columns safely (adds parsed dtype, doesn't drop
    #    rows that fail to parse — those simply become NaT and are reported)
    profile = detect_column_types(cleaned)
    for date_col in profile.date_columns:
        try:
            parsed = get_parsed_date_series(cleaned, date_col)
            unparseable = int(parsed.isna().sum() - cleaned[date_col].isna().sum())
            cleaned[date_col] = parsed
            if unparseable > 0:
                log.append(
                    f"Converted '{date_col}' to date format ({unparseable} value(s) could not be parsed)."
                )
            else:
                log.append(f"Converted '{date_col}' to date format.")
        except Exception:
            continue

    if not log:
        log.append("No cleaning was necessary — the dataset was already tidy.")

    return CleaningResult(dataframe=cleaned, operations_log=log)
