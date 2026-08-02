"""
analytics.py

The deterministic analytics engine. Everything in this module is plain
Pandas/NumPy — no AI involved. This is intentional: numbers that can be
calculated reliably with code should never be delegated to an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.data_profiler import ColumnProfile, get_parsed_date_series
from utils.constants import (
    REVENUE_KEYWORDS,
    COST_KEYWORDS,
    PROFIT_KEYWORDS,
    PRICE_KEYWORDS,
    UNITS_KEYWORDS,
    TREND_SIGNIFICANT_CHANGE_PCT,
)
from utils.helpers import normalize_column_name, percent_change, safe_divide


# ---------------------------------------------------------------------------
# Column role detection (which numeric column is "revenue", "profit", etc.)
# ---------------------------------------------------------------------------

def _find_best_match(columns: List[str], keywords: List[str]) -> Optional[str]:
    """Return the first numeric column whose normalized name contains a keyword."""
    for column in columns:
        normalized = normalize_column_name(column)
        for keyword in keywords:
            if keyword in normalized:
                return column
    return None


@dataclass
class MetricColumns:
    revenue: Optional[str] = None
    cost: Optional[str] = None
    profit: Optional[str] = None
    price: Optional[str] = None
    units: Optional[str] = None


def detect_metric_columns(numeric_columns: List[str]) -> MetricColumns:
    """Map generic numeric columns to business-metric roles using name heuristics."""
    return MetricColumns(
        revenue=_find_best_match(numeric_columns, REVENUE_KEYWORDS),
        cost=_find_best_match(numeric_columns, COST_KEYWORDS),
        profit=_find_best_match(numeric_columns, PROFIT_KEYWORDS),
        price=_find_best_match(numeric_columns, PRICE_KEYWORDS),
        units=_find_best_match(numeric_columns, UNITS_KEYWORDS),
    )


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def compute_numeric_summary(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Standard descriptive statistics for every numeric column."""
    if not numeric_columns:
        return pd.DataFrame()
    stats = df[numeric_columns].agg(["count", "sum", "mean", "median", "min", "max", "std"]).T
    stats = stats.rename(columns={"count": "Count", "sum": "Sum", "mean": "Mean",
                                   "median": "Median", "min": "Min", "max": "Max", "std": "Std Dev"})
    return stats.round(2)


def compute_categorical_summary(df: pd.DataFrame, categorical_columns: List[str], top_n: int = 5) -> Dict[str, pd.Series]:
    """Top value counts for each categorical column."""
    summary = {}
    for column in categorical_columns:
        try:
            counts = df[column].value_counts(dropna=True).head(top_n)
            if not counts.empty:
                summary[column] = counts
        except Exception:
            continue
    return summary


# ---------------------------------------------------------------------------
# KPI calculation
# ---------------------------------------------------------------------------

@dataclass
class KPIResult:
    label: str
    value: Optional[float]
    is_currency_like: bool = False
    is_percentage: bool = False
    delta: Optional[float] = None  # percent change vs. previous period, if available


def calculate_kpis(
    df: pd.DataFrame,
    metric_columns: MetricColumns,
    date_column: Optional[str] = None,
) -> List[KPIResult]:
    """
    Calculate whichever KPIs are possible given the columns that actually exist.
    Never assumes a column exists — every KPI is conditional.
    """
    kpis: List[KPIResult] = []

    total_revenue = None
    total_cost = None
    total_profit = None

    if metric_columns.revenue:
        total_revenue = float(df[metric_columns.revenue].sum())
        kpis.append(KPIResult(label="Total Revenue", value=total_revenue, is_currency_like=True))

    if metric_columns.cost:
        total_cost = float(df[metric_columns.cost].sum())
        kpis.append(KPIResult(label="Total Cost", value=total_cost, is_currency_like=True))

    if metric_columns.profit:
        total_profit = float(df[metric_columns.profit].sum())
        kpis.append(KPIResult(label="Total Profit", value=total_profit, is_currency_like=True))
    elif total_revenue is not None and total_cost is not None:
        # Derive profit only when both revenue and cost genuinely exist
        total_profit = total_revenue - total_cost
        kpis.append(KPIResult(label="Total Profit (derived)", value=total_profit, is_currency_like=True))

    if total_profit is not None and total_revenue not in (None, 0):
        margin = safe_divide(total_profit, total_revenue)
        if margin is not None:
            kpis.append(KPIResult(label="Profit Margin", value=margin * 100, is_percentage=True))

    if metric_columns.units:
        total_units = float(df[metric_columns.units].sum())
        kpis.append(KPIResult(label="Total Units Sold", value=total_units))

    # Average order value: revenue / number of transactions (rows)
    if metric_columns.revenue:
        transaction_count = len(df)
        aov = safe_divide(total_revenue, transaction_count)
        if aov is not None:
            kpis.append(KPIResult(label="Average Order Value", value=aov, is_currency_like=True))

    kpis.append(KPIResult(label="Number of Transactions", value=float(len(df))))

    return kpis


# ---------------------------------------------------------------------------
# Group-by rankings (revenue by product/region/etc.)
# ---------------------------------------------------------------------------

def compute_grouped_totals(
    df: pd.DataFrame, group_column: str, value_column: str, top_n: Optional[int] = None
) -> pd.Series:
    """Sum `value_column` grouped by `group_column`, sorted descending."""
    try:
        grouped = df.groupby(group_column)[value_column].sum().sort_values(ascending=False)
        if top_n:
            grouped = grouped.head(top_n)
        return grouped
    except Exception:
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Trend analysis (requires a date column + a numeric metric)
# ---------------------------------------------------------------------------

@dataclass
class TrendResult:
    period_label: str  # e.g. "Monthly"
    series: pd.Series  # indexed by period, values are metric totals
    current_period: Optional[str] = None
    previous_period: Optional[str] = None
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    absolute_change: Optional[float] = None
    percent_change_value: Optional[float] = None
    is_significant: bool = False


def _choose_frequency(date_series: pd.Series) -> Tuple[str, str]:
    """Pick daily/weekly/monthly aggregation based on the span of dates available."""
    valid_dates = date_series.dropna()
    if valid_dates.empty:
        return "ME", "Monthly"
    span_days = (valid_dates.max() - valid_dates.min()).days
    if span_days <= 31:
        return "D", "Daily"
    if span_days <= 120:
        return "W", "Weekly"
    return "ME", "Monthly"


def compute_trend(df: pd.DataFrame, date_column: str, metric_column: str) -> Optional[TrendResult]:
    """Aggregate a numeric metric over time and compare the two most recent periods."""
    if date_column not in df.columns or metric_column not in df.columns:
        return None

    parsed_dates = get_parsed_date_series(df, date_column)
    working = pd.DataFrame({"__date": parsed_dates, "__value": df[metric_column]}).dropna(subset=["__date"])

    if working.empty or working["__value"].dropna().empty:
        return None

    freq, freq_label = _choose_frequency(working["__date"])

    working = working.set_index("__date")
    aggregated = working["__value"].resample(freq).sum()
    aggregated = aggregated[aggregated.index <= aggregated.index.max()]  # trim to real data

    if len(aggregated) < 2:
        return TrendResult(period_label=freq_label, series=aggregated)

    current_value = float(aggregated.iloc[-1])
    previous_value = float(aggregated.iloc[-2])
    absolute_change = current_value - previous_value
    pct_change = percent_change(current_value, previous_value)
    is_significant = pct_change is not None and abs(pct_change) >= TREND_SIGNIFICANT_CHANGE_PCT

    return TrendResult(
        period_label=freq_label,
        series=aggregated,
        current_period=str(aggregated.index[-1].date()) if hasattr(aggregated.index[-1], "date") else str(aggregated.index[-1]),
        previous_period=str(aggregated.index[-2].date()) if hasattr(aggregated.index[-2], "date") else str(aggregated.index[-2]),
        current_value=current_value,
        previous_value=previous_value,
        absolute_change=absolute_change,
        percent_change_value=pct_change,
        is_significant=is_significant,
    )


def compute_group_trends(
    df: pd.DataFrame, date_column: str, metric_column: str, group_column: str, top_n: int = 5
) -> Dict[str, TrendResult]:
    """Compute period-over-period trend for the top-N groups within a categorical column."""
    results: Dict[str, TrendResult] = {}
    try:
        top_groups = compute_grouped_totals(df, group_column, metric_column, top_n=top_n).index.tolist()
    except Exception:
        return results

    for group_value in top_groups:
        subset = df[df[group_column] == group_value]
        trend = compute_trend(subset, date_column, metric_column)
        if trend is not None and trend.percent_change_value is not None:
            results[str(group_value)] = trend

    return results


def compute_correlation_matrix(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Correlation matrix for numeric columns, when there are at least 2."""
    if len(numeric_columns) < 2:
        return pd.DataFrame()
    try:
        return df[numeric_columns].corr(numeric_only=True).round(2)
    except Exception:
        return pd.DataFrame()
