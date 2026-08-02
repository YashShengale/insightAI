"""
charts.py

Builds Plotly figures dynamically based on which column types are actually
present in the dataset. Every function returns None if it can't produce a
meaningful chart, so the caller can skip it safely instead of crashing.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.analytics import MetricColumns, compute_grouped_totals
from modules.data_profiler import get_parsed_date_series

CHART_TEMPLATE = "plotly_white"


def build_trend_line_chart(
    df: pd.DataFrame, date_column: str, metric_column: str, freq: str = "ME", freq_label: str = "Monthly"
) -> Optional[go.Figure]:
    """Line chart of a metric aggregated over time."""
    parsed = get_parsed_date_series(df, date_column)
    working = pd.DataFrame({"Date": parsed, "Value": df[metric_column]}).dropna()
    if working.empty:
        return None

    aggregated = working.set_index("Date")["Value"].resample(freq).sum().reset_index()
    if len(aggregated) < 2:
        return None

    fig = px.line(
        aggregated,
        x="Date",
        y="Value",
        markers=True,
        title=f"{metric_column} Over Time ({freq_label})",
        template=CHART_TEMPLATE,
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric_column,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_grouped_bar_chart(
    df: pd.DataFrame, group_column: str, value_column: str, top_n: int = 10
) -> Optional[go.Figure]:
    """Horizontal bar chart of a metric summed by a categorical column."""
    grouped = compute_grouped_totals(df, group_column, value_column, top_n=top_n)
    if grouped.empty:
        return None

    fig = px.bar(
        x=grouped.values,
        y=grouped.index.astype(str),
        orientation="h",
        title=f"{value_column} by {group_column}",
        template=CHART_TEMPLATE,
        labels={"x": value_column, "y": group_column},
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_distribution_chart(df: pd.DataFrame, numeric_column: str) -> Optional[go.Figure]:
    """Histogram showing the distribution of a numeric column."""
    series = df[numeric_column].dropna()
    if series.empty or series.nunique() < 2:
        return None

    fig = px.histogram(
        series,
        x=numeric_column,
        title=f"Distribution of {numeric_column}",
        template=CHART_TEMPLATE,
        nbins=30,
    )
    fig.update_layout(
        yaxis_title="Frequency",
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


def build_top_categories_pie(df: pd.DataFrame, group_column: str, value_column: str, top_n: int = 6) -> Optional[go.Figure]:
    """Pie/donut chart of top category shares for a metric."""
    grouped = compute_grouped_totals(df, group_column, value_column, top_n=top_n)
    if grouped.empty or len(grouped) < 2:
        return None

    fig = px.pie(
        values=grouped.values,
        names=grouped.index.astype(str),
        title=f"{value_column} Share by {group_column}",
        template=CHART_TEMPLATE,
        hole=0.4,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_correlation_heatmap(corr_matrix: pd.DataFrame) -> Optional[go.Figure]:
    """Heatmap of a precomputed correlation matrix."""
    if corr_matrix is None or corr_matrix.empty or corr_matrix.shape[0] < 2:
        return None

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        title="Correlation Between Numeric Columns",
        template=CHART_TEMPLATE,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig


def build_dashboard_charts(
    df: pd.DataFrame,
    metric_columns: MetricColumns,
    date_columns: List[str],
    categorical_columns: List[str],
    numeric_columns: List[str],
) -> List[go.Figure]:
    """
    Assemble a curated set of 3-6 charts based on what's actually available.
    This is the main entry point used by the Streamlit UI.
    """
    figures: List[go.Figure] = []

    primary_metric = metric_columns.revenue or metric_columns.profit or (numeric_columns[0] if numeric_columns else None)
    primary_date = date_columns[0] if date_columns else None

    # 1. Trend over time (if we have a date + a metric)
    if primary_date and primary_metric:
        fig = build_trend_line_chart(df, primary_date, primary_metric)
        if fig:
            figures.append(fig)

    # 2. Profit trend, if distinct from primary metric
    if primary_date and metric_columns.profit and metric_columns.profit != primary_metric:
        fig = build_trend_line_chart(df, primary_date, metric_columns.profit)
        if fig:
            figures.append(fig)

    # 3 & 4. Metric by top categorical columns (e.g. product, region)
    charts_added_for_categories = 0
    for cat_col in categorical_columns:
        if charts_added_for_categories >= 2:
            break
        if primary_metric:
            fig = build_grouped_bar_chart(df, cat_col, primary_metric)
            if fig:
                figures.append(fig)
                charts_added_for_categories += 1

    # 5. Pie chart for top categories of the primary metric (only if we have a distinct categorical column left)
    if categorical_columns and primary_metric:
        fig = build_top_categories_pie(df, categorical_columns[0], primary_metric)
        if fig:
            figures.append(fig)

    # 6. Distribution of the primary metric
    if primary_metric:
        fig = build_distribution_chart(df, primary_metric)
        if fig:
            figures.append(fig)

    # Cap at 6 charts to keep the dashboard focused
    return figures[:6]
