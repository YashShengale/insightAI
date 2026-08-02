"""
ai_insights.py

Builds a compact, structured text summary of the dataset (never the raw
DataFrame) and uses it to generate AI business insights. The same summary
builder is reused by ai_chat.py, recommendations.py, and report_generator.py
so every AI feature is grounded in the exact same facts.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from modules.analytics import (
    MetricColumns,
    KPIResult,
    calculate_kpis,
    compute_grouped_totals,
    compute_group_trends,
    compute_trend,
    compute_numeric_summary,
    compute_categorical_summary,
)
from modules.ai_service import AIService, AIResponse, get_ai_service
from utils.helpers import format_number, format_percentage, truncate_text


def build_dataset_context(
    df: pd.DataFrame,
    file_name: str,
    metric_columns: MetricColumns,
    date_columns: List[str],
    categorical_columns: List[str],
    numeric_columns: List[str],
    max_chars: int = 6000,
) -> str:
    """
    Construct a compact, structured summary of the dataset for use in AI
    prompts. Only pre-calculated facts go in here — the AI never sees the
    raw rows, and never has to compute anything itself.
    """
    lines: List[str] = []
    lines.append(f"Dataset: {file_name}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append("")

    kpis = calculate_kpis(df, metric_columns)
    if kpis:
        lines.append("Key Metrics:")
        for kpi in kpis:
            if kpi.value is None:
                continue
            if kpi.is_percentage:
                lines.append(f"- {kpi.label}: {format_percentage(kpi.value)}")
            else:
                lines.append(f"- {kpi.label}: {format_number(kpi.value, decimals=2)}")
        lines.append("")

    primary_metric = metric_columns.revenue or metric_columns.profit or (numeric_columns[0] if numeric_columns else None)

    # Top performer per categorical column (product, region, etc.)
    if primary_metric:
        for cat_col in categorical_columns[:3]:
            grouped = compute_grouped_totals(df, cat_col, primary_metric, top_n=5)
            if grouped.empty:
                continue
            lines.append(f"Top {cat_col} by {primary_metric}:")
            for name, value in grouped.items():
                lines.append(f"- {name}: {format_number(value, decimals=2)}")
            lines.append("")

    # Trend (overall)
    if date_columns and primary_metric:
        trend = compute_trend(df, date_columns[0], primary_metric)
        if trend and trend.percent_change_value is not None:
            direction = "increased" if trend.percent_change_value >= 0 else "decreased"
            lines.append(
                f"Overall {primary_metric} trend ({trend.period_label}): "
                f"{direction} {abs(trend.percent_change_value):.1f}% "
                f"(from {format_number(trend.previous_value, 2)} to {format_number(trend.current_value, 2)})."
            )
            lines.append("")

    # Notable changes by category (e.g. per-region trend)
    if date_columns and primary_metric and categorical_columns:
        group_trends = compute_group_trends(df, date_columns[0], primary_metric, categorical_columns[0], top_n=5)
        significant = {k: v for k, v in group_trends.items() if v.is_significant}
        if significant:
            lines.append("Important Changes:")
            for name, trend in significant.items():
                direction = "+" if trend.percent_change_value >= 0 else ""
                lines.append(f"- {name}: {direction}{trend.percent_change_value:.1f}%")
            lines.append("")

    # Numeric statistics (compact)
    numeric_stats = compute_numeric_summary(df, numeric_columns[:8])
    if not numeric_stats.empty:
        lines.append("Numerical Statistics:")
        for col in numeric_stats.index:
            row = numeric_stats.loc[col]
            lines.append(
                f"- {col}: sum={row['Sum']:.2f}, mean={row['Mean']:.2f}, "
                f"min={row['Min']:.2f}, max={row['Max']:.2f}"
            )
        lines.append("")

    # Categorical summaries (compact)
    cat_summary = compute_categorical_summary(df, categorical_columns[:5], top_n=3)
    if cat_summary:
        lines.append("Categorical Summaries:")
        for col, counts in cat_summary.items():
            top_items = ", ".join(f"{idx} ({val})" for idx, val in counts.items())
            lines.append(f"- {col}: {top_items}")
        lines.append("")

    context = "\n".join(lines)
    return truncate_text(context, max_chars=max_chars)


def generate_ai_insights(context: str, ai_service: Optional[AIService] = None) -> AIResponse:
    """Generate 3-6 business insights grounded strictly in the dataset context."""
    service = ai_service or get_ai_service()

    prompt = f"""Here is the structured summary of a business dataset:

{context}

Task: Act as an experienced business data analyst. Based ONLY on the facts above,
write 3 to 6 concise key insights as a numbered list. Each insight should be one
or two sentences. Clearly separate observations (what the data shows) from any
possible explanation (which must be flagged as a possible reason, not a certainty).
Do not invent any numbers, products, or regions that are not listed above. If the
context does not contain enough information for a particular kind of insight, skip it
rather than guessing."""

    return service.generate(prompt)
