"""
InsightAI — AI-Powered Business Data Analyst

Main Streamlit application entry point. Ties together data loading,
cleaning, profiling, analytics, charts, and the optional Gemini-powered
AI features (insights, chat, recommendations, executive report).
"""

from __future__ import annotations

import os
import traceback

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.data_loader import load_dataset
from modules.data_cleaner import clean_dataset
from modules.data_profiler import detect_column_types, compute_data_health
from modules.analytics import (
    detect_metric_columns,
    calculate_kpis,
    compute_numeric_summary,
    compute_categorical_summary,
    compute_correlation_matrix,
    compute_trend,
)
from modules.charts import build_dashboard_charts, build_correlation_heatmap
from modules.ai_service import get_ai_service
from modules.ai_insights import build_dataset_context, generate_ai_insights
from modules.ai_chat import answer_question
from modules.recommendations import generate_recommendations
from modules.report_generator import generate_executive_report
from utils.constants import MAX_PREVIEW_ROWS, AI_UNAVAILABLE_MESSAGE
from utils.helpers import format_number, format_compact_number, format_percentage

load_dotenv()

st.set_page_config(
    page_title="InsightAI — AI-Powered Business Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "raw_df": None,
    "cleaned_df": None,
    "file_name": None,
    "cleaning_log": [],
    "chat_history": [],
    "ai_insights_text": None,
    "ai_recommendations_text": None,
    "executive_report_text": None,
    "pending_sheet_selection": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_dataset_state():
    for key in ["raw_df", "cleaned_df", "file_name", "cleaning_log", "chat_history",
                "ai_insights_text", "ai_recommendations_text", "executive_report_text",
                "pending_sheet_selection"]:
        st.session_state[key] = DEFAULTS[key]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 InsightAI")
    st.caption("AI-Powered Business Data Analyst")
    st.divider()

    st.markdown("### Upload Dataset")
    uploaded_file = st.file_uploader(
        "Supported: CSV, XLSX",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    remove_duplicates = st.checkbox("Remove exact duplicate rows during cleaning", value=False)

    if uploaded_file is not None:
        is_new_file = st.session_state["file_name"] != uploaded_file.name
        if is_new_file:
            reset_dataset_state()

        if st.session_state["cleaned_df"] is None:
            sheet_choice = None
            if st.session_state["pending_sheet_selection"]:
                sheet_choice = st.session_state["pending_sheet_selection"]

            with st.spinner("Loading dataset..."):
                result = load_dataset(uploaded_file, sheet_name=sheet_choice)

            if result.warning_message == "MULTI_SHEET_SELECTION_REQUIRED":
                st.info("This Excel file has multiple sheets. Choose one to continue.")
                chosen = st.selectbox("Select a sheet", options=result.sheet_names, key="sheet_selector")
                if st.button("Load selected sheet"):
                    st.session_state["pending_sheet_selection"] = chosen
                    st.rerun()
            elif not result.success:
                st.error(result.error_message or "Could not load this file.")
            else:
                if result.warning_message:
                    st.warning(result.warning_message)
                try:
                    cleaning_result = clean_dataset(result.dataframe, remove_duplicates=remove_duplicates)
                    st.session_state["raw_df"] = result.dataframe
                    st.session_state["cleaned_df"] = cleaning_result.dataframe
                    st.session_state["cleaning_log"] = cleaning_result.operations_log
                    st.session_state["file_name"] = uploaded_file.name
                    st.success("Dataset loaded successfully.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"An error occurred while processing the dataset: {exc}")

    st.divider()

    ai_service = get_ai_service()
    st.markdown("### AI Status")
    if ai_service.is_available:
        st.success("Connected (Gemini)")
    else:
        st.warning("Not connected")
        with st.expander("Why?"):
            st.write(ai_service.unavailable_reason)
            st.caption(
                "Add a free Gemini API key to your .env file as GEMINI_API_KEY to enable "
                "AI Insights, Ask Your Data, Recommendations, and AI report narrative."
            )

    if st.session_state["cleaned_df"] is not None:
        st.divider()
        st.markdown("### Dataset Info")
        df = st.session_state["cleaned_df"]
        st.write(f"**File:** {st.session_state['file_name']}")
        st.write(f"**Rows:** {len(df):,}")
        st.write(f"**Columns:** {len(df.columns)}")
        if st.button("Clear dataset"):
            reset_dataset_state()
            st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("InsightAI")
st.caption("Turn spreadsheets into business insights using AI.")

if st.session_state["cleaned_df"] is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")
    st.markdown(
        "Don't have a dataset handy? A sample sales dataset is included at "
        "`data/sample_sales.csv` — upload it to try the app."
    )
    st.stop()

df = st.session_state["cleaned_df"]

# --- Guard: dataset with zero usable columns after cleaning ---
if df.shape[1] == 0 or df.shape[0] == 0:
    st.error("The dataset has no usable rows or columns after cleaning.")
    st.stop()

try:
    column_profile = detect_column_types(df)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not analyze column types: {exc}")
    st.stop()

metric_columns = detect_metric_columns(column_profile.numeric_columns)
data_health = compute_data_health(df)

tab_overview, tab_health, tab_kpi, tab_charts, tab_insights, tab_chat, tab_recs, tab_report = st.tabs(
    ["Overview", "Data Health", "Business KPIs", "Visual Analytics",
     "AI Insights", "Ask Your Data", "AI Recommendations", "Executive Report"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("File", st.session_state["file_name"])
    col2.metric("Rows", f"{len(df):,}")
    col3.metric("Columns", f"{len(df.columns)}")

    st.markdown("#### Preview")
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True)

    st.markdown("#### Detected Column Types")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown("**Numeric**")
    col_a.write(column_profile.numeric_columns or "—")
    col_b.markdown("**Categorical**")
    col_b.write(column_profile.categorical_columns or "—")
    col_c.markdown("**Date**")
    col_c.write(column_profile.date_columns or "—")
    col_d.markdown("**Identifier**")
    col_d.write(column_profile.identifier_columns or "—")

    if st.session_state["cleaning_log"]:
        with st.expander("Cleaning operations performed"):
            for line in st.session_state["cleaning_log"]:
                st.write(f"- {line}")

# ---------------------------------------------------------------------------
# Data Health
# ---------------------------------------------------------------------------
with tab_health:
    st.subheader("Data Health")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Data Quality Score", f"{data_health.quality_score}/100")
    col2.metric("Missing Values", f"{data_health.missing_cells:,}", f"{data_health.missing_percentage}%")
    col3.metric("Duplicate Rows", f"{data_health.duplicate_rows:,}", f"{data_health.duplicate_percentage}%")
    col4.metric("Total Cells", f"{len(df) * len(df.columns):,}")

    st.caption(
        "Quality score = 100 − (missing-value ratio × 60) − (duplicate-row ratio × 40), floored at 0. "
        "This is a deterministic formula, not an AI-generated score."
    )

    st.markdown("#### Column Breakdown")
    breakdown_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing Values": [data_health.missing_by_column.get(c, 0) for c in df.columns],
        "Missing %": [
            round((data_health.missing_by_column.get(c, 0) / max(len(df), 1)) * 100, 1) for c in df.columns
        ],
        "Unique Values": [df[c].nunique(dropna=True) for c in df.columns],
    })
    st.dataframe(breakdown_df, use_container_width=True)

    st.markdown("#### Column Type Summary")
    st.write(f"- Numerical columns: {len(column_profile.numeric_columns)}")
    st.write(f"- Categorical columns: {len(column_profile.categorical_columns)}")
    st.write(f"- Date columns: {len(column_profile.date_columns)}")
    st.write(f"- Boolean columns: {len(column_profile.boolean_columns)}")
    st.write(f"- Identifier columns: {len(column_profile.identifier_columns)}")

# ---------------------------------------------------------------------------
# Business KPIs
# ---------------------------------------------------------------------------
with tab_kpi:
    st.subheader("Business KPIs")

    kpis = calculate_kpis(df, metric_columns)
    if not kpis:
        st.info("No standard business KPIs could be calculated from this dataset's columns.")
    else:
        cols = st.columns(min(len(kpis), 4))
        for i, kpi in enumerate(kpis):
            target_col = cols[i % len(cols)]
            if kpi.value is None:
                continue
            if kpi.is_percentage:
                target_col.metric(kpi.label, format_percentage(kpi.value))
            elif kpi.is_currency_like:
                target_col.metric(kpi.label, format_compact_number(kpi.value))
            else:
                target_col.metric(kpi.label, format_number(kpi.value))

    st.markdown("#### Descriptive Statistics")
    numeric_summary = compute_numeric_summary(df, column_profile.numeric_columns)
    if numeric_summary.empty:
        st.info("No numeric columns available for descriptive statistics.")
    else:
        st.dataframe(numeric_summary, use_container_width=True)

    st.markdown("#### Category Breakdown")
    cat_summary = compute_categorical_summary(df, column_profile.categorical_columns)
    if not cat_summary:
        st.info("No categorical columns available.")
    else:
        cat_cols = st.columns(min(len(cat_summary), 3))
        for i, (col_name, counts) in enumerate(cat_summary.items()):
            with cat_cols[i % len(cat_cols)]:
                st.markdown(f"**{col_name}**")
                st.dataframe(counts.rename("Count"), use_container_width=True)

    if column_profile.date_columns:
        st.markdown("#### Trend Snapshot")
        primary_metric = metric_columns.revenue or metric_columns.profit or (
            column_profile.numeric_columns[0] if column_profile.numeric_columns else None
        )
        if primary_metric:
            trend = compute_trend(df, column_profile.date_columns[0], primary_metric)
            if trend and trend.percent_change_value is not None:
                direction = "increase" if trend.percent_change_value >= 0 else "decrease"
                st.write(
                    f"**{primary_metric}** ({trend.period_label}) — most recent period "
                    f"({trend.current_period}): {format_number(trend.current_value, 2)}, "
                    f"previous period ({trend.previous_period}): {format_number(trend.previous_value, 2)} "
                    f"→ a {abs(trend.percent_change_value):.1f}% {direction}."
                )
            else:
                st.info("Not enough time periods in this dataset to compute a period-over-period trend.")

# ---------------------------------------------------------------------------
# Visual Analytics
# ---------------------------------------------------------------------------
with tab_charts:
    st.subheader("Visual Analytics")
    try:
        figures = build_dashboard_charts(
            df, metric_columns, column_profile.date_columns,
            column_profile.categorical_columns, column_profile.numeric_columns,
        )
    except Exception as exc:  # noqa: BLE001
        figures = []
        st.error(f"Could not generate charts: {exc}")

    if not figures:
        st.info("Not enough suitable data to generate charts for this dataset.")
    else:
        for i in range(0, len(figures), 2):
            row_figs = figures[i:i + 2]
            cols = st.columns(len(row_figs))
            for col, fig in zip(cols, row_figs):
                col.plotly_chart(fig, use_container_width=True)

    if len(column_profile.numeric_columns) >= 2:
        st.markdown("#### Correlation Heatmap")
        corr_matrix = compute_correlation_matrix(df, column_profile.numeric_columns)
        heatmap = build_correlation_heatmap(corr_matrix)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)

# ---------------------------------------------------------------------------
# Build dataset context once, reused by AI Insights / Chat / Recommendations / Report
# ---------------------------------------------------------------------------
dataset_context = build_dataset_context(
    df, st.session_state["file_name"], metric_columns,
    column_profile.date_columns, column_profile.categorical_columns, column_profile.numeric_columns,
)

# ---------------------------------------------------------------------------
# AI Insights
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("AI Insights")

    if not ai_service.is_available:
        st.warning(AI_UNAVAILABLE_MESSAGE)
    else:
        if st.button("Generate Insights", key="btn_insights"):
            with st.spinner("Analyzing dataset..."):
                response = generate_ai_insights(dataset_context, ai_service)
            if response.success:
                st.session_state["ai_insights_text"] = response.text
            else:
                st.error(response.error_message or AI_UNAVAILABLE_MESSAGE)

        if st.session_state["ai_insights_text"]:
            st.markdown(st.session_state["ai_insights_text"])
        else:
            st.caption("Click 'Generate Insights' to have Gemini interpret the calculated metrics above.")

# ---------------------------------------------------------------------------
# Ask Your Data
# ---------------------------------------------------------------------------
with tab_chat:
    st.subheader("Ask Your Data")

    if not ai_service.is_available:
        st.warning(AI_UNAVAILABLE_MESSAGE)
    else:
        for question, answer in st.session_state["chat_history"]:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)

        col_a, col_b = st.columns([5, 1])
        with col_b:
            if st.button("Clear chat"):
                st.session_state["chat_history"] = []
                st.rerun()

        user_question = st.chat_input("Ask a question about your data...")
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            with st.spinner("Thinking..."):
                response = answer_question(
                    user_question, dataset_context, st.session_state["chat_history"], ai_service
                )
            answer_text = response.text if response.success else (
                response.error_message or AI_UNAVAILABLE_MESSAGE
            )
            with st.chat_message("assistant"):
                st.write(answer_text)
            st.session_state["chat_history"].append((user_question, answer_text))

# ---------------------------------------------------------------------------
# AI Recommendations
# ---------------------------------------------------------------------------
with tab_recs:
    st.subheader("AI Recommendations")

    if not ai_service.is_available:
        st.warning(AI_UNAVAILABLE_MESSAGE)
    else:
        if st.button("Generate Recommendations", key="btn_recs"):
            with st.spinner("Generating recommendations..."):
                response = generate_recommendations(dataset_context, ai_service)
            if response.success:
                st.session_state["ai_recommendations_text"] = response.text
            else:
                st.error(response.error_message or AI_UNAVAILABLE_MESSAGE)

        if st.session_state["ai_recommendations_text"]:
            st.markdown(st.session_state["ai_recommendations_text"])
        else:
            st.caption("Click 'Generate Recommendations' for actionable, categorized suggestions.")

# ---------------------------------------------------------------------------
# Executive Report
# ---------------------------------------------------------------------------
with tab_report:
    st.subheader("Executive Report")

    if st.button("Generate Report", key="btn_report"):
        with st.spinner("Compiling report..."):
            try:
                report_text = generate_executive_report(
                    df, st.session_state["file_name"], metric_columns, dataset_context,
                    st.session_state["ai_insights_text"], st.session_state["ai_recommendations_text"],
                    ai_service,
                )
                st.session_state["executive_report_text"] = report_text
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not generate report: {exc}")

    if st.session_state["executive_report_text"]:
        st.markdown(st.session_state["executive_report_text"])
        st.download_button(
            "Download Report (.md)",
            data=st.session_state["executive_report_text"],
            file_name="insightai_executive_report.md",
            mime="text/markdown",
        )
    else:
        st.caption("Click 'Generate Report' to compile an executive summary from all calculated metrics "
                   "(and AI narrative, if available).")

st.divider()
st.caption(
    "InsightAI calculates all metrics locally with Pandas. When AI features are used, a "
    "compact summary of calculated metrics (not your raw data) is sent to the configured "
    "Gemini API."
)
