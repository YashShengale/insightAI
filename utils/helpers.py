"""
Generic, dependency-light helper functions shared across modules.
"""

from __future__ import annotations

import re
from typing import Optional


def format_number(value: Optional[float], decimals: int = 0) -> str:
    """Format a number with thousands separators. Returns 'N/A' for None/NaN."""
    if value is None:
        return "N/A"
    try:
        if value != value:  # NaN check without importing numpy/pandas here
            return "N/A"
    except TypeError:
        return "N/A"

    if decimals == 0:
        return f"{value:,.0f}"
    return f"{value:,.{decimals}f}"


def format_compact_number(value: Optional[float]) -> str:
    """Format large numbers compactly, e.g. 12400000 -> '12.4M'."""
    if value is None:
        return "N/A"
    try:
        if value != value:
            return "N/A"
    except TypeError:
        return "N/A"

    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    if abs_value >= 1_000_000_000:
        return f"{sign}{abs_value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"{sign}{abs_value / 1_000_000:.2f}M"
    if abs_value >= 1_000:
        return f"{sign}{abs_value / 1_000:.1f}K"
    return f"{sign}{abs_value:,.0f}"


def format_percentage(value: Optional[float], decimals: int = 1) -> str:
    """Format a percentage value. Expects value already in percent form (e.g. 25.3, not 0.253)."""
    if value is None:
        return "N/A"
    try:
        if value != value:
            return "N/A"
    except TypeError:
        return "N/A"
    return f"{value:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float) -> Optional[float]:
    """Divide safely, returning None instead of raising on division by zero."""
    if denominator in (0, None):
        return None
    try:
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return None


def percent_change(current: float, previous: float) -> Optional[float]:
    """Percentage change from `previous` to `current`. Returns None if previous is 0/None."""
    if previous in (0, None):
        return None
    try:
        return ((current - previous) / abs(previous)) * 100
    except (TypeError, ZeroDivisionError):
        return None


def normalize_column_name(name: str) -> str:
    """Lowercase, strip, and collapse whitespace/underscores for keyword matching."""
    name = str(name).strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name


def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate long text safely for prompt construction, with a marker."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."
