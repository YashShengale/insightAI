"""
data_loader.py

Handles reading uploaded CSV/Excel files into Pandas DataFrames safely.
Never raises an uncaught exception to the caller — always returns a
LoadResult describing success/failure so the UI layer can show a clean
message instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import pandas as pd

from utils.constants import SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_MB

# Encodings to try in order when a CSV isn't valid UTF-8.
CANDIDATE_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]


@dataclass
class LoadResult:
    success: bool
    dataframe: Optional[pd.DataFrame] = None
    file_name: str = ""
    sheet_names: List[str] = field(default_factory=list)
    selected_sheet: Optional[str] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


def get_file_extension(file_name: str) -> str:
    idx = file_name.rfind(".")
    if idx == -1:
        return ""
    return file_name[idx:].lower()


def get_excel_sheet_names(uploaded_file) -> List[str]:
    """Return sheet names for an Excel file without fully loading it."""
    try:
        uploaded_file.seek(0)
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
        return xls.sheet_names
    except Exception:
        return []


def load_csv(uploaded_file, file_name: str) -> LoadResult:
    """Load a CSV file, trying multiple encodings and delimiters."""
    last_error = None
    for encoding in CANDIDATE_ENCODINGS:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine="python")
            if df.shape[1] == 0:
                raise ValueError("No columns detected in file.")
            return LoadResult(success=True, dataframe=df, file_name=file_name)
        except Exception as exc:  # noqa: BLE001 - we deliberately catch broadly here
            last_error = exc
            continue

    return LoadResult(
        success=False,
        file_name=file_name,
        error_message=(
            "Could not read this CSV file. It may be corrupted, empty, or use an "
            f"unsupported text encoding. Technical detail: {last_error}"
        ),
    )


def load_excel(uploaded_file, file_name: str, sheet_name: Optional[str] = None) -> LoadResult:
    """Load an Excel file. If sheet_name is None and multiple sheets exist,
    returns the sheet list so the caller can prompt the user to choose."""
    sheet_names = get_excel_sheet_names(uploaded_file)

    if not sheet_names:
        return LoadResult(
            success=False,
            file_name=file_name,
            error_message=(
                "Could not read this Excel file. It may be corrupted, password-protected, "
                "or not a valid .xlsx/.xls file."
            ),
        )

    if sheet_name is None and len(sheet_names) > 1:
        return LoadResult(
            success=False,
            file_name=file_name,
            sheet_names=sheet_names,
            error_message=None,
            warning_message="MULTI_SHEET_SELECTION_REQUIRED",
        )

    target_sheet = sheet_name or sheet_names[0]

    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=target_sheet, engine="openpyxl")
        if df.shape[1] == 0:
            raise ValueError("No columns detected in sheet.")
        return LoadResult(
            success=True,
            dataframe=df,
            file_name=file_name,
            sheet_names=sheet_names,
            selected_sheet=target_sheet,
        )
    except Exception as exc:  # noqa: BLE001
        return LoadResult(
            success=False,
            file_name=file_name,
            sheet_names=sheet_names,
            error_message=f"Could not read sheet '{target_sheet}'. Technical detail: {exc}",
        )


def load_dataset(uploaded_file, sheet_name: Optional[str] = None) -> LoadResult:
    """Main entry point: detects file type and dispatches to the right loader."""
    if uploaded_file is None:
        return LoadResult(success=False, error_message="No file was uploaded.")

    file_name = uploaded_file.name
    extension = get_file_extension(file_name)

    if extension not in SUPPORTED_EXTENSIONS:
        return LoadResult(
            success=False,
            file_name=file_name,
            error_message=(
                f"Unsupported file type '{extension}'. Please upload a CSV or Excel "
                "(.xlsx / .xls) file."
            ),
        )

    # Soft file-size check (Streamlit's UploadedFile exposes .size in bytes)
    warning_message = None
    try:
        size_mb = uploaded_file.size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            warning_message = (
                f"This file is {size_mb:.1f} MB, which is quite large. "
                "Processing may be slow."
            )
    except Exception:
        pass

    if extension == ".csv":
        result = load_csv(uploaded_file, file_name)
    else:
        result = load_excel(uploaded_file, file_name, sheet_name=sheet_name)

    if result.success and warning_message:
        result.warning_message = warning_message

    # Guard against completely empty datasets
    if result.success and result.dataframe is not None and result.dataframe.shape[0] == 0:
        result.success = False
        result.error_message = "The uploaded file contains no data rows."

    return result
