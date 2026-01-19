# utils/excel_writer.py

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype


def write_excel_with_vocab_check_comments(
    df_meta: pd.DataFrame | None,
    df_with_comments: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Writes an Excel file where meta rows (if provided) are placed on top,
    and data rows contain 'Validation_Comments'.
    """
    if df_meta is not None and len(df_meta) > 0:
        df_meta = df_meta.copy()
        df_meta["Validation_Comments"] = "META ROW"

        # Ensure meta has all columns present in df_with_comments
        for col in df_with_comments.columns:
            if col not in df_meta.columns:
                df_meta[col] = pd.NA

        # Align column order exactly
        df_meta = df_meta[df_with_comments.columns]

        # Keep stable ordering: meta first, then data
        df_final = pd.concat([df_meta, df_with_comments], ignore_index=True)
    else:
        df_final = df_with_comments.copy()

    df_final.to_excel(output_path, index=False, sheet_name="Validation Results")


def get_col_comment(schema_dict: dict, col_name: str) -> str:
    """Extract the 'comment' field for a column from the schema."""
    for section in ("columns", "core"):
        section_dict = schema_dict.get(section, {})
        if col_name in section_dict:
            return section_dict[col_name].get("comment", "")
    return ""


def generate_vocab_check_comments(
    df_checked: pd.DataFrame,
    schema: dict,
    first_data_index: int,  # kept for API compatibility; not required for comment generation
) -> pd.DataFrame:
    """
    Creates a new column 'Validation_Comments' with human-readable error messages
    for each row based on validation flags.
    """

    # Keep original columns only (not the flags)
    original_cols = [col for col in df_checked.columns if "__" not in col]
    df_result = df_checked[original_cols].copy()

    # Collect flag columns once (and ensure we treat NA as False)
    flag_cols = [c for c in df_checked.columns if "__" in c]

    # If some flag columns are not bool dtype, we still evaluate them safely via fillna(False) + astype(bool)
    flags_view = {}
    for c in flag_cols:
        s = df_checked[c]
        if not is_bool_dtype(s):
            # allow object/int/bool/boolean; treat truthy as True, NA as False
            flags_view[c] = s.fillna(False).astype(bool)
        else:
            # pandas BooleanDtype may contain NA -> fillna(False) for safe bool checks
            flags_view[c] = s.fillna(False)

    validation_comments: list[str] = []

    for idx in df_checked.index:
        row_comments: list[str] = []

        for col in flag_cols:
            # FIX 1: do NOT use "is True" / "is not True" (identity check).
            # Always use boolean evaluation.
            if not bool(flags_view[col].at[idx]):
                continue

            col_name, flag_type = col.split("__", 1)
            col_description = get_col_comment(schema, col_name)

            if flag_type == "missing":
                error_msg = f"[MISSING] {col_name} ({col_description})"
            elif flag_type == "out_of_range":
                error_msg = f"[RANGE ERROR] {col_name} ({col_description})"
            elif flag_type == "invalid":
                error_msg = f"[INVALID VALUE] {col_name} ({col_description})"
            elif flag_type.startswith("cond_"):
                error_msg = f"[CONDITIONAL ERROR] {col_name} ({col_description}): {flag_type}"
            else:
                error_msg = f"[ERROR] {col_name} ({col_description}): {flag_type}"

            row_comments.append(error_msg)

        validation_comments.append(" | ".join(row_comments) if row_comments else "OK")

    df_result["Validation_Comments"] = validation_comments
    return df_result
