import pandas as pd
import re

def normalize_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    norm_cfg = schema.get("normalization", {})
    string_norm = norm_cfg.get("string", {})
    numeric_norm = norm_cfg.get("numeric", {})

    df = df.copy()

    for col_name, col_spec in schema.get("columns", {}).items():
        if col_name not in df.columns:
            continue

        dtype = str(col_spec.get("dtype", ""))

        # ---------- STRING ----------
        if "string" in dtype:
            s = df[col_name].astype("string")

            missing_tokens = string_norm.get("missing_tokens", [])
            if missing_tokens:
                s = s.replace(missing_tokens, pd.NA)

            if string_norm.get("trim", False):
                s = s.str.strip()

            if string_norm.get("collapse_space", False):
                s = s.str.replace(r"\s+", " ", regex=True)

            if string_norm.get("normalize_separator", False):
                s = s.str.replace(",", ";")

            if string_norm.get("case_insensitive", False):
                s = s.str.lower()

            df[col_name] = s

        # ---------- NUMERIC ----------
        elif "float" in dtype or "int" in dtype:
            s = df[col_name].astype("string")

            missing_tokens = numeric_norm.get("missing_tokens", [])
            if missing_tokens:
                s = s.replace(missing_tokens, pd.NA)

            if numeric_norm.get("decimal_comma_to_dot", False):
                s = s.str.replace(",", ".", regex=False)

            for sep in numeric_norm.get("strip_thousands_separators", []):
                s = s.str.replace(sep, "", regex=False)

            df[col_name] = s

    return df

def normalize_only_data_rows(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()

    if "row_type" not in df.columns:
        # no row_type → normalize everything
        return normalize_dataframe(df, schema)

    df_meta = df[df["row_type"] == "meta"].copy()
    df_data = df[df["row_type"] == "data"].copy()

    # normalize only the data rows
    df_data_norm = normalize_dataframe(df_data, schema)

    # stitch back together
    df_out = pd.concat([df_meta, df_data_norm]).sort_index()
    return df_out
