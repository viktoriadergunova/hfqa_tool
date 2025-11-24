import pandas as pd
import re

def normalize_vocabulary_series(s: pd.Series) -> pd.Series:
    """
    Normalize bracketed vocabulary entries, including multi-valued cells.

    Examples (after global lowercase + trim):  # ADD MORE HERE IF FOUND
      "Drilling"                          -> "[drilling]"
      "[Drilling ]"                       -> "[drilling]"
      "[Drilling / Clustering]"           -> "[drilling-clustering]"
      "[Probing (offshore ocean)]"        -> "[probing-(offshore-ocean)]"
      "[x]; [y / z]"                      -> "[x];[y-z]"
    """
    s = s.astype("string")

    def _norm_value(val):
        if val is None or pd.isna(val):
            return pd.NA

        text = str(val).strip()
        if not text:
            return pd.NA

        # split multi entries on ';'
        parts = [p.strip() for p in text.split(";") if p.strip()]
        norm_parts = []

        for p in parts:
            p = p.lower().strip()

            # remove outer brackets if present
            if p.startswith("[") and p.endswith("]"):
                p = p[1:-1].strip()

            # normalize separators inside the token: space, comma, slash -> dash
            p = re.sub(r"[ ,/]+", "-", p)
            # collapse multiple dashes
            p = re.sub(r"-{2,}", "-", p)
            # remove spaces around parentheses and dashes
            p = re.sub(r"\s*\(\s*", "(", p)
            p = re.sub(r"\s*\)\s*", ")", p)
            p = re.sub(r"\s*-\s*", "-", p)

            # wrap back in brackets
            norm_parts.append(f"[{p}]")

        return ";".join(norm_parts) if norm_parts else pd.NA

    return s.map(_norm_value)

def normalize_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    norm_cfg = schema.get("normalization", {})
    string_norm = norm_cfg.get("string", {})
    numeric_norm = norm_cfg.get("numeric", {})

    df = df.copy()

    for col_name, col_spec in schema.get("columns", {}).items():
        if col_name not in df.columns:
            continue

        dtype = str(col_spec.get("dtype", ""))

        # Merge global and column-specific normalization rules
        col_norm = col_spec.get("normalization", {})
        effective_string_norm = string_norm.copy()
        effective_string_norm.update(col_norm)

        # ---------- STRING ----------
        if "string" in dtype:
            s = df[col_name].astype("string")

            missing_tokens = effective_string_norm.get("missing_tokens", [])
            if missing_tokens:
                s = s.replace(missing_tokens, pd.NA)

            if effective_string_norm.get("trim", False):
                s = s.str.strip()

            if effective_string_norm.get("collapse_space", False):
                s = s.str.replace(r"\s+", " ", regex=True)

            if effective_string_norm.get("normalize_separator", False):
                s = s.str.replace(",", ";")

            if effective_string_norm.get("case_insensitive", False):
                s = s.str.lower()

            if "replace_dash" in effective_string_norm:
                dash_char = effective_string_norm["replace_dash"]
                s = s.str.replace(dash_char, "-", regex=False)

            allowed = col_spec.get("allowed")
            needs_vocab_norm = bool(allowed) and any(
                "[" in str(a) or "]" in str(a) for a in allowed
            )

            if needs_vocab_norm:
                s = normalize_vocabulary_series(s)

            df[col_name] = s

        # ---------- NUMERIC ----------
        elif "float" in dtype or "int" in dtype:
            s = df[col_name].astype("string")

            missing_tokens = numeric_norm.get("missing_tokens", [])
            if missing_tokens:
                s = s.replace(missing_tokens, pd.NA)

            for sep in numeric_norm.get("strip_thousands_separators", []):
                s = s.str.replace(sep, "", regex=False)

            if numeric_norm.get("decimal_comma_to_dot", False):
                s = s.str.replace(",", ".", regex=False)

            df[col_name] = s

    return df



def normalize_only_data_rows(df: pd.DataFrame, schema: dict):
    df = df.copy()

    if "row_type" not in df.columns:
        df_data_norm = normalize_dataframe(df, schema)
        df_data_norm["row_type"] = "data"  # treat all as data
        return None, df_data_norm

    df_meta = df[df["row_type"] == "meta"].copy()
    df_data = df[df["row_type"] == "data"].copy()

    df_data_norm = normalize_dataframe(df_data, schema)
    df_data_norm["row_type"] = "data"
    df_meta["row_type"] = "meta"

    return df_meta, df_data_norm
