import pandas as pd

def _iter_schema_specs(schema: dict):
    for section in ("columns", "core"):
        for col_name, col_spec in schema.get(section, {}).items():
            yield col_name, col_spec


def add_mandatory_flags(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()

    if "row_type" in df.columns:
        data_mask = df["row_type"] == "data"
    else:
        data_mask = pd.Series(True, index=df.index)

    for col_name, col_spec in _iter_schema_specs(schema):
        if col_name not in df.columns:
            continue

        obligation = str(col_spec.get("obligation", "")).strip().upper()
        if obligation != "M":
            continue

        cond = data_mask & df[col_name].isna()
        df[f"{col_name}__missing"] = cond

    return df


def add_range_and_allowed_flags(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()

    if "row_type" in df.columns:
        data_mask = df["row_type"] == "data"
    else:
        data_mask = pd.Series(True, index=df.index)

    norm_cfg = schema.get("normalization", {})
    global_string_norm = norm_cfg.get("string", {})

    for col_name, col_spec in _iter_schema_specs(schema):
        if col_name not in df.columns:
            continue

        s = df[col_name]
        dtype_str = str(col_spec.get("dtype", "")).strip().lower()

        # ---------- DATATYPE CHECK ----------
        expected_type = None
        if "float" in dtype_str:
            expected_type = float
        elif "int" in dtype_str:
            expected_type = int

        if expected_type:
            def is_invalid_type(x):
                if pd.isna(x):  # Allow NaN
                    return False
                try:
                    # Enforce strict type matching
                    if expected_type == int and isinstance(x, float) and not x.is_integer():
                        return True
                    expected_type(x)
                    return False
                except (ValueError, TypeError):
                    return True

            cond = data_mask & s.apply(is_invalid_type)
            df[f"{col_name}__invalid_dtype"] = cond

        # ---------- RANGE CHECK ----------
        value_range = col_spec.get("range")
        if value_range is not None and expected_type:
            lo, hi = value_range
            numeric_series = pd.to_numeric(s, errors="coerce")
            cond = data_mask & numeric_series.notna() & ((numeric_series < lo) | (numeric_series > hi))
            df[f"{col_name}__out_of_range"] = cond

        # ---------- ALLOWED-VALUE CHECK ----------
        allowed_raw = col_spec.get("allowed")
        if allowed_raw is not None:
            allowed_set = set(allowed_raw)
            s_str = s.astype("string").str.strip().str.lower()
            multi_choice = bool(col_spec.get("multi_choice", False))
            sep = col_spec.get("separator", ";")

            if multi_choice:
                def invalid_multi(val):
                    if pd.isna(val):
                        return False
                    parts = [p.strip() for p in str(val).split(sep) if p.strip()]
                    return any(p not in allowed_set for p in parts)

                cond = data_mask & s_str.notna() & s_str.apply(invalid_multi)
            else:
                cond = data_mask & s_str.notna() & ~s_str.isin(allowed_set)

            df[f"{col_name}__invalid"] = cond

    return df