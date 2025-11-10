import pandas as pd
from etl.normalization import normalize_vocabulary_series


def _iter_schema_specs(schema: dict):
    specs = {}
    specs.update(schema.get("core", {}))
    specs.update(schema.get("columns", {}))
    return specs.items()

import pandas as pd

def add_mandatory_flags(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Add <col>__missing flag for all columns with obligation: 'M'
    True where value is missing (NaN / <NA>) in data rows.
    """
    df = df.copy()

    if "row_type" in df.columns:
        data_mask = df["row_type"] == "data"
    else:
        data_mask = pd.Series(True, index=df.index)

    # only schema["columns"], not "core"
    for col_name, col_spec in schema.get("columns", {}).items():
        if col_name not in df.columns:
            continue

        obligation = str(col_spec.get("obligation", "")).strip().upper()
        if obligation != "M":
            continue  # skip R, O, -, or missing

        cond = data_mask & df[col_name].isna()
        df[f"{col_name}__missing"] = cond

    return df



def add_range_and_allowed_flags(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Attach boolean flag columns for numeric ranges and allowed-values checks.

    - <col>__out_of_range  for numeric range violations
    - <col>__invalid       for allowed-values violations
    """
    df = df.copy()

    if "row_type" in df.columns:
        data_mask = df["row_type"] == "data"
    else:
        data_mask = pd.Series(True, index=df.index)

    for col_name, col_spec in _iter_schema_specs(schema):
        if col_name not in df.columns:
            continue

        dtype_str = str(col_spec.get("dtype", "")).strip().lower()

        # ---------- RANGE (numeric) ----------
        value_range = col_spec.get("range")
        if value_range is not None and ("float" in dtype_str or "int" in dtype_str):
            lo, hi = value_range
            s = df[col_name]

            cond = (
                data_mask
                & s.notna()
                & ((s < lo) | (s > hi))
            )
            df[f"{col_name}__out_of_range"] = cond

        # ---------- ALLOWED (vocab) ----------
        allowed_raw = col_spec.get("allowed")
        if allowed_raw is not None:
            # normalize allowed list using the same vocabulary logic as data
            allowed_series = pd.Series(allowed_raw, dtype="string")
            allowed_norm = normalize_vocabulary_series(allowed_series)
            allowed_set = {str(a).strip().lower() for a in allowed_norm.dropna()}

            # normalize data side (already normalized, but this is safe)
            s = df[col_name].astype("string").str.strip().str.lower()

            # handle multi-choice columns correctly
            multi_choice = bool(col_spec.get("multi_choice", False))
            sep = col_spec.get("separator", ";")

            if multi_choice:
                def invalid_multi(val):
                    if pd.isna(val):
                        return False
                    parts = [p.strip() for p in str(val).split(sep) if p.strip()]
                    # invalid if any part is not in allowed_set
                    return any(p not in allowed_set for p in parts)

                cond = data_mask & s.notna() & s.apply(invalid_multi)
            else:
                # single-valued: whole cell must be in allowed_set
                cond = data_mask & s.notna() & ~s.isin(allowed_set)

            df[f"{col_name}__invalid"] = cond

    return df
