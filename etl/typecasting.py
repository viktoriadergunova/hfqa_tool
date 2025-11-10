import pandas as pd


def _iter_schema_specs(schema: dict):
    """Combine 'core' and 'columns' specs into one iterator."""
    specs = {}
    specs.update(schema.get("core", {}))
    specs.update(schema.get("columns", {}))
    return specs.items()


def apply_schema_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Cast dataframe columns to target dtypes defined in schema (data rows only)."""
    df = df.copy()

    for col_name, col_spec in _iter_schema_specs(schema):
        if col_name not in df.columns:
            continue

        expected_dtype = str(col_spec.get("dtype", "")).strip().lower()
        if not expected_dtype:
            continue

        # --- LIST TYPES ---
        if "list[" in expected_dtype:
            # Keep as string for now; split later if needed
            df[col_name] = df[col_name].astype("string[pyarrow]")
            continue

        # --- STRING ---
        if "string" in expected_dtype:
            df[col_name] = df[col_name].astype("string[pyarrow]")
            continue

        # --- FLOAT ---
        if "float" in expected_dtype:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("float64")
            continue

        # --- INT ---
        if "int" in expected_dtype:
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")
            continue

        # --- DATETIME ---
        if "timestamp" in expected_dtype or "datetime" in expected_dtype:
            df[col_name] = pd.to_datetime(df[col_name], errors="coerce", dayfirst=True)
            continue

        # --- FALLBACK: STRING ---
        df[col_name] = df[col_name].astype("string[pyarrow]")

    return df

def verify_schema_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Return a DataFrame summarizing dtype verification."""
    rows = []
    for col_name, col_spec in _iter_schema_specs(schema):
        if col_name not in df.columns:
            rows.append((col_name, col_spec.get("dtype", ""), "MISSING"))
            continue

        expected = str(col_spec.get("dtype", "")).strip().lower()
        actual = str(df[col_name].dtype).lower()

        match = (
            ("float" in expected and "float" in actual)
            or ("int" in expected and "int" in actual)
            or ("string" in expected and "string" in actual)
            or (("timestamp" in expected or "datetime" in expected) and "datetime" in actual)
            or ("list[" in expected and "string" in actual)
        )

        status = "OK" if match else "WARN"
        rows.append((col_name, expected, actual, status))

    result = pd.DataFrame(rows, columns=["column", "expected", "actual", "status"])
    return result

