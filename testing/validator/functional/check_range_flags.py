import yaml
import pandas as pd

from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows
from etl.typecasting import apply_schema_types
from validator.functional_validator import add_range_and_allowed_flags


INPUT_XLSX = "testing/range_test.xlsx"
SCHEMA_PATH = "hf_schema.yaml"


def main():
    # --- Load schema ---
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    columns_spec = schema.get("columns", {})

    # determine which columns have ranges
    range_cols = []
    for col_name, col_spec in columns_spec.items():
        dtype = str(col_spec.get("dtype", "")).lower()
        if "range" in col_spec and ("float" in dtype or "int" in dtype):
            range_cols.append(col_name)

    print(f"Range columns in schema ({len(range_cols)}):")
    print("  " + ", ".join(range_cols))

    # --- Excel -> Parquet + row_type ---
    df_raw = excel_to_parquet(
        excel_path=INPUT_XLSX,
        parquet_path="testing/range_test.raw.parquet",
        sheet_name=0,
        meta_rows=0,  # no meta rows in this test file
    )

    # --- Normalize only data rows ---
    df_meta, df_data_norm = normalize_only_data_rows(df_raw, schema)
    df_typed = apply_schema_types(df_data_norm, schema)

    # --- Add range flags (and allowed, but we ignore allowed here) ---
    df_checked = add_range_and_allowed_flags(df_typed, schema)

    if "__target_col" not in df_checked.columns or "__case" not in df_checked.columns:
        print("\nERROR: '__target_col' or '__case' column missing in test file.")
        return

    tests = []

    # iterate over each row (one test case per row)
    for idx, row in df_checked.iterrows():
        target_col = row["__target_col"]
        case = row["__case"]

        # ignore rows without target_col (shouldn't happen)
        if pd.isna(target_col) or target_col not in range_cols:
            continue

        flag_col = f"{target_col}__out_of_range"
        # if for some reason the flag column does not exist, treat as False
        flag_val = bool(row[flag_col]) if flag_col in df_checked.columns else False

        # build test name
        name = f"[range] {target_col} case '{case}'"

        # expected behaviour
        if case in ("too_low", "too_high"):
            expected = True
            if flag_val == expected:
                tests.append((name, True, ""))
            else:
                msg = f"Expected {flag_col} = True for case '{case}', but got False."
                tests.append((name, False, msg))

        elif case in ("at_lower", "at_upper"):
            expected = False
            if flag_val == expected:
                tests.append((name, True, ""))
            else:
                msg = f"Expected {flag_col} = False at boundary case '{case}', but got True."
                tests.append((name, False, msg))

        elif case == "missing":
            # range must NOT flag missing values; out_of_range must be False
            expected = False
            if flag_val == expected:
                tests.append((name, True, ""))
            else:
                msg = f"Expected {flag_col} = False for missing value, but got True."
                tests.append((name, False, msg))

        else:
            # unknown case label
            msg = f"Unknown case label '{case}' for column '{target_col}'."
            tests.append((name, False, msg))

    # --- Summary output with bar and counters ---

    total = len(tests)
    passed = sum(1 for _, ok, _ in tests if ok)
    failed = total - passed

    print("\nRange test summary")
    if total == 0:
        print("No range tests were executed.")
        return

    bar_len = 40
    filled = int(bar_len * passed / total)
    bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

    print(f"{bar}  {passed}/{total} test cases passed, {failed} failed.")

    print("\nDetailed range test cases:")
    for i, (name, ok, msg) in enumerate(tests, start=1):
        status = "PASS" if ok else "FAIL"
        print(f"{i:3d}) {status} - {name}")
        if not ok and msg:
            print(f"     {msg}")

    if failed == 0:
        print("\nAll range tests passed.")
    else:
        print("\nSome range tests failed.")


if __name__ == "__main__":
    main()
