import yaml
import pandas as pd

from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows
from etl.typecasting import apply_schema_types
from vocab_check.functional_validator import add_range_and_allowed_flags


INPUT_XLSX = "testing/run/testing_files/allowed_test.xlsx"
SCHEMA_PATH = "hf_schema.yaml"


def main():
    # --- Load schema ---
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    columns_spec = schema.get("columns", {})

    # determine columns with allowed values
    allowed_cols = []
    multi_choice_map = {}
    for col_name, col_spec in columns_spec.items():
        allowed = col_spec.get("allowed")
        if allowed is not None:
            allowed_cols.append(col_name)
            multi_choice_map[col_name] = bool(col_spec.get("multi_choice", False))

    print(f"Allowed-value columns in schema ({len(allowed_cols)}):")
    print("  " + ", ".join(allowed_cols))

    # --- Excel -> Parquet + row_type ---
    df_raw = excel_to_parquet(
        excel_path=INPUT_XLSX,
        parquet_path="testing/run/testing_files/allowed_test.raw.parquet",
        sheet_name=0,
        meta_rows=0,  # no meta rows in this test file
    )

    # --- Normalize only data rows ---
    df_meta, df_data_norm = normalize_only_data_rows(df_raw, schema)
    df_typed = apply_schema_types(df_data_norm, schema)

    # --- Add allowed flags (and range, but range is ignored here) ---
    df_checked = add_range_and_allowed_flags(df_typed, schema)

    if "__target_col" not in df_checked.columns or "__case" not in df_checked.columns:
        print("\nERROR: '__target_col' or '__case' column missing in test file.")
        return

    tests = []

    for idx, row in df_checked.iterrows():
        target_col = row["__target_col"]
        case = row["__case"]

        if pd.isna(target_col) or target_col not in allowed_cols:
            continue

        invalid_col = f"{target_col}__invalid"
        flag_val = bool(row[invalid_col]) if invalid_col in df_checked.columns else False

        name = f"[allowed] {target_col} case '{case}'"

        # Expected behaviour per case
        if case in ("valid_single", "valid_multi"):
            expected = False
            if flag_val == expected:
                tests.append((name, True, ""))
            else:
                msg = f"Expected {invalid_col} = False for case '{case}', but got True."
                tests.append((name, False, msg))

        elif case == "invalid_single":
            expected = True
            if flag_val == expected:
                tests.append((name, True, ""))
            else:
                msg = f"Expected {invalid_col} = True for case '{case}', but got False."
                tests.append((name, False, msg))

        elif case == "invalid_one_bad":
            expected = True
            if flag_val == expected:
                tests.append((name, True, ""))
            else:
                msg = f"Expected {invalid_col} = True for case '{case}' (one bad entry), but got False."
                tests.append((name, False, msg))

        elif case == "missing":
            # Missing values should NOT be flagged as invalid
            expected = False
            if flag_val == expected:
                tests.append((name, True, ""))
            else:
                msg = f"Expected {invalid_col} = False for missing value, but got True."
                tests.append((name, False, msg))

        else:
            msg = f"Unknown case label '{case}' for column '{target_col}'."
            tests.append((name, False, msg))

    # --- Summary output with bar and counters ---

    total = len(tests)
    passed = sum(1 for _, ok, _ in tests if ok)
    failed = total - passed

    print("\nAllowed-value test summary")
    if total == 0:
        print("No allowed-value tests were executed.")
        return

    bar_len = 40
    filled = int(bar_len * passed / total)
    bar = "[" + "#" * filled + "-" * (bar_len - filled) + "]"

    print(f"{bar}  {passed}/{total} test cases passed, {failed} failed.")

    print("\nDetailed allowed-value test cases:")
    for i, (name, ok, msg) in enumerate(tests, start=1):
        status = "PASS" if ok else "FAIL"
        print(f"{i:3d}) {status} - {name}")
        if not ok and msg:
            print(f"     {msg}")

    if failed == 0:
        print("\nAll allowed-value tests passed.")
    else:
        print("\nSome allowed-value tests failed.")


if __name__ == "__main__":
    main()
