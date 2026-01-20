import yaml
import pandas as pd

from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows, normalize_vocabulary_series
from etl.typecasting import apply_schema_types

from quality_score.apply_m_quality_score import calculate_m_score

INPUT_XLSX = "testing/run/testing_files/m_score_tests.xlsx"
SCHEMA_PATH = "hf_schema.yaml"
QC_SCHEMA_PATH = "quality_score_schema.yaml"


def main():
    # 1) Load Schemas
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    with open(QC_SCHEMA_PATH, "r", encoding="utf-8") as f:
        qc_schema = yaml.safe_load(f)

    # 2) Pipeline
    df_raw = excel_to_parquet(INPUT_XLSX, "temp_m.parquet", sheet_name=0, meta_rows=0)
    _, df_data_norm = normalize_only_data_rows(df_raw, schema)
    df_typed = apply_schema_types(df_data_norm, schema)

    # 3) Apply M-score logic (parent inheritance is handled inside calculate_m_score)
    df_typed["computed_M"] = calculate_m_score(df_typed, qc_schema)

    # 4) Schema-driven site/role columns
    m_calc = qc_schema["m_score"]["calculation"]
    site_col = m_calc["site_name_col"]
    rel_col = m_calc["relevance_col"]
    ROLE_CHILD = m_calc.get("role_child", "[yes]")
    ROLE_PARENT = m_calc.get("role_parent", "[no]")

    # Normalize relevance for robust matching
    relevance = normalize_vocabulary_series(df_typed[rel_col])

    # 5) Verification (expects: expected_M + test_case in the test file)
    tests = []
    for idx, row in df_typed.iterrows():
        expected = row.get("expected_M")
        if pd.isna(expected):
            continue

        expected = str(expected)
        actual = str(row["computed_M"])

        case = row.get("test_case", "unknown")
        site = row.get(site_col)

        role = "Parent" if relevance.loc[idx] == ROLE_PARENT else "Child"

        name = f"[m_score] {site} ({role}) | Case: {case}"
        ok = (actual == expected)
        tests.append((name, ok, "" if ok else f"Expected {expected}, got {actual}"))

    # 6) Report
    total = len(tests)
    passed = sum(1 for _, ok, _ in tests if ok)

    print("\nIHFC M-Score Quality Test Summary")
    if total == 0:
        print("No tests found (no expected results present).")
        return

    bar_len = 40
    fill = int(bar_len * passed / total)
    bar = "[" + "#" * fill + "-" * (bar_len - fill) + "]"
    print(f"{bar}  {passed}/{total} passed")

    for i, (name, ok, msg) in enumerate(tests, start=1):
        print(f"{i:3d}) {'PASS' if ok else 'FAIL'} - {name}")
        if not ok and msg:
            print(f"     {msg}")

    # Optional CI behavior
    if any(not ok for _, ok, _ in tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
