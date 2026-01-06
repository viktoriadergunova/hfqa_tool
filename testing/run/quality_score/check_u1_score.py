import yaml
import pandas as pd

from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows, normalize_vocabulary_series
from etl.typecasting import apply_schema_types

from quality_score.apply_u_quality_score import (
    calculate_u_score,
    inherit_u_score_to_parent
)

INPUT_XLSX = "testing/run/testing_files/u1_score_tests.xlsx"
SCHEMA_PATH = "hf_schema.yaml"
QC_SCHEMA_PATH = "quality_score_schema.yaml"


def main():
    # 1) Load Schemas
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    with open(QC_SCHEMA_PATH, "r", encoding="utf-8") as f:
        qc_schema = yaml.safe_load(f)

    # 2) Pipeline
    df_raw = excel_to_parquet(INPUT_XLSX, "temp_u.parquet", sheet_name=0, meta_rows=0)
    _, df_data_norm = normalize_only_data_rows(df_raw, schema)
    df_typed = apply_schema_types(df_data_norm, schema)

    # 3) Apply U-score logic (child rows only)
    df_typed["quality_U"] = calculate_u_score(df_typed, qc_schema)

    # 4) Schema-driven inheritance setup
    u_calc = qc_schema["u_score"]["calculation"]

    site_col = u_calc["site_name_col"]
    rel_col = u_calc["relevance_col"]
    ROLE_CHILD = u_calc.get("role_child", "[yes]")
    ROLE_PARENT = u_calc.get("role_parent", "[no]")

    relevance = normalize_vocabulary_series(df_typed[rel_col])

    # Parent inheritance table
    df_parents = inherit_u_score_to_parent(df_typed, qc_schema)
    parent_map = df_parents.set_index(site_col)["parent_quality_U"].to_dict()

    # Final U assignment
    def finalize_u(idx: int) -> str:
        if relevance.loc[idx] == ROLE_PARENT:
            site = df_typed.loc[idx, site_col]
            return parent_map.get(site, "Ux")
        if relevance.loc[idx] == ROLE_CHILD:
            return str(df_typed.loc[idx, "quality_U"])
        return "Ux"

    df_typed["final_U"] = pd.Series(
        [finalize_u(i) for i in df_typed.index],
        index=df_typed.index
    )

    # 5) Verification
    # Convention:
    #   expected result column = "_results"
    #   case label column      = "_case"
    tests = []
    for idx, row in df_typed.iterrows():
        expected = row.get("_results")
        if pd.isna(expected):
            continue

        expected = str(expected)
        actual = str(row["final_U"])

        case = row.get("_case", "unknown")
        site = row.get(site_col)
        role = "Parent" if relevance.loc[idx] == ROLE_PARENT else "Child"

        name = f"[u_score] {site} ({role}) | Case: {case}"
        ok = (actual == expected)
        tests.append((name, ok, "" if ok else f"Expected {expected}, got {actual}"))

    # 6) Report
    total = len(tests)
    passed = sum(1 for _, ok, _ in tests if ok)

    print("\nIHFC U-Score Quality Test Summary")
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
