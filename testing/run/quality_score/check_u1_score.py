import yaml
import pandas as pd
import numpy as np
from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows
from etl.typecasting import apply_schema_types
from quality_score.apply_quality_score import calculate_u_score, inherit_u_score_to_parent

INPUT_XLSX = "testing/run/testing_files/u1_score_tests.xlsx"
SCHEMA_PATH = "hf_schema.yaml"
QC_SCHEMA_PATH = "quality_score_schema.yaml"

def main():
    # 1. Load Schemas
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    with open(QC_SCHEMA_PATH, "r", encoding="utf-8") as f:
        qc_schema = yaml.safe_load(f)

    # 2. Pipeline
    df_raw = excel_to_parquet(INPUT_XLSX, "temp_u.parquet", sheet_name=0, meta_rows=0)
    _, df_data_norm = normalize_only_data_rows(df_raw, schema)
    df_typed = apply_schema_types(df_data_norm, schema)

    # 3. Apply Logic
    # Calculate child scores
    df_typed['quality_U'] = calculate_u_score(df_typed, qc_schema)
    # Calculate parent inheritance
    df_parents = inherit_u_score_to_parent(df_typed, qc_schema)
    parent_map = df_parents.set_index('P3')['parent_quality_U'].to_dict()

    # Apply inheritance to the [no] rows
    def finalize(row):
        if row['C9'] == '[no]':
            return parent_map.get(row['P3'], "Ux")
        return row['quality_U']

    df_typed['final_U'] = df_typed.apply(finalize, axis=1)

    # 4. Verification
    tests = []
    for idx, row in df_typed.iterrows():
        expected = str(row.get("_results"))
        actual = str(row["final_U"])
        case = row.get("_case", "unknown")
        site = row.get("P3")
        role = "Parent" if row['C9'] == '[no]' else "Child"

        if pd.notna(row["_results"]):
            name = f"[u_score] {site} ({role}) | Case: {case}"
            ok = (actual == expected)
            tests.append((name, ok, "" if ok else f"Expected {expected}, got {actual}"))

    # 5. Report
    total = len(tests)
    passed = sum(1 for _, ok, _ in tests if ok)
    print(f"\nIHFC U-Score Quality Test Summary")
    bar = "[" + "#" * int(40 * passed / total) + "-" * (40 - int(40 * passed / total)) + "]"
    print(f"{bar}  {passed}/{total} passed")

    for i, (name, ok, msg) in enumerate(tests, start=1):
        print(f"{i:3d}) {'PASS' if ok else 'FAIL'} - {name}")
        if not ok: print(f"     {msg}")

if __name__ == "__main__":
    main()