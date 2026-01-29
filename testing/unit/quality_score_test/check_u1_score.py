import yaml
import pandas as pd
import numpy as np
from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows
from etl.typecasting import apply_schema_types
from quality_score.apply_u_quality_score import calculate_u_score
from quality_score.helper_functions import load_quality_schema 

INPUT_XLSX = "testing/run/testing_files/u1_score_tests.xlsx"
SCHEMA_PATH = "hf_schema.yaml"
QC_SCHEMA_PATH = "quality_score_schema.yaml"

def main():
    # 1. Load Schemas
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)
        qc_schema = load_quality_schema(QC_SCHEMA_PATH)
    # 2. Pipeline
    df_raw = excel_to_parquet(INPUT_XLSX, "temp_u.parquet", sheet_name=0, meta_rows=0)
    _, df_data_norm = normalize_only_data_rows(df_raw, schema)
    df_typed = apply_schema_types(df_data_norm, schema)

    # 3. Apply U-Score Calculation
    df_typed['_calc_quality_U'] = calculate_u_score(df_typed, qc_schema)

    # 4. Verification
    tests = []
    for idx, row in df_typed.iterrows():
        expected_label = str(row.get("expected_result")).strip()
        actual_label = str(row["_calc_quality_U"]).strip()
        expected_cov = row.get("_numeric_result", np.nan)

        case = row.get("_case", "unknown")
        site = row.get("P3", "unknown")

        # Compute actual COV from C1 and C2
        try:
            val = float(row["C1"])
            unc = float(row["C2"])
            actual_cov = (unc / abs(val)) * 100 if val != 0 else np.nan
        except:
            actual_cov = np.nan

        # Check label and cov
        label_ok = actual_label == expected_label
        cov_ok = pd.isna(expected_cov) and pd.isna(actual_cov) or round(expected_cov, 2) == round(actual_cov, 2)

        name = f"[u_score] {site} | Case: {case}"
        ok = label_ok and cov_ok

        # Compose detailed message
        msg = ""
        if not ok:
            if not label_ok:
                msg += f"Label mismatch: Expected {expected_label}, got {actual_label}. "
            if not cov_ok:
                msg += f"COV mismatch: Expected {expected_cov}, got {actual_cov:.2f}. "

        tests.append((name, ok, msg))


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
