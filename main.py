import argparse
import json
import logging
from pathlib import Path
import time

import pandas as pd
import yaml

from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows
from etl.typecasting import apply_schema_types
from vocab_check.functional_validator import add_mandatory_flags, add_range_and_allowed_flags
from vocab_check.conditional_validator import apply_conditional_rules


def run_validation(
    input_path: str,
    out_report: str,
    sheet_name: int = 0,
    meta_rows: int = 7,
    debug_prefix: str | None = None,
) -> None:
    start_time = time.time()

    input_path = Path(input_path)

    # -------------------------------------------------------------
    # 1) Excel -> Parquet + row_type via excel_to_parquet
    # -------------------------------------------------------------
    parquet_path = input_path.with_suffix(".raw.parquet")

    logging.info("reading Excel via excel_to_parquet: %s (sheet=%s)", input_path, sheet_name)
    df_raw = excel_to_parquet(
        excel_path=str(input_path),
        parquet_path=str(parquet_path),
        sheet_name=sheet_name,
        meta_rows=meta_rows,
    )

    if "row_type" not in df_raw.columns:
        logging.warning("no 'row_type' column present; treating all rows as data")
        df_raw["row_type"] = "data"

    if debug_prefix:
        df_raw.to_parquet(f"{debug_prefix}_01_raw.parquet", index=False)

    # -------------------------------------------------------------
    # 2) Schema laden
    # -------------------------------------------------------------
    schema_path = "hf_schema.yaml"
    logging.info("loading schema: %s", schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    # -------------------------------------------------------------
    # 2b) Conditional rules laden
    # -------------------------------------------------------------
    conditional_path = "conditional_rules.yaml"
    logging.info("loading conditional rules: %s", conditional_path)
    with open(conditional_path, "r", encoding="utf-8") as f:
        cond_cfg = yaml.safe_load(f)

    # -------------------------------------------------------------
    # 3) Normalisierung
    # -------------------------------------------------------------
    df_meta, df_data_norm = normalize_only_data_rows(df_raw, schema)
    if debug_prefix:
        df_data_norm.to_parquet(f"{debug_prefix}_02_data_norm.parquet", index=False)

    # -------------------------------------------------------------
    # 4) Typ-Casting 
    # -------------------------------------------------------------
    df_data_typed = apply_schema_types(df_data_norm, schema)
    if debug_prefix:
        df_data_typed.to_parquet(f"{debug_prefix}_03_data_typed.parquet", index=False)

    # -------------------------------------------------------------
    # 5) Validation (funktional + konditional)
    # -------------------------------------------------------------
    df_checked = add_mandatory_flags(df_data_typed, schema)
    df_checked = add_range_and_allowed_flags(df_checked, schema)
    df_checked = apply_conditional_rules(df_checked, cond_cfg)

    if debug_prefix:
        df_checked.to_parquet(f"{debug_prefix}_04_checked.parquet", index=False)

    # -------------------------------------------------------------
    # 6) Report + Statistik
    # -------------------------------------------------------------

    if len(df_checked.index) > 0:
        first_data_index = int(df_checked.index.min())
    else:
        first_data_index = 0

    def get_col_comment(schema_dict: dict, col_name: str) -> str:
        for section in ("columns", "core"):
            section_dict = schema_dict.get(section, {})
            if col_name in section_dict:
                return section_dict[col_name].get("comment", "")
        return ""

    problems_grouped = []
    violations = []  # jede Verletzung einzeln

    # Gesamtstatistik
    functional_error_count = 0
    conditional_error_count = 0
    per_flag_counts: dict[str, int] = {}

    # NEU: funktionale Unterteilung
    missing_error_count = 0          # __missing
    range_error_count = 0            # __out_of_range
    allowed_error_count = 0          # __invalid
    other_functional_error_count = 0 # falls später noch andere Flags dazu kommen

    rows_any_error = set()
    rows_functional_error = set()
    rows_conditional_error = set()

    for col in df_checked.columns:
        if "__" not in col:
            continue

        flag_col = df_checked[col]
        if flag_col.dtype != bool:
            continue

        col_name, flag_type = col.split("__", 1)

        bad_indices_raw = df_checked.index[flag_col].tolist()
        if not bad_indices_raw:
            continue

        n_errors_flag = int(flag_col.sum())

        is_conditional = flag_type.startswith("cond_")

        if is_conditional:
            conditional_error_count += n_errors_flag
        else:
            functional_error_count += n_errors_flag
            # aufsplitten
            if flag_type.endswith("missing"):
                missing_error_count += n_errors_flag
            elif flag_type.endswith("out_of_range"):
                range_error_count += n_errors_flag
            elif flag_type.endswith("invalid"):
                allowed_error_count += n_errors_flag
            else:
                other_functional_error_count += n_errors_flag

        per_flag_counts[flag_type] = per_flag_counts.get(flag_type, 0) + n_errors_flag

        bad_indices = [int(i) for i in bad_indices_raw]
        comment = get_col_comment(schema, col_name)
        data_row_numbers = []

        for idx in bad_indices:
            data_row = int((idx - first_data_index) + 1)
            data_row_numbers.append(data_row)

            logging.info(
                "row=%d col=%s flag=%s comment=%s",
                data_row,
                col_name,
                flag_type,
                comment,
            )

            violations.append(
                {
                    "row_data_number": data_row,
                    "column": col_name,
                    "flag": flag_type,
                    "comment": comment,
                }
            )

            rows_any_error.add(data_row)
            if is_conditional:
                rows_conditional_error.add(data_row)
            else:
                rows_functional_error.add(data_row)

        problems_grouped.append(
            {
                "column": str(col_name),
                "flag": str(flag_type),
                "comment": str(comment),
            }
        )

    end_time = time.time()
    runtime_sec = end_time - start_time

    total_rows = int(len(df_raw))
    data_rows = int(len(df_data_norm))

    summary = {
        "total_rows_in_sheet": total_rows,
        "data_rows_validated": data_rows,

        "rows_with_any_error": len(rows_any_error),
        "rows_with_functional_error": len(rows_functional_error),
        "rows_with_conditional_error": len(rows_conditional_error),

        "functional_error_count": functional_error_count,
        "conditional_error_count": conditional_error_count,

        # NEU: funktional nach Typ
        "functional_missing_error_count": missing_error_count,
        "functional_range_error_count": range_error_count,
        "functional_allowed_error_count": allowed_error_count,
        "functional_other_error_count": other_functional_error_count,

        "per_flag_counts": per_flag_counts,
        "runtime_seconds": runtime_sec,
    }

    logging.info("====VOCABULARY VALIDATION SUMMARY ====")
    logging.info("Total rows in sheet (incl. meta): %d", total_rows)
    logging.info("Data rows validated: %d", data_rows)
    logging.info("Rows with any error: %d", len(rows_any_error))
    logging.info("Rows with functional errors: %d", len(rows_functional_error))
    logging.info("Rows with conditional errors: %d", len(rows_conditional_error))
    logging.info("Functional errors (cell count): %d", functional_error_count)
    logging.info("  ├─ Missing errors: %d", missing_error_count)
    logging.info("  ├─ Range errors: %d", range_error_count)
    logging.info("  ├─ Allowed-value errors: %d", allowed_error_count)
    logging.info("  └─ Other functional errors: %d", other_functional_error_count)
    logging.info("Conditional errors (cell count): %d", conditional_error_count)
    logging.info("Processing time: %.2f seconds", runtime_sec)

    logging.info("writing JSON report: %s", out_report)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(
            {
                "violations": violations,
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Path to input Excel file")
    parser.add_argument("--out", required=False, help="Path to output JSON report")
    parser.add_argument("--sheet", type=int, default=0, help="Excel sheet index (e.g. 0, 1, ...)")
    parser.add_argument("--meta-rows", type=int, default=7, help="Number of meta rows in Excel sheet")
    parser.add_argument(
        "--debug-prefix",
        default=None,
        help="Get intermediate Parquet files with this prefix")
    parser.add_argument("--run-tests", action="store_true", help="Run internal test suites")
    

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.run_tests:
        logging.info("Starting internal test suites...")
        

        from testing.run.functional.check_range_flags import main as run_range_tests
        from testing.run.functional.check_obligation_flags import main as run_ob_tests
        from testing.run.functional.check_allowed_flags import main as run_allowed_tests
        from testing.run.conditional.check_conditions_test import main as run_cond_tests
        
        # Test functions
        run_range_tests()
        run_ob_tests()
        run_allowed_tests()
        #run_cond_tests()

        return 

    run_validation(
        input_path=args.input,
        out_report=args.out,
        sheet_name=args.sheet,
        meta_rows=args.meta_rows,
        debug_prefix=args.debug_prefix,
    )


if __name__ == "__main__":
    main()
