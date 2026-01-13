import argparse
import json
import logging
from pathlib import Path
import time

import pandas as pd
import yaml
import numpy as np
from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows
from etl.typecasting import apply_schema_types
from vocab_check.apply_functional_check import add_mandatory_flags, add_range_and_allowed_flags
from vocab_check.apply_conditional_check import apply_conditional_rules
from quality_score.apply_u_quality_score import calculate_u_score, inherit_u_score_to_parent
from quality_score.apply_m_quality_score import calculate_m_score


def generate_validation_comments(df_checked: pd.DataFrame, schema: dict, first_data_index: int) -> pd.DataFrame:
    """
    Generate a validation comments column for each row in the dataframe.
    Returns a DataFrame with original data + 'Validation_Comments' column.
    """
    # Get original columns (without flag columns)
    original_cols = [col for col in df_checked.columns if "__" not in col]
    df_result = df_checked[original_cols].copy()
    
    def get_col_comment(schema_dict: dict, col_name: str) -> str:
        for section in ("columns", "core"):
            section_dict = schema_dict.get(section, {})
            if col_name in section_dict:
                return section_dict[col_name].get("comment", "")
        return ""
    
    # Create validation comments for each row
    validation_comments = []
    
    for idx in df_checked.index:
        row_comments = []
        
        # Check all flag columns for this row
        for col in df_checked.columns:
            if "__" not in col:
                continue
            
            if df_checked.loc[idx, col] == True:
                col_name, flag_type = col.split("__", 1)
                col_description = get_col_comment(schema, col_name)
                
                # Format the error message based on flag type
                if flag_type == "missing":
                    error_msg = f"[MISSING] {col_name} ({col_description})"
                elif flag_type == "out_of_range":
                    value = df_checked.loc[idx, col_name]
                    error_msg = f"[RANGE ERROR] {col_name} ({col_description})"
                elif flag_type == "invalid":
                    value = df_checked.loc[idx, col_name]
                    error_msg = f"[INVALID VALUE] {col_name} ({col_description})"
                elif flag_type.startswith("cond_"):
                    value = df_checked.loc[idx, col_name]
                    error_msg = f"[CONDITIONAL ERROR] {col_name} ({col_description}): {flag_type}"
                else:
                    error_msg = f"[ERROR] {col_name} ({col_description}): {flag_type}"
                
                row_comments.append(error_msg)
        
        # Join all comments for this row with separator
        if row_comments:
            validation_comments.append(" | ".join(row_comments))
        else:
            validation_comments.append("OK")
    
    df_result['Validation_Comments'] = validation_comments
    
    return df_result


def run_pipeline(
    input_path: str,
    out_report: str,
    mode: str,
    sheet_name: int = 0,
    meta_rows: int = 7,
    debug_prefix: str | None = None,
    out_excel: str | None = None,
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
    conditional_schema_path = "quality_score_schema.yaml"
    qc_schema_path = "conditional_rules.yaml"
    
    logging.info("Loading configuration schema")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    with open(conditional_schema_path, "r", encoding="utf-8") as f:
        cond_cfg = yaml.safe_load(f)

    with open(qc_schema_path, "r", encoding="utf-8") as f:
        qc_schema = yaml.safe_load(f)

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
    # 5) Branching: Quality Score vs. Vocab Check   
    # -------------------------------------------------------------
    if mode == "vocab":
        logging.info("Starting Vocabulary Validation Mode...")

    # -------------------------------------------------------------
    # Vocab Check: (funktional + konditional)
    # -------------------------------------------------------------
        df_checked = add_mandatory_flags(df_data_typed, schema)
        df_checked = add_range_and_allowed_flags(df_checked, schema)
        df_checked = apply_conditional_rules(df_checked, cond_cfg)

        if debug_prefix:
            df_checked.to_parquet(f"{debug_prefix}_04_checked.parquet", index=False)

    # -------------------------------------------------------------
    # Report + Statistik
    # -------------------------------------------------------------

        # Calculate row offset including meta rows
        # meta_rows are at the top, so data rows start after them
        row_offset = meta_rows + 1  # +1 for Excel header row
        
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

        # funktionale Unterteilung
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
                # Excel row = meta_rows + header + data_row_position
                excel_row = row_offset + (idx - first_data_index)
                site_name = str(df_checked.at[idx, 'P3']) if 'P3' in df_checked.columns else "N/A"

                logging.info(
                    "row=%d col=%s flag=%s site=%s comment=%s",
                    excel_row,
                    col_name,
                    flag_type,
                    site_name,
                    comment,
                )

                violations.append(
                    {
                        "row_excel_number": excel_row,
                        "column": col_name,
                        "site_name": site_name, 
                        "flag": flag_type,
                        "comment": comment,
                    }
                )

                rows_any_error.add(excel_row)
                if is_conditional:
                    rows_conditional_error.add(excel_row)
                else:
                    rows_functional_error.add(excel_row)

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

            # funktional nach Typ
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

        # -------------------------------------------------------------
        # Generate Excel output with validation comments
        # -------------------------------------------------------------
        if out_excel:
            logging.info("Generating Excel output with validation comments...")
            
            # Generate validation comments for data rows
            df_with_comments = generate_validation_comments(df_checked, schema, first_data_index)
            
            # Combine meta rows (if any) with data rows
            if df_meta is not None and len(df_meta) > 0:
                # Add empty validation comment column to meta rows
                df_meta['Validation_Comments'] = 'META ROW'
                
                # Ensure both dataframes have the same columns in the same order
                all_cols = df_with_comments.columns.tolist()
                for col in all_cols:
                    if col not in df_meta.columns:
                        df_meta[col] = pd.NA
                
                df_meta = df_meta[all_cols]
                df_final = pd.concat([df_meta, df_with_comments], ignore_index=False).sort_index()
            else:
                df_final = df_with_comments
            
            # Write to Excel
            logging.info(f"Writing Excel file: {out_excel}")
            df_final.to_excel(out_excel, index=False, sheet_name='Validation Results')
            logging.info(f"Excel file created successfully: {out_excel}")

        # Write JSON report
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

    elif mode == "quality":
        logging.info("Starting IHFC Quality Scoring Mode...")
        
        # 1. Calculate Scores
        df_data_typed['quality_U'] = calculate_u_score(df_data_typed, qc_schema)
        df_parent_summary = inherit_u_score_to_parent(df_data_typed, qc_schema)

        # 2. Fix NaN values for JSON compatibility (converts NaN to None/null)
        df_display = df_data_typed.replace({np.nan: None})
        
        # 3. Create a lookup dictionary for Parent Scores
        parent_lookup = df_parent_summary.set_index('ID')['parent_quality_U'].to_dict()

        first_data_index = int(df_display.index.min()) if len(df_display.index) > 0 else 0

        child_determinations = []
        for idx, row in df_display.iterrows():
            current_id = row.get('ID')
            data_row_number = int((idx - first_data_index) + 1)
            
            # 4. Build the improved visual structure
            child_determinations.append({
                "row_data_number": data_row_number,
                "site_id": current_id,
                "site_name": row.get('P3', 'N/A'),
                "determination_score": row['quality_U'],
                "inherited_site_score": parent_lookup.get(current_id, "Ux"),
                "values": {
                    "C1_HF": row.get('C1'),
                    "C2_Unc": row.get('C2')
                }
            })

        final_output = {
            "mode": "quality_scoring",
            "summary": {
                "total_determinations": len(df_data_typed),
                "high_accuracy_U1_U2": int(df_data_typed['quality_U'].isin(['U1', 'U2']).sum()),
                "runtime_seconds": time.time() - start_time
            },
            "detailed_results": child_determinations
        }

        # 6) Final Write
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        logging.info("Process completed in %.2f seconds.", time.time() - start_time)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, help="Path to input Excel file")
    parser.add_argument("--out-json", required=False, help="Path to output JSON report")
    parser.add_argument("--out-excel", required=False, help="Path to output Excel file with validation comments")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--vocab-check", action="store_true", help="Run vocabulary validation")
    mode_group.add_argument("--quality-score", action="store_true", help="Run U, M, P quality scoring")

    parser.add_argument("--sheet", type=int, default=0, help="Excel sheet index (e.g. 0, 1, ...)")
    parser.add_argument("--meta-rows", type=int, default=7, help="Number of meta rows in Excel sheet")
    parser.add_argument(
        "--debug-prefix",
        default=None,
        help="Get intermediate Parquet files with this prefix")
    parser.add_argument("--run-tests", action="store_true", help="Run internal test suites")
    

    args = parser.parse_args()
    mode = "vocab" if args.vocab_check else "quality"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.run_tests:
        logging.info("Starting internal test suites...")
        

        from testing.run.functional.check_range_flags import main as run_range_tests
        from testing.run.functional.check_obligation_flags import main as run_ob_tests
        from testing.run.functional.check_allowed_flags import main as run_allowed_tests
        from testing.run.conditional.check_conditions_test import main as run_cond_tests
        from testing.run.quality_score.check_u1_score import main as run_u_score_tests
        from testing.run.quality_score.check_m1_score import main as run_m_score_tests
        
        # Test functions
        run_range_tests()
        run_ob_tests()
        run_allowed_tests()
        #run_cond_tests()
        run_u_score_tests()
        run_m_score_tests()


        return 

    run_pipeline(
        input_path=args.input,
        out_report=args.out_json,
        mode=mode,
        sheet_name=args.sheet,
        meta_rows=args.meta_rows,
        debug_prefix=args.debug_prefix,
        out_excel=args.out_excel,
    )


if __name__ == "__main__":
    main()