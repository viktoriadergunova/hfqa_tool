# main.py
import argparse
import logging
import time
from pathlib import Path

import yaml

from etl.preprocessing import prepare_data_from_excel
from vocab_check.apply_functional_check import add_mandatory_flags, add_range_and_allowed_flags
from vocab_check.apply_conditional_check import apply_conditional_rules
from quality_score.apply_u_quality_score import calculate_u_score
from quality_score.combine_scores import combine_u_m_p_scores
from quality_score.apply_m_quality_score import calculate_m_score
from quality_score.apply_p_flags import calculate_p_flags

from utils.logging_utils import (
    setup_logging,
    log_file_written,
    log_vocab_validation_summary,
    log_row_results,
    log_violations,
)
from utils.schema_loader import load_all_schemas
from utils.excel_writer import (
    generate_vocab_check_comments,
    write_excel_with_vocab_check_comments,
    generate_quality_score_column,
    write_excel_with_quality_score,
)



from utils.json_writer import write_validation_report, write_quality_report


def run_pipeline(
    input_path: str,
    out_report: str,
    mode: str,
    sheet_name: int = 0,
    meta_rows: int = 7,
    debug_prefix: str | None = None,
    out_excel: str | None = None,
    log_each_row: bool = True,
    log_each_violation: bool = True,
    include_row_results_in_json: bool = False,
) -> None:
    start_time = time.time()
    input_path = Path(input_path)

    # 1) Load schemas (already normalized by loader, if implemented there)
    schema, cond_cfg, qc_schema = load_all_schemas()

    # 2) Prepare data
    parquet_path = input_path.with_suffix(".raw.parquet")
    df_raw, df_meta, df_data_typed = prepare_data_from_excel(
        excel_path=str(input_path),
        schema=schema,
        sheet_name=sheet_name,
        meta_rows=meta_rows,
        parquet_path=str(parquet_path),
    )

    # Debug dumps (data + schemas)
    if debug_prefix:

        debug_dir = Path(debug_prefix)
        debug_dir.mkdir(parents=True, exist_ok=True)

        # dump prepared data
        df_data_typed.to_parquet(debug_dir / "02_data_prepared.parquet", index=False)

        # dump raw loaded schemas
        with open(debug_dir / "schema.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(schema, f, sort_keys=False, allow_unicode=True)

        with open(debug_dir / "conditional_rules.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cond_cfg, f, sort_keys=False, allow_unicode=True)

        with open(debug_dir / "quality_schema.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(qc_schema, f, sort_keys=False, allow_unicode=True)

        # dump effective/normalized schema
        used_columns = set(df_data_typed.columns)
        schema_normalized = {
            "columns": {
                k: v
                for k, v in schema.get("columns", {}).items()
                if k in used_columns
            },
            "core": schema.get("core", {}),
        }

        with open(debug_dir / "schema_normalized.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(schema_normalized, f, sort_keys=False, allow_unicode=True)

        logging.info("Debug artifacts written to folder: %s", debug_dir)


    # 3) Vocab mode
    if mode == "vocab":
        logging.info("Starting Vocabulary Validation Mode...")

        df_checked = add_mandatory_flags(df_data_typed, schema)
        df_checked = add_range_and_allowed_flags(df_checked, schema)
        df_checked = apply_conditional_rules(df_checked, cond_cfg)

        if debug_prefix:
            df_checked.to_parquet(f"{debug_prefix}_04_checked.parquet", index=False)

        first_data_index = int(df_checked.index.min()) if len(df_checked) > 0 else 0

        summary, row_results, violations = write_validation_report(
            df_checked=df_checked,
            schema=schema,
            meta_rows=meta_rows,
            first_data_index=first_data_index,
            df_raw=df_raw,
            df_data_typed=df_data_typed,
            start_time=start_time,
            out_path=out_report,
            include_row_results_in_json=include_row_results_in_json,
        )

        # Console logging
        if log_each_row:
            log_row_results(row_results)
        if log_each_violation:
            log_violations(violations)
        log_vocab_validation_summary(summary)

        # Excel output
        if out_excel:
            df_for_excel = generate_vocab_check_comments(
                df_raw=df_raw,
                df_checked=df_checked,
                schema=schema,
                meta_rows=meta_rows,
                out_col="Validation_Comments",
            )

            write_excel_with_vocab_check_comments(df_meta=None, df_with_comments=df_for_excel, output_path=out_excel)
            log_file_written("Excel", out_excel)


    # 4) Quality mode
    elif mode == "quality":
        logging.info("Starting IHFC Quality Scoring Mode...")
        # --- U score ---
        df_data_typed["quality_U"] = calculate_u_score(
            df_data_typed,
            qc_schema=qc_schema,
        )

        # --- M score (marine / continental routed) ---
        df_data_typed["quality_M"] = calculate_m_score(
            df_data_typed,
            qc_schema=qc_schema,
        )

        # --- P flags (7-letter paper code) ---
        df_data_typed["quality_P"] = calculate_p_flags(
            df_data_typed,
            qc_schema=qc_schema,
        )

        # --- Combine U + M + P  ---
        df_data_typed = combine_u_m_p_scores(
            df_data_typed,
            u_col="quality_U",
            m_col="quality_M",
            p_col="quality_P",
           # out_col="quality_Q",          # U2.M3x
            out_col_with_p="quality_QP",  # U2.M3x.SxxxCxh
           # out_rank_col="quality_rank",
            separator=".",
        )

        # Excel output (quality mode): original sheet + one extra column `quality_score`
        if out_excel:
            df_for_excel = generate_quality_score_column(
                df_raw=df_raw,
                df_scored=df_data_typed,
                meta_rows=meta_rows,
                source_col="quality_QP",   # or "quality_Q"
                out_col="quality_score",
            )

            write_excel_with_quality_score(df_meta, df_for_excel, out_excel)
            log_file_written("Excel", out_excel)

        if out_report:

            write_quality_report(
                df_data_typed=df_data_typed,
                meta_rows=meta_rows,
                start_time=start_time,
                out_path=out_report,
            )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to input Excel file")
    parser.add_argument("--out-json", help="Path to output JSON report")
    parser.add_argument("--out-excel", help="Path to output Excel file with validation comments")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--vocab-check", action="store_true", help="Run vocabulary validation")
    mode_group.add_argument("--quality-score", action="store_true", help="Run quality scoring")

    parser.add_argument("--sheet", type=int, default=0)
    parser.add_argument("--meta-rows", type=int, default=7)
    parser.add_argument("--debug-prefix", default=None)

    parser.add_argument("--run-tests", action="store_true")

    # Logging toggles
    parser.add_argument("--no-row-log", action="store_true", help="Disable per-row OK/ERROR console logging")
    parser.add_argument("--no-violation-log", action="store_true", help="Disable per-violation console logging")
    parser.add_argument(
        "--include-row-results-in-json",
        action="store_true",
        help="Also store row_results array in the JSON report",
    )

    args = parser.parse_args()
    setup_logging()

    if args.run_tests:
        from testing.run.functional.check_range_flags import main as run_range_tests
        from testing.run.functional.check_obligation_flags import main as run_ob_tests
        from testing.run.functional.check_allowed_flags import main as run_allowed_tests
        from testing.run.conditional.check_conditions_test import main as run_cond_tests
        from testing.run.quality_score.check_u1_score import main as run_u_score_tests
        from testing.run.quality_score.check_m1_score import main as run_m_score_tests

        run_range_tests()
        run_ob_tests()
        run_allowed_tests()
       # run_cond_tests()
       # run_u_score_tests()
       # run_m_score_tests()
        return

    mode = "vocab" if args.vocab_check else "quality"

    run_pipeline(
        input_path=args.input,
        out_report=args.out_json,
        mode=mode,
        sheet_name=args.sheet,
        meta_rows=args.meta_rows,
        debug_prefix=args.debug_prefix,
        out_excel=args.out_excel,
        log_each_row=not args.no_row_log,
        log_each_violation=not args.no_violation_log,
        include_row_results_in_json=args.include_row_results_in_json,
    )


if __name__ == "__main__":
    main()
