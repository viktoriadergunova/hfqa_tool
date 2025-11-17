import argparse
import json
import logging
import pandas as pd
import yaml

from etl.normalization import normalize_only_data_rows
from etl.typing import apply_schema_types
from etl.validation import add_mandatory_flags, add_range_and_allowed_flags


def run_validation(input_path: str, schema_path: str, out_report: str):
    logging.info("reading excel: %s", input_path)
    df_raw = pd.read_excel(input_path, dtype=str)

    logging.info("loading schema: %s", schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    # pipeline
    df_meta, df_data_norm = normalize_only_data_rows(df_raw, schema)
    df_data_typed = apply_schema_types(df_data_norm, schema)
    df_checked = add_mandatory_flags(df_data_typed, schema)
    df_checked = add_range_and_allowed_flags(df_checked, schema)

    # flags sammeln
    issue_cols = [c for c in df_checked.columns
                  if c.endswith("__missing") or c.endswith("__invalid") or c.endswith("__out_of_range")]

    issues = []
    for idx, row in df_checked.iterrows():
        found = {col: True for col in issue_cols if bool(row.get(col, False))}
        if found:
            issues.append({"row_index": int(idx), "issues": found})

    report = {
        "file": input_path,
        "num_rows": int(len(df_checked)),
        "num_issue_rows": int(len(issues)),
        "columns_with_flags": sorted(issue_cols),
        "issues": issues,
    }

    logging.info("writing report: %s", out_report)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pfad zur Excel-Datei")
    parser.add_argument("--schema", required=True, help="Pfad zum YAML-Schema")
    parser.add_argument("--out", required=True, help="Pfad für JSON-Report")
    args = parser.parse_args()

    run_validation(args.input, args.schema, args.out)


if __name__ == "__main__":
    main()
