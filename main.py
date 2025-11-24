import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from etl.extract_excel import excel_to_parquet
from etl.normalization import normalize_only_data_rows
from etl.typecasting import apply_schema_types
from etl.validator import add_mandatory_flags, add_range_and_allowed_flags


def run_validation(
    input_path: str,
    schema_path: str,
    out_report: str,
    sheet_name: int = 0,
    meta_rows: int = 7,
    debug_prefix: str | None = None,
) -> None:
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
    # 2) Schema
    # -------------------------------------------------------------
    logging.info("loading schema: %s", schema_path)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f)

    # -------------------------------------------------------------
    # 3) Normalization 
    # -------------------------------------------------------------
    df_meta, df_data_norm = normalize_only_data_rows(df_raw, schema)
    # df_meta wird für die Validierung ignoriert
    if debug_prefix:
        df_data_norm.to_parquet(f"{debug_prefix}_02_data_norm.parquet", index=False)

    # -------------------------------------------------------------
    # 4) Typ-Casting 
    # -------------------------------------------------------------
    df_data_typed = apply_schema_types(df_data_norm, schema)

    if debug_prefix:
        df_data_typed.to_parquet(f"{debug_prefix}_03_data_typed.parquet", index=False)

    # -------------------------------------------------------------
    # 5) Validation
    # -------------------------------------------------------------
    df_checked = add_mandatory_flags(df_data_typed, schema)
    df_checked = add_range_and_allowed_flags(df_checked, schema)

    if debug_prefix:
        df_checked.to_parquet(f"{debug_prefix}_04_checked.parquet", index=False)

    # -------------------------------------------------------------
    # 6) Report 
    # -------------------------------------------------------------

    # Startindex der Datenzeilen (für Zählung ab 1)
    if len(df_checked.index) > 0:
        first_data_index = int(df_checked.index.min())
    else:
        first_data_index = 0

    def get_col_comment(schema_dict: dict, col_name: str) -> str:
        """Kommentar für eine Spalte aus dem Schema holen."""
        for section in ("columns", "core"):
            section_dict = schema_dict.get(section, {})
            if col_name in section_dict:
                return section_dict[col_name].get("comment", "")
        return ""

    problems = []

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

        problems.append(
            {
                "column": str(col_name),
                "flag": str(flag_type),
                "rows_data_number": [int(x) for x in data_row_numbers], 
                "comment": str(comment),
            }
        )

    logging.info("writing JSON report: %s", out_report)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump({"problems": problems}, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pfad zur Excel-Datei")
    parser.add_argument("--schema", required=True, help="Pfad zum YAML-Schema")
    parser.add_argument("--out", required=True, help="Pfad für JSON-Report")
    parser.add_argument("--sheet", type=int, default=0, help="Excel sheet index (e.g. 0, 1, ...)")
    parser.add_argument("--meta-rows", type=int, default=7, help="Anzahl Meta-Zeilen am Tabellenkopf")
    parser.add_argument(
        "--debug-prefix",
        default=None,
        help="Prefix für Debug-Parquet-Dateien (ohne Endung); wenn gesetzt, werden _01.._04 geschrieben",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_validation(
        input_path=args.input,
        schema_path=args.schema,
        out_report=args.out,
        sheet_name=args.sheet,
        meta_rows=args.meta_rows,
        debug_prefix=args.debug_prefix,
    )


if __name__ == "__main__":
    main()

