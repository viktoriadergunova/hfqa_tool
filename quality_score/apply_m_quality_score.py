# quality_score/calculate_m_score.py
from __future__ import annotations

import pandas as pd

from quality_score.apply_m_quality_score_borehole import calculate_m_score_borehole
from quality_score.apply_m_quality_score_marine import calculate_m_score_marine

# P7 vocabulary values that route to marine (probe-sensing) logic.
# Source: Fuchs et al. (2023) Appendix A, field P07.
# Full vocabulary:
#   [Onshore (continental)]       → borehole
#   [Onshore (lake, river, etc.)] → MARINE  (probe-sensing in lakes/rivers)
#   [Offshore (continental)]      → MARINE  (probe-sensing on continental shelf)
#   [Offshore (marine)]           → MARINE  (probe-sensing in open ocean)
#   [unspecified]                 → borehole (conservative fallback)

_MARINE_TOKENS = {
    "[offshore-(marine)]",
    "[offshore-(continental)]",
    "[onshore-(lake-river-etc.)]",
}


def _is_marine_row(cell_value: str) -> bool:
    """
    Returns True if ANY token in the cell matches a marine environment.
    Handles multi-value cells separated by ';'.
    NOTE: do NOT split on comma — commas appear inside vocabulary tokens
    such as '[Onshore (lake, river, etc.)]'.
    Input is already lowercased.
    """
    if not cell_value:
        return False
    tokens = {t.strip() for t in cell_value.split(";") if t.strip()}
    return bool(tokens & _MARINE_TOKENS)


def calculate_m_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Unified M-score entrypoint.

    Routes each row to the correct scoring logic based on P7 (environment):
      - [Offshore (marine)]           → marine / probe-sensing  (Table 2)
      - [Offshore (continental)]      → marine / probe-sensing  (Table 2)
      - [Onshore (lake, river, etc.)] → marine / probe-sensing  (Table 2)
      - [Onshore (continental)]       → borehole / mine         (Table 3)
      - [unspecified] / empty         → borehole / mine         (Table 3, conservative)
    """
    m_cfg = qc_schema.get("m_score", {})
    calc  = m_cfg.get("calculation", {})
    route_col = calc.get("m_route_col", "P7")

    if route_col not in df.columns:
        raise KeyError(
            f"Routing column '{route_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    # Normalise to lowercase string, fill NA → empty string
    env_values = (
        df[route_col]
        .astype("string")
        .str.lower()
        .fillna("")
        .str.strip()
    )

    is_marine = env_values.apply(_is_marine_row)
    is_borehole = ~is_marine

    out = pd.Series(pd.NA, index=df.index, dtype="string")

    if is_marine.any():
        marine_idx = df.index[is_marine]
        out.loc[marine_idx] = calculate_m_score_marine(
            df.loc[marine_idx].copy(), qc_schema=qc_schema
        )

    if is_borehole.any():
        borehole_idx = df.index[is_borehole]
        out.loc[borehole_idx] = calculate_m_score_borehole(
            df.loc[borehole_idx].copy(), qc_schema=qc_schema
        )

    return out