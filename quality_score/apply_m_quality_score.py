from __future__ import annotations

import pandas as pd

from quality_score.apply_m_quality_score_borehole import calculate_m_score_borehole
from quality_score.apply_m_quality_score_marine import calculate_m_score_marine



def calculate_m_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Unified M-score entrypoint.
    Routes by location type in column P7 (multi-select).
    Marine = contains '[Offshore (marine)]'
    Continental = all other (including empty)
    """
    m_cfg = qc_schema.get("m_score", {})
    calc = m_cfg.get("calculation", {})
    route_col = calc.get("m_route_col", "P7")

    if route_col not in df.columns:
        raise KeyError(f"Routing column '{route_col}' not found in dataframe")

    values = df[route_col].astype("string").str.lower().fillna("")
    is_marine = values.str.contains(r"\[offshore \(marine\)\]", regex=True)
    is_cont = ~is_marine  # All others

    out = pd.Series(pd.NA, index=df.index, dtype="string")

    if is_marine.any():
        out.loc[is_marine] = calculate_m_score_marine(df.loc[is_marine].copy(), qc_schema=qc_schema)

    if is_cont.any():
        out.loc[is_cont] = calculate_m_score_borehole(df.loc[is_cont].copy(), qc_schema=qc_schema)

    return out
