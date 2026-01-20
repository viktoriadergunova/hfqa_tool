from __future__ import annotations

import pandas as pd

from quality_score.apply_m_quality_score_borehole import calculate_m_score_borehole
from quality_score.apply_m_quality_score_marine import calculate_m_score_marine


def calculate_m_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Unified M-score entrypoint.
    Routes by domain column (A7: continental/marine).
    """
    m_cfg = qc_schema.get("m_score", {})
    calc = m_cfg.get("calculation", {})
    route_col = calc.get("m_route_col", "A7")

    if route_col not in df.columns:
        raise KeyError(f"Routing column '{route_col}' not found in dataframe")

    domain = df[route_col].astype("string").str.strip().str.lower().fillna("")
    is_marine = domain.eq("marine")
    is_cont = domain.eq("continental") | (domain == "")

    out = pd.Series(pd.NA, index=df.index, dtype="string")

    if bool(is_marine.any()):
        out.loc[is_marine] = calculate_m_score_marine(df.loc[is_marine].copy(), qc_schema=qc_schema)

    if bool(is_cont.any()):
        out.loc[is_cont] = calculate_m_score_borehole(df.loc[is_cont].copy(), qc_schema=qc_schema)

    # If there are unexpected values, mark them as missing with x to be conservative
    unknown = ~(is_marine | is_cont)
    if bool(unknown.any()):
        # fallback: borehole scoring would be wrong; return M4x (worst + flagged)
        missing_suffix = str(m_cfg.get("thresholds", {}).get("missing_suffix", "x"))
        out.loc[unknown] = f"M4{missing_suffix}"

    return out
