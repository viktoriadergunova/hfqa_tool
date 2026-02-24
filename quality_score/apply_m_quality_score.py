# quality_score/calculate_m_score.py
from __future__ import annotations

import pandas as pd

from quality_score.apply_m_quality_score_borehole import calculate_m_score_borehole
from quality_score.apply_m_quality_score_marine import calculate_m_score_marine

_MARINE_P12_TOKENS = {
    "[probing-(onshore-lake-river-etc.)]",
    "[probing-(offshore-ocean)]",
    "[probing-clustering]",
}

_BOREHOLE_P12_TOKENS = {
    "[drilling]",
    "[mining]",
    "[tunneling]",
    "[drilling-clustering]",
    "[indirect-(gtm-bsr-cpd-etc.)]",
}


def _get_route(cell_value: str) -> str:
    """Returns 'marine', 'borehole', or 'not_determined'."""
    if not cell_value:
        return "not_determined"
    tokens = {t.strip() for t in cell_value.split(";") if t.strip()}
    if tokens & _MARINE_P12_TOKENS:
        return "marine"
    if tokens & _BOREHOLE_P12_TOKENS:
        return "borehole"
    return "not_determined"  # [unspecified], [other], empty


def calculate_m_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    m_cfg = qc_schema.get("m_score", {})
    calc  = m_cfg.get("calculation", {})
    route_col = calc.get("m_route_col", "P12")  # <-- change default to P12
    thr = m_cfg.get("thresholds", {})
    missing_suffix = str(thr.get("missing_suffix", "x"))

    p12_values = (
        df[route_col]
        .astype("string")
        .str.lower()
        .fillna("")
        .str.strip()
    )

    routes = p12_values.apply(_get_route)

    out = pd.Series(pd.NA, index=df.index, dtype="string")

    marine_idx   = df.index[routes == "marine"]
    borehole_idx = df.index[routes == "borehole"]
    nd_idx       = df.index[routes == "not_determined"]

    if len(marine_idx):
        out.loc[marine_idx] = calculate_m_score_marine(df.loc[marine_idx].copy(), qc_schema)

    if len(borehole_idx):
        out.loc[borehole_idx] = calculate_m_score_borehole(df.loc[borehole_idx].copy(), qc_schema)

    if len(nd_idx):
        out.loc[nd_idx] = "Mx"  # not determined, no calculation

    return out


def calculate_m_route_debug(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """Debug column: returns 'marine', 'borehole', or 'not_determined' per row."""
    calc = qc_schema.get("m_score", {}).get("calculation", {})
    route_col = calc.get("m_route_col", "P12")

    p12_values = (
        df[route_col]
        .astype("string")
        .str.lower()
        .fillna("")
        .str.strip()
    )
    return p12_values.apply(_get_route).astype("string")