# quality_score/combine_quality_scores.py
from __future__ import annotations

import pandas as pd


_U_ORDER = {"U1": 1, "U2": 2, "U3": 3, "U4": 4, "Ux": 9}
_M_ORDER = {"M1": 1, "M2": 2, "M3": 3, "M4": 4, "Mx": 9}


def _norm_u(u: str | None) -> str:
    if u is None:
        return "Ux"
    s = str(u).strip()
    return s if s in _U_ORDER else "Ux"


def _norm_m(m: str | None) -> str:
    if m is None:
        return "Mx"
    s = str(m).strip()
    if s in ("M1", "M2", "M3", "M4", "M1x", "M2x", "M3x", "M4x"):
        return s
    return "Mx"


def _base_m(m: str) -> str:
    return m[:2] if m.startswith("M") and len(m) >= 2 else "Mx"


def combine_u_m_scores(
    df: pd.DataFrame,
    u_col: str = "quality_U",
    m_col: str = "quality_M",
    out_col: str = "quality_Q",
    out_rank_col: str = "quality_rank",
    separator: str = ".",
) -> pd.DataFrame:
    """
    Adds:
      - out_col: 'U2.M3x'
      - out_rank_col: worst-case numeric rank for sorting/filtering
    """
    u = (
        df[u_col].astype("string")
        if u_col in df.columns
        else pd.Series(["Ux"] * len(df), index=df.index, dtype="string")
    )
    m = (
        df[m_col].astype("string")
        if m_col in df.columns
        else pd.Series(["Mx"] * len(df), index=df.index, dtype="string")
    )

    u_norm = u.fillna("Ux").map(_norm_u)
    m_norm = m.fillna("Mx").map(_norm_m)

    df[out_col] = (u_norm.astype("string") + separator + m_norm.astype("string")).astype("string")

    # rank = max(U-rank, M-rank) using base M (ignore suffix for rank)
    u_rank = u_norm.map(lambda x: _U_ORDER.get(x, 9)).astype("int64")
    m_rank = m_norm.map(lambda x: _M_ORDER.get(_base_m(x), 9)).astype("int64")
    df[out_rank_col] = u_rank.where(u_rank >= m_rank, m_rank)

    return df
