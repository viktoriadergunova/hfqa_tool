# quality_score/apply_u_quality_score.py
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_u_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Schema-driven U-score via coefficient of variation (uncertainty/value)*100.

    Uses:
      qc_schema["u_score"]["calculation"]["value_col"]
      qc_schema["u_score"]["calculation"]["uncertainty_col"]
      qc_schema["u_score"]["thresholds"]["U1".."U4"]

    Returns: Series of "U1","U2","U3","U4","Ux"
    """
    u_cfg = qc_schema["u_score"]["calculation"]
    t = qc_schema["u_score"]["thresholds"]

    value_col = u_cfg["value_col"]
    uncertainty_col = u_cfg["uncertainty_col"]

    # thresholds must be numeric
    U1 = float(t["U1"])
    U2 = float(t["U2"])
    U3 = float(t["U3"])


    val_num = pd.to_numeric(df[value_col], errors="coerce")
    unc_num = pd.to_numeric(df[uncertainty_col], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (unc_num / val_num.abs()) * 100.0

    conditions = [
        (cov < U1),
        (cov >= U1) & (cov < U2),
        (cov >= U2) & (cov < U3),
        (cov >= U3),
    ]
    choices = ["U1", "U2", "U3", "U4"]
    scores = np.select(conditions, choices, default="Ux")

    # treat NaN/inf (e.g., missing or value==0) as Ux
    scores = np.where(~np.isfinite(cov), "Ux", scores)

    return pd.Series(scores, index=df.index, dtype="string")
