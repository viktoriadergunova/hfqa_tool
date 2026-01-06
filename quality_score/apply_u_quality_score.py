import pandas as pd
import numpy as np
from quality_score.helper_functions import norm_vocab


def calculate_u_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    u_cfg = qc_schema["u_score"]["calculation"]
    t = qc_schema["u_score"]["thresholds"]

    value_col = u_cfg["value_col"]
    uncertainty_col = u_cfg["uncertainty_col"]
    site_col = u_cfg["site_name_col"]
    rel_col = u_cfg["relevance_col"]

    ROLE_CHILD = u_cfg.get("role_child", "[yes]")
    ROLE_PARENT = u_cfg.get("role_parent", "[no]")

    relevance = norm_vocab(df, rel_col)

    val_num = pd.to_numeric(df[value_col], errors="coerce")
    unc_num = pd.to_numeric(df[uncertainty_col], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (unc_num / val_num.abs()) * 100.0

    conditions = [
        (cov < t["U1"]),
        (cov >= t["U1"]) & (cov < t["U2"]),
        (cov >= t["U2"]) & (cov < t["U3"]),
        (cov >= t["U3"]),
    ]
    choices = ["U1", "U2", "U3", "U4"]
    child_raw_scores = np.select(conditions, choices, default="Ux")

    df_temp = df.copy()
    df_temp["temp_u"] = child_raw_scores

    rank_map = {"U1": 1, "U2": 2, "U3": 3, "U4": 4, "Ux": 5}
    rev_map = {v: k for k, v in rank_map.items()}

    children = df_temp[relevance == ROLE_CHILD].copy()
    if not children.empty:
        children["u_rank"] = children["temp_u"].map(rank_map)
        site_poorest = (
            children.groupby(site_col)["u_rank"]
            .max()
            .map(rev_map)
            .to_dict()
        )
    else:
        site_poorest = {}

    def finalize(idx):
        row_rel = relevance.loc[idx]
        row_site = df.loc[idx, site_col]
        if row_rel == ROLE_PARENT:
            return site_poorest.get(row_site, "Ux")
        if row_rel == ROLE_CHILD:
            return df_temp.loc[idx, "temp_u"]
        return "Ux"

    return pd.Series([finalize(i) for i in df.index], index=df.index)


def inherit_u_score_to_parent(df_child: pd.DataFrame, qc_schema: dict) -> pd.DataFrame:
    u_cfg = qc_schema["u_score"]["calculation"]

    site_col = u_cfg["site_name_col"]
    rel_col = u_cfg["relevance_col"]
    ROLE_CHILD = u_cfg.get("role_child", "[yes]")

    relevance = norm_vocab(df_child, rel_col)
    relevant_df = df_child[relevance == ROLE_CHILD].copy()

    if relevant_df.empty:
        return pd.DataFrame(columns=[site_col, "parent_quality_U"])

    rank_map = {"U1": 1, "U2": 2, "U3": 3, "U4": 4, "Ux": 5}
    rev_map = {v: k for k, v in rank_map.items()}

    relevant_df["u_numeric"] = relevant_df["quality_U"].map(rank_map)
    parent_scores = relevant_df.groupby(site_col)["u_numeric"].max().reset_index()
    parent_scores["parent_quality_U"] = parent_scores["u_numeric"].map(rev_map)

    return parent_scores[[site_col, "parent_quality_U"]]
