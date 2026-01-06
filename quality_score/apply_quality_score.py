import pandas as pd
import numpy as np
from etl.normalization import normalize_vocabulary_series


# ----------------------------
# Shared helpers
# ----------------------------
def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _is_missing_token(x) -> bool:
    if pd.isna(x):
        return True
    x = str(x).strip()
    return x == "" or x.lower() in {"nan", "none", "null"} or x in {"[unspecified]", "[Unspecified]"}


def _worst_penalty_from_rules(rules: dict) -> float:
    penalties = []
    for _, r in rules.items():
        if isinstance(r, dict) and "penalty" in r:
            penalties.append(float(r["penalty"]))
    return min(penalties) if penalties else 0.0


def _any_method_matches(top: str, bottom: str, allowed: list[str]) -> bool:
    return (top in allowed) or (bottom in allowed)


# ----------------------------
# U-score (as you provided)
# ----------------------------
def calculate_u_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Schema-driven U-score (child computes own U; parent inherits poorest child per site).

    Required keys in qc_schema['u_score']['calculation']:
      - value_col
      - uncertainty_col
      - site_name_col
      - relevance_col
      - role_child (optional, default '[yes]')
      - role_parent (optional, default '[no]')

    Thresholds in qc_schema['u_score']['thresholds']:
      - U1, U2, U3 (U4 is implicit: >= U3)
    """
    u_cfg = qc_schema["u_score"]["calculation"]
    t = qc_schema["u_score"]["thresholds"]

    value_col = u_cfg["value_col"]
    uncertainty_col = u_cfg["uncertainty_col"]
    site_col = u_cfg["site_name_col"]
    rel_col = u_cfg["relevance_col"]

    ROLE_CHILD = u_cfg.get("role_child", "[yes]")
    ROLE_PARENT = u_cfg.get("role_parent", "[no]")

    relevance = normalize_vocabulary_series(df[rel_col])

    val_num = pd.to_numeric(df[value_col], errors="coerce")
    unc_num = pd.to_numeric(df[uncertainty_col], errors="coerce")

    # Child COV (%)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = (unc_num / val_num.abs()) * 100.0

    # Child classification
    conditions = [
        (cov < t["U1"]),
        (cov >= t["U1"]) & (cov < t["U2"]),
        (cov >= t["U2"]) & (cov < t["U3"]),
        (cov >= t["U3"]),
    ]
    choices = ["U1", "U2", "U3", "U4"]
    child_raw_scores = np.select(conditions, choices, default="Ux")

    # Temporary DF for grouping
    df_temp = df.copy()
    df_temp["temp_u"] = child_raw_scores

    # Inheritance: poorest child per site -> parent
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

    # Final assignment per row
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
    """
    Schema-driven helper to compute parent U from an already-scored child DF.

    Expects df_child to contain a column 'quality_U' (U1..U4/Ux).
    Returns a DF with [site_name_col, 'parent_quality_U'].
    """
    u_cfg = qc_schema["u_score"]["calculation"]

    site_col = u_cfg["site_name_col"]
    rel_col = u_cfg["relevance_col"]
    ROLE_CHILD = u_cfg.get("role_child", "[yes]")

    # Normalize relevance for robust matching
    relevance = normalize_vocabulary_series(df_child[rel_col])

    relevant_df = df_child[relevance == ROLE_CHILD].copy()

    if relevant_df.empty:
        return pd.DataFrame(columns=[site_col, "parent_quality_U"])

    rank_map = {"U1": 1, "U2": 2, "U3": 3, "U4": 4, "Ux": 5}
    rev_map = {v: k for k, v in rank_map.items()}

    relevant_df["u_numeric"] = relevant_df["quality_U"].map(rank_map)

    parent_scores = (
        relevant_df.groupby(site_col)["u_numeric"]
        .max()
        .reset_index()
    )

    parent_scores["parent_quality_U"] = parent_scores["u_numeric"].map(rev_map)

    return parent_scores[[site_col, "parent_quality_U"]]


# ----------------------------
# M-score (Borehole/Mine, schema-driven)
# ----------------------------
def calculate_m_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Schema-driven M-score for Borehole/Mine (Domain B).

    Requires qc_schema['m_score']['calculation'] mapping (NO hardcoded Cxx):
      - site_name_col, relevance_col, role_child, role_parent
      - t_method_top_col, t_method_bottom_col, t_number_col
      - q_top_col, q_bottom_col
      - tc_location_col, tc_source_col, tc_number_col, tc_saturation_col, tc_pT_conditions_col

    Uses qc_schema['m_score']['borehole_logic'] rules (your YAML).

    Returns per-row label: M1..M4 with optional suffix (missing_suffix) if metadata missing/unspecified.
    Parent rows inherit the poorest child per site. If the poorest rank has any missing, parent gets suffix.
    """
    m_cfg = qc_schema["m_score"]
    calc = m_cfg["calculation"]
    thr = m_cfg["thresholds"]
    missing_suffix = thr.get("missing_suffix", "x")

    site_col = calc["site_name_col"]
    rel_col = calc["relevance_col"]
    ROLE_CHILD = calc.get("role_child", "[yes]")
    ROLE_PARENT = calc.get("role_parent", "[no]")

    relevance = normalize_vocabulary_series(df[rel_col])

    # mapped columns
    col_T_top = calc["t_method_top_col"]
    col_T_bot = calc["t_method_bottom_col"]
    col_T_n = calc["t_number_col"]

    col_q_top = calc["q_top_col"]
    col_q_bot = calc["q_bottom_col"]
    col_tc_loc = calc["tc_location_col"]
    col_tc_src = calc["tc_source_col"]
    col_tc_n = calc["tc_number_col"]
    col_tc_sat = calc["tc_saturation_col"]
    col_tc_pt = calc["tc_pT_conditions_col"]

    # normalized series
    T_top = normalize_vocabulary_series(df[col_T_top])
    T_bot = normalize_vocabulary_series(df[col_T_bot])
    T_n = _as_num(df[col_T_n])

    q_top = _as_num(df[col_q_top])
    q_bot = _as_num(df[col_q_bot])

    tc_loc = normalize_vocabulary_series(df[col_tc_loc])
    tc_src = normalize_vocabulary_series(df[col_tc_src])
    tc_n = _as_num(df[col_tc_n])
    tc_sat = normalize_vocabulary_series(df[col_tc_sat])
    tc_pt = normalize_vocabulary_series(df[col_tc_pt])

    # logic
    bh = m_cfg["borehole_logic"]
    t_logic = bh["temperature"]
    tc_logic = bh["conductivity"]

    cases = t_logic["cases"]
    cont = cases["continuous_log"]
    multi = cases["multiple_single_points"]
    single = cases["one_single_point_plus_surface_T"]

    # continuous rules
    cont_eq = cont["rules"]["equilibrium_or_corrected"]
    cont_p = cont["rules"]["perturbed"]
    cont_eq_methods = cont_eq["methods_any_of"]
    cont_eq_pen = float(cont_eq["penalty"])
    cont_p_methods = cont_p["methods_any_of"]
    cont_p_pen = float(cont_p["penalty"])

    multi_rules = multi["rules"]
    single_rules = single["rules"]
    multi_worst = _worst_penalty_from_rules(multi_rules)
    single_worst = _worst_penalty_from_rules(single_rules)

    gate_fixed = float(tc_logic["gate_interval_depth_reported"]["if_missing"]["tc_score_fixed"])
    blocks = tc_logic["blocks"]
    loc_map = blocks["location"]["mapping"]
    src_map = blocks["source_type"]["mapping"]
    sat_map = blocks["saturation"]["mapping"]
    pt_map = blocks["pT_conditions"]["mapping"]

    lit_token = "[Literature/unspecified]"

    def classify(raw: float) -> str:
        if pd.isna(raw):
            return "M4"
        if raw >= thr["M1"]:
            return "M1"
        if raw >= thr["M2"]:
            return "M2"
        if raw >= thr["M3"]:
            return "M3"
        return "M4"

    out_label = pd.Series(index=df.index, dtype=object)
    out_rank = pd.Series(index=df.index, dtype="float")
    out_has_x = pd.Series(False, index=df.index, dtype=bool)

    rank_map = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
    inv_rank = {v: k for k, v in rank_map.items()}

    for i in df.index:
        if relevance.loc[i] != ROLE_CHILD:
            continue

        missing = False

        # ---- T-score
        T_score = float(t_logic.get("start_value", 1.0))
        top = T_top.loc[i]
        bot = T_bot.loc[i]
        nT = T_n.loc[i]

        if _is_missing_token(top) or _is_missing_token(bot) or pd.isna(nT):
            missing = True

        if top == "[SUR]":
            applied = False
            for rule in single_rules.values():
                if bot in rule["methods_any_of"]:
                    T_score += float(rule["penalty"])
                    applied = True
                    break
            if not applied:
                T_score += single_worst
                missing = True

        elif (not pd.isna(nT)) and (nT > 3):
            if _any_method_matches(top, bot, cont_eq_methods):
                T_score += cont_eq_pen
            elif _any_method_matches(top, bot, cont_p_methods):
                T_score += cont_p_pen
            else:
                T_score += min(cont_eq_pen, cont_p_pen)
                missing = True

        else:
            applied = False
            for rule in multi_rules.values():
                allowed = rule["methods_any_of"]
                if _any_method_matches(top, bot, allowed):
                    T_score += float(rule["penalty"])
                    applied = True
                    break
            if not applied:
                T_score += multi_worst
                missing = True

        # ---- TC-score
        TC_score = float(tc_logic.get("start_value", 1.0))

        if pd.isna(q_top.loc[i]) or pd.isna(q_bot.loc[i]):
            TC_score = gate_fixed
            missing = True
        else:
            loc = tc_loc.loc[i]
            src = tc_src.loc[i]
            nC = tc_n.loc[i]
            sat = tc_sat.loc[i]
            pt = tc_pt.loc[i]

            # location
            if _is_missing_token(loc) or loc not in loc_map:
                TC_score += min(loc_map.values())
                missing = True
            else:
                TC_score += float(loc_map[loc])

            # source
            if _is_missing_token(src) or src not in src_map:
                TC_score += min(src_map.values())
                missing = True
            else:
                TC_score += float(src_map[src])

            # number (only if not literature/unspecified)
            if loc != lit_token:
                if pd.isna(nC):
                    TC_score += -0.1
                    missing = True
                else:
                    TC_score += 0.0 if nC > 15 else -0.1

            # saturation
            if _is_missing_token(sat) or sat not in sat_map:
                TC_score += min(sat_map.values())
                missing = True
            else:
                TC_score += float(sat_map[sat])

            # pT conditions
            if _is_missing_token(pt) or pt not in pt_map:
                TC_score += min(pt_map.values())
                missing = True
            else:
                TC_score += float(pt_map[pt])

        raw = T_score * TC_score
        base = classify(raw)
        label = f"{base}{missing_suffix}" if missing else base

        out_label.loc[i] = label
        out_rank.loc[i] = rank_map[base]
        out_has_x.loc[i] = missing

    # ---- inherit to parent: poorest child per site
    site_to_parent_label = {}
    child_mask = (relevance == ROLE_CHILD)

    if child_mask.any():
        tmp = pd.DataFrame({
            site_col: df.loc[child_mask, site_col],
            "rank": out_rank.loc[child_mask],
            "has_x": out_has_x.loc[child_mask],
        }).dropna(subset=["rank"])

        if not tmp.empty:
            worst_rank_by_site = tmp.groupby(site_col)["rank"].max()
            for site, worst_rank in worst_rank_by_site.items():
                worst_rank = int(worst_rank)
                base = inv_rank[worst_rank]
                worst_children = tmp[(tmp[site_col] == site) & (tmp["rank"] == worst_rank)]
                has_x = bool(worst_children["has_x"].any())
                site_to_parent_label[site] = f"{base}{missing_suffix}" if has_x else base

    def finalize(i):
        rel = relevance.loc[i]
        if rel == ROLE_CHILD:
            return out_label.loc[i] if pd.notna(out_label.loc[i]) else f"M4{missing_suffix}"
        if rel == ROLE_PARENT:
            site = df.loc[i, site_col]
            return site_to_parent_label.get(site, f"M4{missing_suffix}")
        return f"M4{missing_suffix}"

    return pd.Series([finalize(i) for i in df.index], index=df.index)



def apply_qc_scores(df: pd.DataFrame, qc_schema: dict,
                    u_col: str = "quality_U",
                    m_col: str = "quality_M") -> pd.DataFrame:
    """
    Convenience helper: returns a copy of df with both QC score columns added.
    """
    out = df.copy()
    out[u_col] = calculate_u_score(out, qc_schema)
    out[m_col] = calculate_m_score(out, qc_schema)
    return out
