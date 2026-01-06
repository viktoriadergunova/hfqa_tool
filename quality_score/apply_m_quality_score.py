import pandas as pd
from quality_score.helper_functions import (
    norm_vocab, as_num, is_missing_token,
    any_method_matches, worst_penalty_from_rules
)


def calculate_m_score(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    m_cfg = qc_schema["m_score"]
    calc = m_cfg["calculation"]
    thr = m_cfg["thresholds"]
    missing_suffix = thr.get("missing_suffix", "x")

    site_col = calc["site_name_col"]
    rel_col = calc["relevance_col"]
    ROLE_CHILD = calc.get("role_child", "[yes]")
    ROLE_PARENT = calc.get("role_parent", "[no]")

    relevance = norm_vocab(df, rel_col)

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
    T_top = norm_vocab(df, col_T_top)
    T_bot = norm_vocab(df, col_T_bot)
    T_n = as_num(df[col_T_n])

    q_top = as_num(df[col_q_top])
    q_bot = as_num(df[col_q_bot])

    tc_loc = norm_vocab(df, col_tc_loc)
    tc_src = norm_vocab(df, col_tc_src)
    tc_n = as_num(df[col_tc_n])
    tc_sat = norm_vocab(df, col_tc_sat)
    tc_pt = norm_vocab(df, col_tc_pt)

    bh = m_cfg["borehole_logic"]
    t_logic = bh["temperature"]
    tc_logic = bh["conductivity"]

    cases = t_logic["cases"]
    cont = cases["continuous_log"]
    multi = cases["multiple_single_points"]
    single = cases["one_single_point_plus_surface_T"]

    cont_eq = cont["rules"]["equilibrium_or_corrected"]
    cont_p = cont["rules"]["perturbed"]
    cont_eq_methods = cont_eq["methods_any_of"]
    cont_eq_pen = float(cont_eq["penalty"])
    cont_p_methods = cont_p["methods_any_of"]
    cont_p_pen = float(cont_p["penalty"])

    multi_rules = multi["rules"]
    single_rules = single["rules"]
    multi_worst = worst_penalty_from_rules(multi_rules)
    single_worst = worst_penalty_from_rules(single_rules)

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

        # T-score
        T_score = float(t_logic.get("start_value", 1.0))
        top = T_top.loc[i]
        bot = T_bot.loc[i]
        nT = T_n.loc[i]

        if is_missing_token(top) or is_missing_token(bot) or pd.isna(nT):
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
            if any_method_matches(top, bot, cont_eq_methods):
                T_score += cont_eq_pen
            elif any_method_matches(top, bot, cont_p_methods):
                T_score += cont_p_pen
            else:
                T_score += min(cont_eq_pen, cont_p_pen)
                missing = True

        else:
            applied = False
            for rule in multi_rules.values():
                allowed = rule["methods_any_of"]
                if any_method_matches(top, bot, allowed):
                    T_score += float(rule["penalty"])
                    applied = True
                    break
            if not applied:
                T_score += multi_worst
                missing = True

        # TC-score
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

            if is_missing_token(loc) or loc not in loc_map:
                TC_score += min(loc_map.values())
                missing = True
            else:
                TC_score += float(loc_map[loc])

            if is_missing_token(src) or src not in src_map:
                TC_score += min(src_map.values())
                missing = True
            else:
                TC_score += float(src_map[src])

            if loc != lit_token:
                if pd.isna(nC):
                    TC_score += -0.1
                    missing = True
                else:
                    TC_score += 0.0 if nC > 15 else -0.1

            if is_missing_token(sat) or sat not in sat_map:
                TC_score += min(sat_map.values())
                missing = True
            else:
                TC_score += float(sat_map[sat])

            if is_missing_token(pt) or pt not in pt_map:
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

    # inherit to parent: poorest child per site
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
