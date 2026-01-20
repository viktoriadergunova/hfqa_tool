# quality_score/apply_m_quality_score.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd


_SEPS = (";", ",", "|")

_EXPLICIT_UNSPEC_TOKENS = {
    "[unspecified]",
    "[literature/unspecified]",
    "unspecified",
}


def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _split_tokens(cell: Any) -> set[str]:
    """
    Parse multi-entry cells into a set of canonical tokens.
    Assumes tokens already vocabulary-normalized upstream.
    We only do strip + lower and separator splitting.
    """
    if _is_nan(cell):
        return set()

    if isinstance(cell, (list, tuple, set)):
        return {str(v).strip().lower() for v in cell if not _is_nan(v) and str(v).strip()}

    s = str(cell).strip().lower()
    if not s or s == "nan":
        return set()

    for sep in _SEPS[1:]:
        s = s.replace(sep, _SEPS[0])
    parts = [p.strip() for p in s.split(_SEPS[0])]
    return {p for p in parts if p}


def _has_explicit_unspecified(tokens: set[str]) -> bool:
    return bool(tokens & _EXPLICIT_UNSPEC_TOKENS)


def _eval_when(cond: Any, cell_value: Any) -> bool:
    """
    Evaluate schema condition:
      - {op, value}: numeric compare if value is number; token compare if string
    No normalization performed; assumes canonical tokens.
    """
    if cond is None:
        return False
    if isinstance(cond, dict) and "op" in cond and "value" in cond:
        op = str(cond["op"]).strip()
        rhs = cond["value"]

        # numeric
        if isinstance(rhs, (int, float)) and not _is_nan(rhs):
            if _is_nan(cell_value):
                return False
            try:
                lhs = float(cell_value)
                rhsf = float(rhs)
            except Exception:
                return False
            if op == ">":
                return lhs > rhsf
            if op == ">=":
                return lhs >= rhsf
            if op == "<":
                return lhs < rhsf
            if op == "<=":
                return lhs <= rhsf
            if op == "==":
                return lhs == rhsf
            if op == "!=":
                return lhs != rhsf
            return False

        # token/string
        rhs_s = str(rhs).strip().lower()
        lhs_tokens = _split_tokens(cell_value)
        if op == "==":
            return rhs_s in lhs_tokens
        if op == "!=":
            if not lhs_tokens and _is_nan(cell_value):
                return False
            return rhs_s not in lhs_tokens
        return False

    return False


def calculate_m_score_borehole(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Borehole-only M-score, no inheritance, no vocabulary normalization here.
    Assumes df values already normalized upstream.
    Multi-entry supported via splitting on ; , |.

    Conservative behavior:
      - mapping blocks: use worst (minimum) penalty among matched tokens
      - temperature rules: use worst (most negative) penalty among matched rules
      - missing suffix ("x") only if something is truly missing/unresolvable (not merely "unspecified" tokens)
    """
    m_cfg = qc_schema["m_score"]
    calc = m_cfg["calculation"]
    thr = m_cfg["thresholds"]
    missing_suffix = str(thr.get("missing_suffix", "x"))

    # columns
    col_T_top = calc["t_method_top_col"]        # C31
    col_T_bot = calc["t_method_bottom_col"]     # C32
    col_T_n = calc["t_number_col"]              # C37

    col_q_top = calc["q_top_col"]               # C4
    col_q_bot = calc["q_bottom_col"]            # C5

    col_tc_loc = calc["tc_location_col"]        # C42
    col_tc_src = calc["tc_source_col"]          # C41
    col_tc_n = calc["tc_number_col"]            # C47
    col_tc_sat = calc["tc_saturation_col"]      # C44
    col_tc_pt = calc["tc_pT_conditions_col"]    # C45

    # numeric series
    T_n = _as_num(df[col_T_n])
    q_top = _as_num(df[col_q_top])
    q_bot = _as_num(df[col_q_bot])
    tc_n = _as_num(df[col_tc_n])

    bh = m_cfg["borehole_logic"]
    t_logic = bh["temperature"]
    tc_logic = bh["conductivity"]

    cases = t_logic["cases"]
    blocks = tc_logic["blocks"]

    # canonical tokens expected (already normalized upstream)
    sur_tok = "[sur]"
    lit_tok = "[literature/unspecified]"

    def classify(raw: float) -> str:
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return "M4"
        if raw >= float(thr["M1"]):
            return "M1"
        if raw >= float(thr["M2"]):
            return "M2"
        if raw >= float(thr["M3"]):
            return "M3"
        return "M4"

    def _worst_matched_penalty(tokens: set[str], mapping: dict[str, float]) -> tuple[float, bool]:
        """
        For mapping blocks: return (penalty, missing_flag)
        - if any tokens match: return min(penalties_of_matches), missing=False
        - if none match: return worst penalty; missing=True only if not explicitly unspecified
        """
        matched = [float(mapping[t]) for t in tokens if t in mapping]
        if not matched:
            missing = not _has_explicit_unspecified(tokens)
            return min(float(v) for v in mapping.values()), missing
        return min(matched), False

    def _worst_rule_penalty(
        top: set[str],
        bot: set[str],
        rules: dict[str, dict],
        methods_key: str = "methods_any_of",
        match_bot_only: bool = False,
    ) -> tuple[float | None, bool]:
        """
        For temperature rules:
          - if any rule matches: return worst penalty among matched rules, missing=False
          - if none matches: return None, missing=True
        """
        matched_penalties: list[float] = []
        for _, rule in rules.items():
            methods = {str(x).strip().lower() for x in rule.get(methods_key, [])}
            if not methods:
                continue
            if match_bot_only:
                if bot.intersection(methods):
                    matched_penalties.append(float(rule["penalty"]))
            else:
                if top.intersection(methods) or bot.intersection(methods):
                    matched_penalties.append(float(rule["penalty"]))

        if not matched_penalties:
            return None, True
        return min(matched_penalties), False  # most negative = worst

    def apply_temperature(i) -> tuple[float, bool]:
        score = float(t_logic.get("start_value", 1.0))
        has_missing = False

        top = _split_tokens(df.at[i, col_T_top])
        bot = _split_tokens(df.at[i, col_T_bot])
        nT = T_n.at[i]

        # -------------------------
        # Case 3: one single point + surface temperature (schema-driven)
        # -------------------------
        case3 = cases["one_single_point_plus_surface_T"]
        case3_when = case3.get("when", {}).get("C31_T_method_top")
        is_case3 = _eval_when(case3_when, df.at[i, col_T_top]) if case3_when else (sur_tok in top)

        if is_case3:
            rules = case3["rules"]
            pen, _ = _worst_rule_penalty(
                top=top,
                bot=bot,
                rules=rules,
                methods_key="C32_methods_any_of",
                match_bot_only=True,
            )
            if pen is None:
                # unresolvable -> max penalty + x
                score += min(float(r["penalty"]) for r in rules.values())
                has_missing = True
            else:
                score += float(pen)
            return score, has_missing

        # -------------------------
        # Case 1: continuous log (paper-conform)
        # Only if C37>3 AND method is actually one of the continuous-log methods.
        # -------------------------
        cont_case = cases["continuous_log"]
        cond = cont_case.get("when", {}).get("C37_T_number")
        c37_ok = _eval_when(cond, nT) if cond else ((not pd.isna(nT)) and (float(nT) > 3.0))

        cont_rules = cont_case.get("rules", {})
        cont_methods: set[str] = set()
        for rule in cont_rules.values():
            cont_methods |= {str(x).strip().lower() for x in rule.get("methods_any_of", [])}

        has_cont_method = bool((top | bot) & cont_methods)
        is_cont = bool(c37_ok and has_cont_method)

        if is_cont:
            rules = cont_case["rules"]
            pen, _ = _worst_rule_penalty(top=top, bot=bot, rules=rules, methods_key="methods_any_of", match_bot_only=False)
            if pen is None:
                score += min(float(r["penalty"]) for r in rules.values())
                has_missing = True
            else:
                score += float(pen)
            return score, has_missing

        # -------------------------
        # Case 2: multiple single T points (default)
        # -------------------------
        case2 = cases["multiple_single_T_points"]
        rules = case2["rules"]
        pen, _ = _worst_rule_penalty(top=top, bot=bot, rules=rules, methods_key="methods_any_of", match_bot_only=False)
        if pen is None:
            score += min(float(r["penalty"]) for r in rules.values())
            has_missing = True
        else:
            score += float(pen)

        return score, has_missing

    def apply_conductivity(i) -> tuple[float, bool]:
        score = float(tc_logic.get("start_value", 1.0))
        has_missing = False

        # Gate: interval depth reported?
        if pd.isna(q_top.at[i]) or pd.isna(q_bot.at[i]):
            fixed = float(tc_logic["gate_interval_depth_reported"]["if_missing"]["tc_score_fixed"])
            return fixed, True  # gate missing -> always x

        loc = _split_tokens(df.at[i, col_tc_loc])
        src = _split_tokens(df.at[i, col_tc_src])
        sat = _split_tokens(df.at[i, col_tc_sat])
        pt = _split_tokens(df.at[i, col_tc_pt])
        nC = tc_n.at[i]

        # Block 1: location
        loc_map = {str(k).strip().lower(): float(v) for k, v in blocks["location"]["mapping"].items()}
        pen, miss = _worst_matched_penalty(loc, loc_map)
        score += pen
        has_missing = has_missing or miss

        # Block 2: source_type
        src_map = {str(k).strip().lower(): float(v) for k, v in blocks["source_type"]["mapping"].items()}
        pen, miss = _worst_matched_penalty(src, src_map)
        score += pen
        has_missing = has_missing or miss

        # Block 3: number_of_conductivities (respect apply_only_if)
        num_block = blocks["number_of_conductivities"]
        ao = num_block.get("apply_only_if", {}).get("C42_tc_location")
        do_apply = _eval_when(ao, df.at[i, col_tc_loc]) if ao else (lit_tok not in loc)

        if do_apply:
            mapping = num_block["mapping"]
            if pd.isna(nC):
                # penalty already covers "unspecified_or_missing" -> do NOT force x here
                score += float(mapping["unspecified_or_missing"])
            elif float(nC) > 15.0:
                score += float(mapping[">15"])
            else:
                score += float(mapping["1-15"])

        # Block 4: saturation
        sat_map = {str(k).strip().lower(): float(v) for k, v in blocks["saturation"]["mapping"].items()}
        pen, miss = _worst_matched_penalty(sat, sat_map)
        score += pen
        has_missing = has_missing or miss

        # Block 5: pT conditions
        pt_map = {str(k).strip().lower(): float(v) for k, v in blocks["pT_conditions"]["mapping"].items()}
        pen, miss = _worst_matched_penalty(pt, pt_map)
        score += pen
        has_missing = has_missing or miss

        return score, has_missing

    out: list[str] = []
    for i in df.index:
        t_score, t_missing = apply_temperature(i)
        tc_score, tc_missing = apply_conductivity(i)

        raw = float(t_score) * float(tc_score)
        base = classify(raw)
        out.append(f"{base}{missing_suffix}" if (t_missing or tc_missing) else base)

    return pd.Series(out, index=df.index, dtype="string")
