# quality_score/apply_m_quality_score_marine.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd

_SEPS = (";", ",")

_EXPLICIT_UNSPEC_TOKENS = {
    "[unspecified]",
    "[literature-unspecified]",   # dash — matches hf_schema.yaml C42 vocab
    "unspecified",
}


def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _col_as_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([math.nan] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _split_tokens(cell: Any) -> set[str]:
    if _is_nan(cell):
        return set()
    if isinstance(cell, (list, tuple, set)):
        return {str(v).strip().lower() for v in cell if not _is_nan(v) and str(v).strip()}
    s = str(cell).strip().lower()
    if not s or s == "nan":
        return set()
    for sep in _SEPS[1:]:
        s = s.replace(sep, _SEPS[0])
    return {p.strip() for p in s.split(_SEPS[0]) if p.strip()}


def _has_explicit_unspecified(tokens: set[str]) -> bool:
    return bool(tokens & _EXPLICIT_UNSPEC_TOKENS)


def _eval_bin(val: float, op: str, threshold: float) -> bool:
    if op == ">":  return val > threshold
    if op == ">=": return val >= threshold
    if op == "<":  return val < threshold
    if op == "<=": return val <= threshold
    if op == "==": return val == threshold
    return False


def _eval_bins_largest_first(bins: dict, value: Any) -> tuple[float, bool]:
    """
    FIX T1 / TC2: evaluate bins largest-threshold-first so that the most
    restrictive (highest value) condition wins before less restrictive ones.
    bins: ordered dict of {name: {when: {op, value}, penalty: float}}
    Returns (penalty, missing_flag).
    """
    if _is_nan(value):
        # unspecified — return worst penalty, no x (it's explicit)
        worst = min(float(b["penalty"]) for b in bins.values())
        return worst, False  # unspecified is documented, not missing

    try:
        fval = float(value)
    except (TypeError, ValueError):
        return min(float(b["penalty"]) for b in bins.values()), True

    # Sort bins by threshold descending so largest fires first
    def sort_key(item):
        cond = item[1].get("when", {})
        return float(cond.get("value", 0))

    sorted_bins = sorted(bins.items(), key=sort_key, reverse=True)

    for _, bin_def in sorted_bins:
        cond = bin_def.get("when", {})
        op = cond.get("op", "")
        thresh = float(cond.get("value", 0))
        if _eval_bin(fval, op, thresh):
            return float(bin_def["penalty"]), False

    # nothing matched — return worst
    return min(float(b["penalty"]) for b in bins.values()), True


def _worst_matched_penalty(tokens: set[str], mapping: dict[str, float]) -> tuple[float, bool]:
    matched = [float(mapping[t]) for t in tokens if t in mapping]
    if matched:
        return min(matched), False
    missing = not _has_explicit_unspecified(tokens)
    return min(float(v) for v in mapping.values()), missing


def calculate_m_score_marine(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Marine / probe-sensing M-score (Table 2 of Fuchs et al. 2023).

    Fixes applied vs prior schema-naive implementation:
      T1  — bin ordering: evaluate largest threshold first (tilt, T_number,
             penetration, TC number) so lower-value bins don't shadow higher ones.
      T2  — water depth: use abs(P6) because elevation is negative for offshore.
      T3  — BWT short-circuit: if C17 == [Present and corrected], water-depth
             penalty is 0 immediately without evaluating depth bins.
      TC1 — source+saturation combined: single combined penalty per paper Table 2
             (offshore in-situ/recovered/saturated, onshore lab, estimation).
      TC2 — TC number bins: same largest-first fix as T1.
    """
    m_cfg = qc_schema["m_score"]
    calc = m_cfg["calculation"]
    thr = m_cfg["thresholds"]
    missing_suffix = str(thr.get("missing_suffix", "x"))

    # --- column names from schema ---
    col_pen_depth  = calc["probe_penetration_col"]   # C6
    col_T_n        = calc["t_number_col"]             # C37
    col_elev       = calc["elevation_col"]            # P6
    col_BWT_flag   = calc["corr_BWT_flag_col"]        # C17
    col_tilt       = calc["probe_tilt_col"]           # C23
    col_tilt_flag  = calc["corr_IS_flag_col"]         # C12
    col_tc_loc     = calc["tc_location_col"]          # C42
    col_tc_src     = calc["tc_source_col"]            # C41
    col_tc_method  = calc["tc_method_col"]            # C43
    col_tc_sat     = calc["tc_saturation_col"]        # C44
    col_tc_pt      = calc["tc_pT_conditions_col"]     # C45
    col_tc_n       = calc["tc_number_col"]            # C47

    # numeric series
    pen_depth_s = _col_as_num(df, col_pen_depth)
    T_n_s       = _col_as_num(df, col_T_n)
    elev_s      = _col_as_num(df, col_elev)
    tilt_s      = _col_as_num(df, col_tilt)
    tc_n_s      = _col_as_num(df, col_tc_n)

    marine = m_cfg["marine_logic"]
    t_logic  = marine["temperature"]
    tc_logic = marine["conductivity"]
    t_blocks  = t_logic["blocks"]
    tc_blocks = tc_logic["blocks"]

    # --- Derive comparison tokens directly from schema (never hardcode vocab) ---
    # Each token is read from the schema path that owns it, then normalised
    # to lowercase so it matches _split_tokens() output.
    # If the schema vocabulary changes, the code automatically follows.

    def _norm(s: str) -> str:
        return str(s).strip().lower()

    # BWT correction flag value — water_depth.corrected_if.flag_value
    BWT_corrected_tok = _norm(
        t_blocks["water_depth"]["corrected_if"]["flag_value"]
    )

    # Tilt correction flag value — probe_tilt.corrected_if.flag_value
    tilt_corrected_tok = _norm(
        t_blocks["probe_tilt"]["corrected_if"]["flag_value"]
    )

    # Literature token for apply_only_if — number_of_conductivities.apply_only_if value
    lit_tok = _norm(
        tc_blocks["number_of_conductivities"]["apply_only_if"]["C42_tc_location"]["value"]
    )

    # Pulse technique token — first key in tc_method mapping that contains 'pulse'
    # (also present in conditional_rules; use the mapping as single source of truth)
    pulse_tok = next(
        _norm(k) for k in tc_blocks["tc_method"]["mapping"]
        if "pulse" in k.lower()
    )

    # In-situ probe source token — from conditional_rules when_all C41 value
    insitu_src = _norm(
        tc_logic["conditional_rules"][0]["when_all"]["C41_tc_source"]["value"]
    )

    # In-situ saturation token — from conditional_rules when_all C44 value
    sat_insitu_tok = _norm(
        tc_logic["conditional_rules"][0]["when_all"]["C44_tc_saturation"]["value"]
    )

    def classify(raw: float) -> str:
        if _is_nan(raw):
            return "M4"
        if raw >= float(thr["M1"]): return "M1"
        if raw >= float(thr["M2"]): return "M2"
        if raw >= float(thr["M3"]): return "M3"
        return "M4"

    # ------------------------------------------------------------------
    def apply_temperature(i) -> tuple[float, bool]:
        score = float(t_logic.get("start_value", 1.0))
        has_missing = False

        # --- 1. Penetration depth ---
        pen, miss = _eval_bins_largest_first(
            t_blocks["penetration_depth"]["bins"],
            pen_depth_s.at[i]
        )
        score += pen
        has_missing = has_missing or miss

        # --- 2. Number of T points ---
        pen, miss = _eval_bins_largest_first(
            t_blocks["number_of_temperature_points"]["bins"],
            T_n_s.at[i]
        )
        score += pen
        has_missing = has_missing or miss

        # --- 3. Water depth — FIX T2 (abs) + FIX T3 (BWT short-circuit) ---
        bwt_cell = df.at[i, col_BWT_flag] if col_BWT_flag in df.columns else None
        bwt_tokens = _split_tokens(bwt_cell)
        if BWT_corrected_tok in bwt_tokens:
            # T3: corrected for BWT → penalty = 0, no missing flag
            pass
        else:
            raw_elev = elev_s.at[i]
            # T2: water depth = abs(elevation) for offshore
            water_depth = abs(float(raw_elev)) if not _is_nan(raw_elev) else float("nan")
            pen, miss = _eval_bins_largest_first(
                t_blocks["water_depth"]["bins"],
                water_depth
            )
            score += pen
            has_missing = has_missing or miss

        # --- 4. Tilt — FIX T1 (largest-first already handled by _eval_bins_largest_first) ---
        tilt_flag_cell = df.at[i, col_tilt_flag] if col_tilt_flag in df.columns else None
        tilt_flag_tokens = _split_tokens(tilt_flag_cell)
        if tilt_corrected_tok in tilt_flag_tokens:
            # tilt corrected → penalty = 0
            pass
        else:
            pen, miss = _eval_bins_largest_first(
                t_blocks["probe_tilt"]["bins"],
                tilt_s.at[i]
            )
            score += pen
            has_missing = has_missing or miss

        return score, has_missing

    # ------------------------------------------------------------------
    # Pre-build all mapping block dicts once (normalized lowercase keys).
    # Each is read directly from the schema — no hardcoded vocab anywhere.
    _loc_map  = {_norm(k): float(v) for k, v in tc_blocks["location"]["mapping"].items()}
    # FIX 1: _src_map removed — marine has no separate source_type block (Table 2).
    # C41 (tc_source) is read per-row only for conditional_rules evaluation.
    _sat_map  = {_norm(k): float(v) for k, v in tc_blocks["saturation"]["mapping"].items()}
    _pt_map   = {_norm(k): float(v) for k, v in tc_blocks["pT_conditions"]["mapping"].items()}
    # tc_method: only estimation entries carry a penalty; pulse=0.0 and lab methods
    # are absent — both correctly score 0.0 with no missing flag (not estimation ≠ missing).
    _est_map  = {_norm(k): float(v)
                 for k, v in tc_blocks["tc_method"]["mapping"].items()
                 if "estimation" in k.lower()}

    # ------------------------------------------------------------------
    def apply_conductivity(i) -> tuple[float, bool]:
        """
        Evaluates each schema conductivity block independently in the order
        they appear in the schema, then applies conditional_rules bonuses.

        Block order (mirrors schema, Table 2):
          1. location          → _worst_matched_penalty  (every row must have a value)
          2. saturation        → _worst_matched_penalty  (covers source+sat combined)
          3. tc_method         → estimation-only lookup  (lab methods = 0.0, no x)
          4. number_of_cond.   → _eval_bins_largest_first with apply_only_if guard
          5. pT_conditions     → _worst_matched_penalty  (every row must have a value)
          6. conditional_rules → bonuses applied on top
        Note: source_type has NO separate block in marine (Table 2 has no such
        section). C41 is read per-row only for conditional_rules.
        """
        score = float(tc_logic.get("start_value", 1.0))
        has_missing = False

        loc    = _split_tokens(df.at[i, col_tc_loc]    if col_tc_loc    in df.columns else None)
        src    = _split_tokens(df.at[i, col_tc_src]    if col_tc_src    in df.columns else None)
        method = _split_tokens(df.at[i, col_tc_method] if col_tc_method in df.columns else None)
        sat    = _split_tokens(df.at[i, col_tc_sat]    if col_tc_sat    in df.columns else None)
        pt     = _split_tokens(df.at[i, col_tc_pt]     if col_tc_pt     in df.columns else None)
        nC     = tc_n_s.at[i]

        # --- Block 1: location ---
        pen, miss = _worst_matched_penalty(loc, _loc_map)
        score += pen
        has_missing = has_missing or miss

        # --- Block 2: saturation (covers "Source type and saturation" from Table 2) ---
        pen, miss = _worst_matched_penalty(sat, _sat_map)
        score += pen
        has_missing = has_missing or miss

        # --- Block 3: tc_method (water/mineral estimation entries only) ---
        # Lab methods are absent from _est_map → 0.0, no x flag.
        # [Estimation-from lithology] is also absent: per Table 2 that penalty
        # is conditional on tc_location=literature, already covered by Block 1.
        est_matched = [_est_map[t] for t in method if t in _est_map]
        if est_matched:
            score += min(est_matched)          # worst estimation penalty, no x

        # --- Block 4: number of conductivities (largest-first bins) ---
        num_block = tc_blocks["number_of_conductivities"]
        ao = num_block.get("apply_only_if", {}).get("C42_tc_location")
        if ao:
            rhs = _norm(ao.get("value", ""))
            op  = ao.get("op", "!=")
            do_apply = (rhs not in loc) if op == "!=" else (rhs in loc)
        else:
            do_apply = lit_tok not in loc
        if do_apply:
            pen, miss = _eval_bins_largest_first(num_block["bins"], nC)
            score += pen
            has_missing = has_missing or miss

        # --- Block 5: pT conditions ---
        pen, miss = _worst_matched_penalty(pt, _pt_map)
        score += pen
        has_missing = has_missing or miss

        # --- Block 6: conditional bonuses (schema conditional_rules) ---
        # These are the +0.1 bonuses for in-situ pulse and actual pT conditions.
        # They are separate from and additive on top of the block scores above.
        for rule in tc_logic.get("conditional_rules", []):
            when_all = rule.get("when_all", {})
            bonus    = float(rule.get("bonus", 0.0))
            match    = True
            for field_key, cond in when_all.items():
                op  = cond.get("op", "==")
                val = _norm(cond.get("value", ""))
                if   "tc_method"    in field_key: tokens = method
                elif "tc_source"    in field_key: tokens = src
                elif "tc_saturat"   in field_key: tokens = sat
                elif "tc_pT"        in field_key: tokens = pt
                else:                             tokens = set()
                if op == "==" and val not in tokens:
                    match = False; break
                if op == "!=" and val in tokens:
                    match = False; break
            if match:
                score += bonus

        return score, has_missing

    # ------------------------------------------------------------------
    out: list[str] = []
    for i in df.index:
        t_score,  t_miss  = apply_temperature(i)
        tc_score, tc_miss = apply_conductivity(i)

        raw  = float(t_score) * float(tc_score)
        base = classify(raw)
        out.append(f"{base}{missing_suffix}" if (t_miss or tc_miss) else base)

    return pd.Series(out, index=df.index, dtype="string")