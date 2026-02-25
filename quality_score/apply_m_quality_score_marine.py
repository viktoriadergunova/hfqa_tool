# quality_score/apply_m_quality_score_marine.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd

_SEPS = (";", ",")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _norm(s: str) -> str:
    return str(s).strip().lower()


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


# ---------------------------------------------------------------------------
# Bin evaluator (numeric threshold blocks)
# ---------------------------------------------------------------------------

def _eval_bin(val: float, op: str, threshold: float) -> bool:
    if op == ">":  return val > threshold
    if op == ">=": return val >= threshold
    if op == "<":  return val < threshold
    if op == "<=": return val <= threshold
    if op == "==": return val == threshold
    return False


def _eval_bins_largest_first(bins: dict, value: Any) -> tuple[float, bool]:
    """
    Evaluate numeric bins largest-threshold-first.

    Returns (penalty, missing_flag):
      - Value present and matched:      (penalty, False)
      - Value empty / NaN / unparseable: (worst_penalty, True)   ← largest penalty + x flag
      - Value present, nothing matched: (worst_penalty, False)   ← data issue, no x
    """
    # Missing / empty → worst penalty + x flag
    if _is_nan(value):
        if not bins:
            return 0.0, True
        worst_penalty = min(float(bin_def["penalty"]) for bin_def in bins.values())
        return worst_penalty, True

    try:
        fval = float(value)
    except (TypeError, ValueError):
        # Unparseable → treat as missing: worst penalty + x flag
        if not bins:
            return 0.0, True
        worst_penalty = min(float(bin_def["penalty"]) for bin_def in bins.values())
        return worst_penalty, True

    sorted_bins = sorted(
        bins.items(),
        key=lambda item: float(item[1].get("when", {}).get("value", 0)),
        reverse=True,
    )

    for _, bin_def in sorted_bins:
        cond   = bin_def.get("when", {})
        op     = cond.get("op", "")
        thresh = float(cond.get("value", 0))
        if _eval_bin(fval, op, thresh):
            return float(bin_def["penalty"]), False

    # present but no bin matched → conservative: worst penalty, no x
    if not bins:
        return 0.0, False
    worst_penalty = min(float(bin_def["penalty"]) for bin_def in bins.values())
    return worst_penalty, False


# ---------------------------------------------------------------------------
# Flat mapping evaluator (location block)
# ---------------------------------------------------------------------------

def _worst_matched_penalty(tokens: set[str], mapping: dict[str, float]) -> tuple[float, bool]:
    """
    Returns (penalty, missing_flag):
      - Token matched:             (worst matched penalty, False)
      - Empty tokens:              (worst_penalty, True)   ← x flag + largest penalty
      - Present but no match:      (worst_penalty, False)  ← data issue, no x
    """
    if not tokens:
        # Empty → worst penalty + x flag
        if not mapping:
            return 0.0, True
        worst = min(float(v) for v in mapping.values())
        return worst, True

    matched = [float(mapping[t]) for t in tokens if t in mapping]
    if matched:
        return min(matched), False

    # Present but unrecognised → worst penalty, no x
    if not mapping:
        return 0.0, False
    worst = min(float(v) for v in mapping.values())
    return worst, False


# ---------------------------------------------------------------------------
# Cases evaluator (saturation + pT blocks)
# ---------------------------------------------------------------------------

def _eval_cases(
    cases: list[dict],
    row_tokens: dict[str, set[str]],
) -> tuple[float, bool]:
    """
    Evaluates a list of cases with multi-field when conditions.

    Schema syntax:
      - C44: "[value]"            -> exact match required (AND across fields)
      - C44_any: ["[v1]","[v2]"] -> any of the values must be present (OR within field)

    Returns (penalty, missing_flag):
      - Case matched:                          (penalty, False)
      - Missing required field(s) or no case matched: (worst_penalty, True)  ← x flag + largest penalty
    """
    # Determine required fields from all cases
    required_fields = set()
    for case in cases:
        required_fields.update(case.get("when", {}).keys())

    # Check if any required field is completely missing/empty
    missing_required = any(
        not row_tokens.get(field.replace("_any", ""), set())
        for field in required_fields
    )

    if missing_required:
        # Missing required field(s) → worst penalty + x flag
        if not cases:
            return 0.0, True
        worst = min(float(case["penalty"]) for case in cases)
        return worst, True

    # Try to match cases
    for case in cases:
        when = case.get("when", {})
        match = True
        for field, condition in when.items():
            if field.endswith("_any"):
                col = field[:-4]
                tokens = row_tokens.get(col, set())
                values = {_norm(v) for v in condition}
                if not tokens & values:
                    match = False
                    break
            else:
                col = field
                tokens = row_tokens.get(col, set())
                val = _norm(condition)
                if val not in tokens:
                    match = False
                    break
        if match:
            return float(case["penalty"]), False

    # Present but no case matched → worst penalty, no x
    if not cases:
        return 0.0, False
    worst = min(float(case["penalty"]) for case in cases)
    return worst, False


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def calculate_m_score_marine(
    df: pd.DataFrame,
    qc_schema: dict,
    return_debug: bool = False
) -> pd.Series | tuple[pd.Series, dict]:
    """
    Marine / probe-sensing M-score (Table 2 of Fuchs et al. 2023).

    Temperature blocks:
      T1 - penetration depth         (C6,  bins largest-first)
      T2 - number of T points        (C37, bins largest-first)
      T3 - water depth               (P6,  bins; OR short-circuit if C17=[Present and corrected])
      T4 - probe tilt                (C23, bins; OR short-circuit if C12=[Tilt corrected])

    Conductivity blocks:
      TC1 - location                 (C42, flat mapping)
      TC2 - source type + saturation (C44+C43+C41, cases with AND/OR conditions)
      TC3 - number of conductivities (C47, bins largest-first; skip if C42=literature)
      TC4 - pT conditions            (C45+C43, cases with AND/OR conditions)

    Missing data policy (aligned with paper):
      - Missing/empty field          → worst penalty + x flag
      - Present, no match/unrecognized → worst penalty, no x flag
      - Matched                      → case/bin penalty, no x flag
    """
    m_cfg = qc_schema["m_score"]
    calc  = m_cfg["calculation"]
    thr   = m_cfg["thresholds"]
    missing_suffix = str(thr.get("missing_suffix", "x"))

    # --- column names resolved from schema ---
    col_pen_depth = calc["probe_penetration_col"]   # C6
    col_T_n       = calc["t_number_col"]             # C37
    col_elev      = calc["elevation_col"]            # P6
    col_BWT_flag  = calc["corr_BWT_flag_col"]        # C17
    col_tilt      = calc["probe_tilt_col"]           # C23
    col_tilt_flag = calc["corr_IS_flag_col"]         # C12
    col_tc_loc    = calc["tc_location_col"]          # C42
    col_tc_src    = calc["tc_source_col"]            # C41
    col_tc_method = calc["tc_method_col"]            # C43
    col_tc_sat    = calc["tc_saturation_col"]        # C44
    col_tc_pt     = calc["tc_pT_conditions_col"]     # C45
    col_tc_n      = calc["tc_number_col"]            # C47

    # --- numeric series ---
    pen_depth_s = _col_as_num(df, col_pen_depth)
    T_n_s       = _col_as_num(df, col_T_n)
    elev_s      = _col_as_num(df, col_elev)
    tilt_s      = _col_as_num(df, col_tilt)
    tc_n_s      = _col_as_num(df, col_tc_n)

    marine    = m_cfg["marine_logic"]
    t_logic   = marine["temperature"]
    tc_logic  = marine["conductivity"]
    t_blocks  = t_logic["blocks"]
    tc_blocks = tc_logic["blocks"]

    # --- flag tokens read from schema (never hardcoded) ---
    BWT_corrected_tok  = _norm(t_blocks["water_depth"]["corrected_if"]["flag_value"])
    tilt_corrected_tok = _norm(t_blocks["probe_tilt"]["corrected_if"]["flag_value"])

    # --- flat mapping for location block ---
    _loc_map = {_norm(k): float(v) for k, v in tc_blocks["location"]["mapping"].items()}

    def classify(raw: float) -> str:
        if _is_nan(raw):
            return "M4"
        if raw >= float(thr["M1"]): return "M1"
        if raw >= float(thr["M2"]): return "M2"
        if raw >= float(thr["M3"]): return "M3"
        return "M4"

    # ------------------------------------------------------------------
    def apply_temperature(i) -> tuple[float, bool]:
        score       = float(t_logic.get("start_value", 1.0))
        has_missing = False

        # --- T1: Penetration depth ---
        pen, miss = _eval_bins_largest_first(
            t_blocks["penetration_depth"]["bins"],
            pen_depth_s.at[i],
        )
        score += pen
        has_missing = has_missing or miss

        # --- T2: Number of T points ---
        pen, miss = _eval_bins_largest_first(
            t_blocks["number_of_temperature_points"]["bins"],
            T_n_s.at[i],
        )
        score += pen
        has_missing = has_missing or miss

        # --- T3: Water depth ---
        bwt_tokens = _split_tokens(
            df.at[i, col_BWT_flag] if col_BWT_flag in df.columns else None
        )
        if BWT_corrected_tok in bwt_tokens:
            pass  # corrected for BWT -> penalty 0, no x
        else:
            raw_elev = elev_s.at[i]
            if _is_nan(raw_elev):
                water_depth = float("nan")
            else:
                # Assume negative = depth, positive = elevation (land) → treat as shallow/unspecified
                if raw_elev < 0:
                    water_depth = abs(float(raw_elev))  # ocean depth
                else:
                    water_depth = 0.0  # positive elevation → treat as lt_1500_or_unspecified

            pen, miss = _eval_bins_largest_first(
                t_blocks["water_depth"]["bins"],
                water_depth,
            )
            score += pen
            has_missing = has_missing or miss

        # --- T4: Probe tilt ---
        tilt_flag_tokens = _split_tokens(
            df.at[i, col_tilt_flag] if col_tilt_flag in df.columns else None
        )
        if tilt_corrected_tok in tilt_flag_tokens:
            pass  # tilt corrected -> penalty 0, no x
        else:
            pen, miss = _eval_bins_largest_first(
                t_blocks["probe_tilt"]["bins"],
                tilt_s.at[i],
            )
            score += pen
            has_missing = has_missing or miss

        return score, has_missing

    # ------------------------------------------------------------------
    def apply_conductivity(i) -> tuple[float, bool]:
        score       = float(tc_logic.get("start_value", 1.0))
        has_missing = False

        def _tok(col):
            return _split_tokens(df.at[i, col] if col in df.columns else None)

        row_tokens = {
            "C41": _tok(col_tc_src),
            "C42": _tok(col_tc_loc),
            "C43": _tok(col_tc_method),
            "C44": _tok(col_tc_sat),
            "C45": _tok(col_tc_pt),
        }

        # --- TC1: location ---
        pen, miss = _worst_matched_penalty(row_tokens["C42"], _loc_map)
        score += pen
        has_missing = has_missing or miss

        # --- TC2: source type + saturation ---
        pen, miss = _eval_cases(
            tc_blocks["saturation"]["cases"],
            row_tokens,
        )
        score += pen
        has_missing = has_missing or miss

        # --- TC3: number of conductivities ---
        num_block = tc_blocks["number_of_conductivities"]
        ao = num_block.get("apply_only_if", {}).get("C42_tc_location")
        if ao:
            rhs      = _norm(ao.get("value", ""))
            op       = ao.get("op", "!=")
            do_apply = (rhs not in row_tokens["C42"]) if op == "!=" else (rhs in row_tokens["C42"])
        else:
            do_apply = True

        if do_apply:
            pen, miss = _eval_bins_largest_first(num_block["bins"], tc_n_s.at[i])
            score += pen
            has_missing = has_missing or miss

        # --- TC4: pT conditions ---
        pen, miss = _eval_cases(
            tc_blocks["pT_conditions"]["cases"],
            row_tokens,
        )
        score += pen
        has_missing = has_missing or miss

        return score, has_missing

    # ────────────────────────────────────────────────────────────────
    # Debug collection (only if requested)
    t_scores_list = [] if return_debug else None
    tc_scores_list = [] if return_debug else None
    raw_list = [] if return_debug else None
    # ────────────────────────────────────────────────────────────────

    out: list[str] = []
    for i in df.index:
        t_score,  t_miss  = apply_temperature(i)
        tc_score, tc_miss = apply_conductivity(i)

        # Collect debug values
        if return_debug:
            t_scores_list.append(t_score)
            tc_scores_list.append(tc_score)
            raw  = float(t_score) * float(tc_score)
            raw_list.append(raw)

        base = classify(raw)
        out.append(f"{base}{missing_suffix}" if (t_miss or tc_miss) else base)

    out_series = pd.Series(out, index=df.index, dtype="string")

    if return_debug:
        debug_dict = {
            "debug_t_score": pd.Series(t_scores_list, index=df.index, dtype=float),
            "debug_tc_score": pd.Series(tc_scores_list, index=df.index, dtype=float),
            "debug_raw_combined": pd.Series(raw_list, index=df.index, dtype=float),
        }
        return out_series, debug_dict
    else:
        return out_series





