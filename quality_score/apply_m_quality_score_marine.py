# quality_score/apply_m_quality_score_marine.py
from __future__ import annotations

import math
from typing import Any

import pandas as pd


_SEPS = (";", ",", "|")

# Explicit tokens that already encode "unspecified" (penalty applies, but should not trigger suffix x)
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


def _col_from_condition_key(k: str) -> str:
    """
    conditional_rules may use keys like "C43_tc_method" or just "C43".
    We treat the column name as the part before the first underscore.
    """
    k = str(k).strip()
    return k.split("_", 1)[0].strip() if "_" in k else k


def calculate_m_score_marine(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Marine / probe-sensing M-score (Paper Table 2):
      - Temperature score: 4 additive blocks (penetration, T points, water depth, tilt)
      - Conductivity score: additive blocks + optional conditional bonus rules
    Missing/insufficient metadata:
      - apply max penalty for the affected block and flag 'x'
      - explicit '[Unspecified]' should NOT itself trigger 'x' (penalty already encodes it)
    """
    m_cfg = qc_schema["m_score"]
    thr = m_cfg["thresholds"]
    missing_suffix = str(thr.get("missing_suffix", "x"))

    marine_logic = m_cfg.get("marine_logic")
    if not isinstance(marine_logic, dict):
        raise ValueError("qc_schema.m_score.marine_logic missing or not a dict")

    # ---------- Columns from schema (preferred) ----------
    calc = m_cfg.get("calculation", {})
    # expected by your marine schema integration
    col_pen = calc.get("probe_penetration_col", "C6")
    col_tilt = calc.get("probe_tilt_col", "C23")
    col_elev = calc.get("elevation_col", "P6")  # used as water depth via abs()
    col_corr_is = calc.get("corr_IS_flag_col", "C12")
    col_corr_bwt = calc.get("corr_BWT_flag_col", "C17")

    col_tc_loc = calc.get("tc_location_col", "C42")
    col_tc_src = calc.get("tc_source_col", "C41")
    col_tc_meth = calc.get("tc_method_col", "C43")
    col_tc_sat = calc.get("tc_saturation_col", "C44")
    col_tc_pt = calc.get("tc_pT_conditions_col", "C45")
    col_tc_n = calc.get("tc_number_col", "C47")

    # numeric series
    pen_depth = _as_num(df[col_pen]) if col_pen in df.columns else pd.Series([math.nan] * len(df), index=df.index)
    tilt_deg = _as_num(df[col_tilt]) if col_tilt in df.columns else pd.Series([math.nan] * len(df), index=df.index)
    elev = _as_num(df[col_elev]) if col_elev in df.columns else pd.Series([math.nan] * len(df), index=df.index)
    tc_n = _as_num(df[col_tc_n]) if col_tc_n in df.columns else pd.Series([math.nan] * len(df), index=df.index)

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

    def _worst_penalty_from_mapping(mapping: dict) -> float:
        return min(float(v) for v in mapping.values())

    def _worst_penalty_from_bins(bins: dict) -> float:
        penalties: list[float] = []
        for b in bins.values():
            if isinstance(b, dict) and "penalty" in b:
                penalties.append(float(b["penalty"]))
        return min(penalties) if penalties else 0.0

    def _mapping_penalty(tokens: set[str], mapping: dict[str, float]) -> tuple[float, bool]:
        """
        Return (penalty, missing_flag)
          - if any tokens match -> min(match penalties), missing=False
          - if none match -> worst penalty, missing=True only if not explicitly unspecified
        """
        matched = [float(mapping[t]) for t in tokens if t in mapping]
        if matched:
            return min(matched), False
        missing = not _has_explicit_unspecified(tokens)
        return _worst_penalty_from_mapping(mapping), missing

    def _bins_penalty(value: Any, bins: dict) -> tuple[float, bool]:
        """
        Evaluate numeric bins (ordered). First matching bin wins.
        If missing: apply worst bin penalty and missing flag depends on whether value is explicit unspecified (not possible for numeric).
        """
        if _is_nan(value) or (isinstance(value, float) and math.isnan(value)):
            return _worst_penalty_from_bins(bins), True

        for _, b in bins.items():
            if not isinstance(b, dict):
                continue
            w = b.get("when")
            if not isinstance(w, dict):
                continue
            if _eval_when(w, value):
                return float(b.get("penalty", 0.0)), False

        # value present but no bin matches -> unresolvable -> worst + x
        return _worst_penalty_from_bins(bins), True

    def _corrected_override(i, corrected_if: dict | None) -> tuple[float | None, bool]:
        """
        If corrected_if matches, return (penalty_if_corrected, False).
        If not, return (None, False).
        """
        if not isinstance(corrected_if, dict):
            return None, False
        flag_col = str(corrected_if.get("flag_col", "")).strip()
        flag_val = corrected_if.get("flag_value")
        if not flag_col or flag_val is None:
            return None, False
        if flag_col not in df.columns:
            return None, False
        flag_tokens = _split_tokens(df.at[i, flag_col])
        if str(flag_val).strip().lower() in flag_tokens:
            return float(corrected_if.get("penalty_if_corrected", 0.0)), False
        return None, False

    # -------------------------
    # Temperature: 4 blocks
    # -------------------------
    t_logic = marine_logic.get("temperature", {})
    t_blocks = t_logic.get("blocks", {})

    def apply_temperature(i) -> tuple[float, bool]:
        score = float(t_logic.get("start_value", 1.0))
        has_missing = False

        # penetration_depth (C6)
        blk = t_blocks.get("penetration_depth", {})
        bins = blk.get("bins", {})
        pen, miss = _bins_penalty(pen_depth.at[i] if i in pen_depth.index else math.nan, bins)
        score += pen
        has_missing = has_missing or miss

        # number_of_temperature_points (C37) is numeric but you already store in C37;
        # if your schema uses C37 here, it will arrive as bins on that block.
        blk = t_blocks.get("number_of_temperature_points", {})
        bins = blk.get("bins", {})
        if "C_field" in blk and str(blk["C_field"]).strip() in df.columns:
            n_series = _as_num(df[str(blk["C_field"]).strip()])
            val = n_series.at[i]
        else:
            # fallback: try "C37" if present
            val = _as_num(df["C37"]).at[i] if "C37" in df.columns else math.nan
        pen, miss = _bins_penalty(val, bins)
        score += pen
        has_missing = has_missing or miss

        # water_depth from elevation (abs(P6)), corrected_if via C17
        blk = t_blocks.get("water_depth", {})
        corr_pen, _ = _corrected_override(i, blk.get("corrected_if"))
        if corr_pen is not None:
            score += corr_pen
        else:
            bins = blk.get("bins", {})
            wd = elev.at[i]
            wd = abs(float(wd)) if not pd.isna(wd) else math.nan
            pen, miss = _bins_penalty(wd, bins)
            score += pen
            has_missing = has_missing or miss

        # probe_tilt, corrected_if via C12
        blk = t_blocks.get("probe_tilt", {})
        corr_pen, _ = _corrected_override(i, blk.get("corrected_if"))
        if corr_pen is not None:
            score += corr_pen
        else:
            bins = blk.get("bins", {})
            pen, miss = _bins_penalty(tilt_deg.at[i] if i in tilt_deg.index else math.nan, bins)
            score += pen
            has_missing = has_missing or miss

        return score, has_missing

    # -------------------------
    # Conductivity: blocks + conditional_rules
    # -------------------------
    tc_logic = marine_logic.get("conductivity", {})
    tc_blocks = tc_logic.get("blocks", {})
    tc_cond_rules = tc_logic.get("conditional_rules", [])

    def apply_conductivity(i) -> tuple[float, bool]:
        score = float(tc_logic.get("start_value", 1.0))
        has_missing = False

        # location (C42)
        blk = tc_blocks.get("location", {})
        mapping = {str(k).strip().lower(): float(v) for k, v in blk.get("mapping", {}).items()}
        tokens = _split_tokens(df.at[i, col_tc_loc]) if col_tc_loc in df.columns else set()
        pen, miss = _mapping_penalty(tokens, mapping) if mapping else (0.0, True)
        score += pen
        has_missing = has_missing or miss

        # source_type (C41)
        blk = tc_blocks.get("source_type", {})
        mapping = {str(k).strip().lower(): float(v) for k, v in blk.get("mapping", {}).items()}
        tokens = _split_tokens(df.at[i, col_tc_src]) if col_tc_src in df.columns else set()
        pen, miss = _mapping_penalty(tokens, mapping) if mapping else (0.0, True)
        score += pen
        has_missing = has_missing or miss

        # saturation (C44)
        blk = tc_blocks.get("saturation", {})
        mapping = {str(k).strip().lower(): float(v) for k, v in blk.get("mapping", {}).items()}
        tokens = _split_tokens(df.at[i, col_tc_sat]) if col_tc_sat in df.columns else set()
        pen, miss = _mapping_penalty(tokens, mapping) if mapping else (0.0, True)
        score += pen
        has_missing = has_missing or miss

        # tc_method (C43)
        blk = tc_blocks.get("tc_method", {})
        mapping = {str(k).strip().lower(): float(v) for k, v in blk.get("mapping", {}).items()}
        tokens = _split_tokens(df.at[i, col_tc_meth]) if col_tc_meth in df.columns else set()
        pen, miss = _mapping_penalty(tokens, mapping) if mapping else (0.0, True)
        score += pen
        has_missing = has_missing or miss

        # number_of_conductivities (C47) with apply_only_if possibly
        blk = tc_blocks.get("number_of_conductivities", {})
        do_apply = True
        ao = blk.get("apply_only_if")
        if isinstance(ao, dict):
            # supports {C42_tc_location: {op,value}}
            # evaluate against raw cell value (so token splitting inside _eval_when works)
            # if there are multiple conditions, all must pass
            for k, cond in ao.items():
                col = _col_from_condition_key(k)
                cell = df.at[i, col] if col in df.columns else None
                if not _eval_when(cond, cell):
                    do_apply = False
                    break

        if do_apply:
            bins = blk.get("bins", {})
            val = tc_n.at[i] if i in tc_n.index else math.nan
            pen, miss = _bins_penalty(val, bins)
            score += pen
            # Important: "unspecified" is encoded as -0.2 in probe-sensing Table 2 for 0-1/unspecified,
            # so we don't want to add x just because it's small; but if it's actually missing/unresolvable, we do.
            has_missing = has_missing or miss

        # pT_conditions (C45)
        blk = tc_blocks.get("pT_conditions", {})
        mapping = {str(k).strip().lower(): float(v) for k, v in blk.get("mapping", {}).items()}
        tokens = _split_tokens(df.at[i, col_tc_pt]) if col_tc_pt in df.columns else set()
        pen, miss = _mapping_penalty(tokens, mapping) if mapping else (0.0, True)
        score += pen
        has_missing = has_missing or miss

        # conditional bonus rules
        if isinstance(tc_cond_rules, list):
            for r in tc_cond_rules:
                if not isinstance(r, dict):
                    continue
                when_all = r.get("when_all")
                if not isinstance(when_all, dict):
                    continue
                ok = True
                for k, cond in when_all.items():
                    col = _col_from_condition_key(k)
                    cell = df.at[i, col] if col in df.columns else None
                    if not _eval_when(cond, cell):
                        ok = False
                        break
                if ok:
                    score += float(r.get("bonus", 0.0))

        return score, has_missing

    out: list[str] = []
    for i in df.index:
        t_score, t_missing = apply_temperature(i)
        tc_score, tc_missing = apply_conductivity(i)

        raw = float(t_score) * float(tc_score)
        base = classify(raw)
        out.append(f"{base}{missing_suffix}" if (t_missing or tc_missing) else base)

    return pd.Series(out, index=df.index, dtype="string")
