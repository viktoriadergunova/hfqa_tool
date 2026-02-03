import copy
import pandas as pd
from etl.normalization_utils import (
    normalize_token,
    normalize_token_list,
    normalize_allowed_values,
    normalize_bracketed_token_series,
)


def normalize_schema(schema: dict) -> dict:
    """
    Normalize 'allowed' values and enforce normalization rules in the schema.
    Returns a new, normalized schema dict.
    """
    schema = copy.deepcopy(schema)
    global_string_norm = schema.get("normalization", {})

    # -----------------------------
    # Helpers
    # -----------------------------
    def _normalize_when_block(when: dict) -> None:
        if not isinstance(when, dict):
            return
        for wk, wv in when.items():
            if isinstance(wv, dict) and isinstance(wv.get("value"), str):
                wv["value"] = normalize_token(wv["value"])
            elif isinstance(wv, str):
                when[wk] = normalize_token(wv)

    def _normalize_mapping_keys(mapping: dict) -> dict:
        if not isinstance(mapping, dict):
            return mapping
        new_mapping = {}
        for mk, mv in mapping.items():
            nk = normalize_token(mk) if isinstance(mk, str) else mk
            new_mapping[nk] = mv
        return new_mapping

    def _normalize_bins(bins: dict) -> None:
        if not isinstance(bins, dict):
            return
        for b in bins.values():
            if isinstance(b, dict):
                when = b.get("when")
                if isinstance(when, dict) and isinstance(when.get("value"), str):
                    when["value"] = normalize_token(when["value"])

    def _normalize_corrected_if(ci: dict) -> None:
        if not isinstance(ci, dict):
            return
        if ci.get("flag_col") is not None:
            ci["flag_col"] = str(ci["flag_col"]).strip()
        if isinstance(ci.get("flag_value"), str):
            ci["flag_value"] = normalize_token(ci["flag_value"])

    def _normalize_conditional_rules(rules: list) -> None:
        if not isinstance(rules, list):
            return
        for r in rules:
            when_all = r.get("when_all")
            if isinstance(when_all, dict):
                _normalize_when_block(when_all)

    # -----------------------------
    # Normalize columns/core allowed
    # -----------------------------
    for section in ("columns", "core"):
        for _, col_spec in schema.get(section, {}).items():
            allowed = col_spec.get("allowed")
            if allowed is not None:
                col_spec["allowed"] = list(
                    normalize_allowed_values(allowed, col_spec, global_string_norm)
                )

    # -----------------------------
    # Normalize generic conditions
    # -----------------------------
    if "conditions" in schema:
        for cond in schema["conditions"]:
            for key in ("when", "then"):
                if key in cond:
                    for col, val in cond[key].items():
                        if isinstance(val, list):
                            cond[key][col] = [normalize_token(v) for v in val]
                        else:
                            cond[key][col] = normalize_token(val)

    # -----------------------------
    # hf_quality:v1 — M-score logic blocks
    # -----------------------------
    m_score = schema.get("m_score")
    if isinstance(m_score, dict):

        # --- Borehole logic ---
        borehole = m_score.get("borehole_logic", {})
        if isinstance(borehole, dict):
            temp_cases = borehole.get("temperature", {}).get("cases", {})
            for case in temp_cases.values():
                if isinstance(case, dict):
                    _normalize_when_block(case.get("when", {}))
                    rules = case.get("rules", {})
                    for rule in rules.values():
                        if isinstance(rule, dict):
                            for mk in ("methods_any_of", "C32_methods_any_of"):
                                if isinstance(rule.get(mk), list):
                                    rule[mk] = normalize_token_list(rule[mk])
                            for rk, rv in list(rule.items()):
                                if isinstance(rv, str):
                                    rule[rk] = normalize_token(rv)

            cond_blocks = borehole.get("conductivity", {}).get("blocks", {})
            for block in cond_blocks.values():
                if isinstance(block, dict):
                    if "C_field" in block:
                        block["C_field"] = str(block["C_field"]).strip()
                    _normalize_when_block(block.get("apply_only_if", {}))
                    if isinstance(block.get("mapping"), dict):
                        block["mapping"] = _normalize_mapping_keys(block["mapping"])

        # --- Marine logic ---
        marine = m_score.get("marine_logic", {})
        if isinstance(marine, dict):
            temp_blocks = marine.get("temperature", {}).get("blocks", {})
            for blk in temp_blocks.values():
                if isinstance(blk, dict):
                    if "C_field" in blk:
                        blk["C_field"] = str(blk["C_field"]).strip()
                    _normalize_corrected_if(blk.get("corrected_if"))
                    _normalize_bins(blk.get("bins"))

            cond_blocks = marine.get("conductivity", {}).get("blocks", {})
            for blk in cond_blocks.values():
                if isinstance(blk, dict):
                    if "C_field" in blk:
                        blk["C_field"] = str(blk["C_field"]).strip()
                    _normalize_when_block(blk.get("apply_only_if", {}))
                    if isinstance(blk.get("mapping"), dict):
                        blk["mapping"] = _normalize_mapping_keys(blk["mapping"])
                    _normalize_bins(blk.get("bins"))
                    _normalize_corrected_if(blk.get("corrected_if"))

            _normalize_conditional_rules(marine.get("conductivity", {}).get("conditional_rules", []))

        # --- P-FLAGS ---
        p_flags = m_score.get("p_flags", {})
        if isinstance(p_flags, dict):
            for fld in ("fields", "letters"):
                d = p_flags.get(fld)
                if isinstance(d, dict):
                    for k, v in d.items():
                        if isinstance(v, str):
                            d[k] = v.strip()

            enc = p_flags.get("encoding", {})
            if isinstance(enc, dict):
                new_enc = {}
                for k, v in enc.items():
                    nk = normalize_token(k) if isinstance(k, str) else k
                    new_enc[nk] = v
                p_flags["encoding"] = new_enc

    return schema
