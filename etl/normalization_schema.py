# etl/normalization_schema.py

import copy
import pandas as pd

from etl.normalization_utils import (
    is_bracketed_token,
    normalize_bracketed_token_series,
    normalize_token,
    normalize_token_list,
)


def normalize_allowed_values(allowed_values, col_spec: dict, global_string_norm: dict) -> set:
    """
    Normalize schema allowed-values to a lowercase set.

    Uses bracketed vocabulary normalization only when:
      - column enforce_brackets is True, or
      - global enforce_brackets is True, or
      - allowed values already contain bracketed tokens

    NOTE: Returns a SET on purpose (dedupe). Ordering is enforced in normalize_schema()
    via sorted(...), to guarantee deterministic/idempotent output.
    """
    if allowed_values is None:
        return set()

    allowed_series = pd.Series(list(allowed_values), dtype="string")
    has_bracketed = any(is_bracketed_token(str(a)) for a in allowed_values if a is not None)

    col_norm = col_spec.get("normalization", {})
    if col_norm.get("enforce_brackets") is not None:
        enforce_brackets = bool(col_norm.get("enforce_brackets"))
    else:
        enforce_brackets = bool(global_string_norm.get("enforce_brackets", False))

    if enforce_brackets or has_bracketed:
        allowed_series = normalize_bracketed_token_series(allowed_series)

    return {str(a).strip().lower() for a in allowed_series.dropna()}


def normalize_schema(schema: dict) -> dict:
    """
    Normalize 'allowed' values and token-like values inside schema conditions/logic blocks.
    Supports:
      - heatflow schema (columns/core/conditions)
      - quality_score schema (m_score blocks)
      - top-level conditional_rules.yaml format (when/target/params/require)

    Returns a normalized deep copy (does not mutate the input).

    IMPORTANT:
      - To make output deterministic and idempotent, any "allowed" lists that pass
        through set() are written back as sorted(list).
    """
    schema = copy.deepcopy(schema)

    norm_cfg = schema.get("normalization", {})
    global_string_norm = norm_cfg.get("string", {})

    def _normalize_when_block(when: dict) -> None:
        """
        Normalize a {col: {op,value}} (m_score) when/apply_only_if dict.
        Only normalizes string 'value' fields and legacy string expressions.
        """
        if not isinstance(when, dict):
            return
        for wk, wv in when.items():
            if isinstance(wv, dict):
                if isinstance(wv.get("value"), str):
                    wv["value"] = normalize_token(wv["value"])
            elif isinstance(wv, str):
                when[wk] = normalize_token(wv)

    def _normalize_mapping_keys(mapping: dict) -> dict:
        """
        Normalize mapping keys:
          - bracketed tokens -> normalize_token
          - otherwise -> strip/lower
        """
        if not isinstance(mapping, dict):
            return mapping
        new_mapping = {}
        for mk, mv in mapping.items():
            if isinstance(mk, str):
                nk = normalize_token(mk) if is_bracketed_token(mk) else mk.strip().lower()
            else:
                nk = mk
            new_mapping[nk] = mv
        return new_mapping

    def _normalize_bins(bins: dict) -> None:
        """
        Normalize bins.*.when.value if it's a string token.
        """
        if not isinstance(bins, dict):
            return
        for _, b in bins.items():
            if not isinstance(b, dict):
                continue
            w = b.get("when")
            if isinstance(w, dict) and isinstance(w.get("value"), str):
                w["value"] = normalize_token(w["value"])

    def _normalize_corrected_if(ci: dict) -> None:
        """
        corrected_if:
          flag_col: "C12"
          flag_value: "[Tilt corrected]"
        Normalize flag_value and strip flag_col.
        """
        if not isinstance(ci, dict):
            return
        if ci.get("flag_col") is not None:
            ci["flag_col"] = str(ci["flag_col"]).strip()
        if isinstance(ci.get("flag_value"), str):
            ci["flag_value"] = normalize_token(ci["flag_value"])

    def _normalize_conditional_rules_when_all(rules: list) -> None:
        """
        m_score.marine_logic.conductivity.conditional_rules[*].when_all.*.(value)
        """
        if not isinstance(rules, list):
            return
        for r in rules:
            if not isinstance(r, dict):
                continue
            when_all = r.get("when_all")
            if isinstance(when_all, dict):
                _normalize_when_block(when_all)

    # -----------------------------
    # Normalize columns/core allowed (deterministic ordering!)
    # -----------------------------
    for section in ("columns", "core"):
        for _, col_spec in schema.get(section, {}).items():
            allowed = col_spec.get("allowed")
            if allowed is not None:
                col_spec["allowed"] = sorted(
                    normalize_allowed_values(allowed, col_spec, global_string_norm)
                )

    # -----------------------------
    # Normalize generic conditions (heatflow schema)
    # -----------------------------
    if "conditions" in schema:
        for cond in schema["conditions"]:
            for key in ("when", "then"):
                if key not in cond:
                    continue
                for col, val in list(cond[key].items()):
                    if isinstance(val, list):
                        cond[key][col] = [normalize_token(v) for v in val]
                    else:
                        cond[key][col] = normalize_token(val)

    # -----------------------------
    # Normalize top-level conditional_rules.yaml (rules-file format)
    # -----------------------------
    rules = schema.get("conditional_rules")
    if isinstance(rules, list):
        for r in rules:
            if not isinstance(r, dict):
                continue

            when = r.get("when")
            if isinstance(when, dict):
                if isinstance(when.get("column"), str):
                    when["column"] = when["column"].strip()
                if isinstance(when.get("mode"), str):
                    when["mode"] = when["mode"].strip().lower()

                if isinstance(when.get("value"), str):
                    when["value"] = normalize_token(when["value"])
                if isinstance(when.get("values"), list):
                    when["values"] = [normalize_token(v) for v in when["values"]]

                if isinstance(when.get("columns"), list):
                    when["columns"] = [str(c).strip() for c in when["columns"]]

            target = r.get("target")
            if isinstance(target, dict):
                if isinstance(target.get("column"), str):
                    target["column"] = target["column"].strip()
                if isinstance(target.get("allowed"), list):
                    target_allowed = [normalize_token(v) for v in target["allowed"]]
                    target["allowed"] = sorted(target_allowed)  # deterministic

            params = r.get("params")
            if isinstance(params, dict):
                for k, v in list(params.items()):
                    if k.endswith("_tokens") and isinstance(v, list):
                        params[k] = [normalize_token(x) for x in v]

            req = r.get("require")
            if isinstance(req, list):
                for item in req:
                    if not isinstance(item, dict):
                        continue
                    if isinstance(item.get("column"), str):
                        item["column"] = item["column"].strip()
                    if isinstance(item.get("mode"), str):
                        item["mode"] = item["mode"].strip().lower()
                    if isinstance(item.get("value"), str):
                        item["value"] = normalize_token(item["value"])
                    if isinstance(item.get("values"), list):
                        item["values"] = [normalize_token(x) for x in item["values"]]
            elif isinstance(req, dict):
                if isinstance(req.get("column"), str):
                    req["column"] = req["column"].strip()
                if isinstance(req.get("mode"), str):
                    req["mode"] = req["mode"].strip().lower()
                if isinstance(req.get("value"), str):
                    req["value"] = normalize_token(req["value"])
                if isinstance(req.get("values"), list):
                    req["values"] = [normalize_token(x) for x in req["values"]]

    # -----------------------------
    # Normalize hf_quality / m_score blocks (optional)
    # -----------------------------
    m_score = schema.get("m_score")
    if isinstance(m_score, dict):

        borehole_logic = m_score.get("borehole_logic")
        if isinstance(borehole_logic, dict):

            temperature = borehole_logic.get("temperature")
            if isinstance(temperature, dict):
                cases = temperature.get("cases")
                if isinstance(cases, dict):
                    for _, case in cases.items():
                        if not isinstance(case, dict):
                            continue
                        when = case.get("when")
                        if isinstance(when, dict):
                            _normalize_when_block(when)

                        rules = case.get("rules")
                        if isinstance(rules, dict):
                            for _, rule in rules.items():
                                if not isinstance(rule, dict):
                                    continue

                                for mk in ("methods_any_of", "C32_methods_any_of"):
                                    mv = rule.get(mk)
                                    if isinstance(mv, list):
                                        rule[mk] = normalize_token_list(mv)

                                for rk, rv in list(rule.items()):
                                    if isinstance(rv, str):
                                        rule[rk] = normalize_token(rv)

            conductivity = borehole_logic.get("conductivity")
            if isinstance(conductivity, dict):
                blocks = conductivity.get("blocks")
                if isinstance(blocks, dict):
                    for _, block in blocks.items():
                        if not isinstance(block, dict):
                            continue

                        if block.get("C_field") is not None:
                            block["C_field"] = str(block["C_field"]).strip()

                        apply_only_if = block.get("apply_only_if")
                        if isinstance(apply_only_if, dict):
                            _normalize_when_block(apply_only_if)

                        mapping = block.get("mapping")
                        if isinstance(mapping, dict):
                            block["mapping"] = _normalize_mapping_keys(mapping)

        marine_logic = m_score.get("marine_logic")
        if isinstance(marine_logic, dict):

            t = marine_logic.get("temperature")
            if isinstance(t, dict):
                blocks = t.get("blocks")
                if isinstance(blocks, dict):
                    for _, blk in blocks.items():
                        if not isinstance(blk, dict):
                            continue

                        if blk.get("C_field") is not None:
                            blk["C_field"] = str(blk["C_field"]).strip()

                        _normalize_corrected_if(blk.get("corrected_if"))
                        _normalize_bins(blk.get("bins"))

            c = marine_logic.get("conductivity")
            if isinstance(c, dict):
                blocks = c.get("blocks")
                if isinstance(blocks, dict):
                    for _, blk in blocks.items():
                        if not isinstance(blk, dict):
                            continue

                        if blk.get("C_field") is not None:
                            blk["C_field"] = str(blk["C_field"]).strip()

                        apply_only_if = blk.get("apply_only_if")
                        if isinstance(apply_only_if, dict):
                            _normalize_when_block(apply_only_if)

                        mapping = blk.get("mapping")
                        if isinstance(mapping, dict):
                            blk["mapping"] = _normalize_mapping_keys(mapping)

                        _normalize_bins(blk.get("bins"))
                        _normalize_corrected_if(blk.get("corrected_if"))

                _normalize_conditional_rules_when_all(c.get("conditional_rules"))

        p_flags = m_score.get("p_flags")
        if isinstance(p_flags, dict):
            fields = p_flags.get("fields")
            if isinstance(fields, dict):
                for k, v in fields.items():
                    if isinstance(v, str):
                        fields[k] = v.strip()

            letters = p_flags.get("letters")
            if isinstance(letters, dict):
                for k, v in letters.items():
                    if isinstance(v, str):
                        letters[k] = v.strip()

            encoding = p_flags.get("encoding")
            if isinstance(encoding, dict):
                new_enc = {}
                for k, v in encoding.items():
                    if isinstance(k, str):
                        nk = normalize_token(k) if k else k
                        new_enc[nk] = v
                    else:
                        new_enc[k] = v
                p_flags["encoding"] = new_enc

    return schema
