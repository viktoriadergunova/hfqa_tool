import copy
import pandas as pd
from etl.normalization import normalize_vocabulary_series


def normalize_token(val: str) -> str:
    if val is None:
        return ""
    s = str(val)
    if "[" in s and "]" in s:
        series = pd.Series([s], dtype="string")
        norm = normalize_vocabulary_series(series).iloc[0]
        return str(norm).strip().lower()
    else:
        return s.strip().lower()


def normalize_token_list(vals) -> list:
    if not vals:
        return []
    series = pd.Series(list(vals), dtype="string")
    norm = normalize_vocabulary_series(series)
    return [str(v).strip().lower() for v in norm.dropna()]


def normalize_when_series(df: pd.DataFrame, col_name: str, value_for_mode) -> pd.Series:
    s = df[col_name].astype("string")
    if value_for_mode is not None and "[" in str(value_for_mode) and "]" in str(value_for_mode):
        s_norm = normalize_vocabulary_series(s)
        return s_norm.astype("string").str.strip().str.lower()
    else:
        return s.str.strip().str.lower()


def normalize_allowed_values(allowed_values, col_spec, global_string_norm) -> set:
    if allowed_values is None:
        return set()

    allowed_series = pd.Series(allowed_values, dtype="string")
    has_bracketed = any("[" in str(a) or "]" in str(a) for a in allowed_values)
    col_norm = col_spec.get("normalization", {})
    enforce_brackets = (
        bool(col_norm.get("enforce_brackets"))
        if col_norm.get("enforce_brackets") is not None
        else bool(global_string_norm.get("enforce_brackets", False))
    )

    if enforce_brackets or has_bracketed:
        allowed_series = normalize_vocabulary_series(allowed_series)

    return {str(a).strip().lower() for a in allowed_series.dropna()}


import copy
def normalize_schema(schema: dict) -> dict:
    """
    Normalize 'allowed' values and enforce normalization rules in the schema.
    Returns a new, normalized schema dict.

    IMPORTANT:
    - For p_flags.encoding we MUST NOT call normalize_vocabulary_series().
      These tokens belong to the quality schema and should be matched by
      strict, normalized string equality (strip/lower + whitespace cleanup).
    """
    import copy
    import pandas as pd
    from etl.normalization import normalize_vocabulary_series

    schema = copy.deepcopy(schema)  # don't mutate original
    global_string_norm = schema.get("normalization", {})

    # -----------------------------
    # Helpers (local)
    # -----------------------------
    def normalize_token(val: str) -> str:
        if val is None:
            return ""
        s = str(val)
        if "[" in s and "]" in s:
            series = pd.Series([s], dtype="string")
            norm = normalize_vocabulary_series(series).iloc[0]
            return str(norm).strip().lower()
        else:
            return s.strip().lower()

    def normalize_token_list(vals) -> list:
        if not vals:
            return []
        series = pd.Series(list(vals), dtype="string")
        norm = normalize_vocabulary_series(series)
        return [str(v).strip().lower() for v in norm.dropna()]

    def normalize_when_series(df: pd.DataFrame, col_name: str, value_for_mode) -> pd.Series:
        s = df[col_name].astype("string")
        if value_for_mode is not None and "[" in str(value_for_mode) and "]" in str(value_for_mode):
            s_norm = normalize_vocabulary_series(s)
            return s_norm.astype("string").str.strip().str.lower()
        else:
            return s.str.strip().str.lower()

    def normalize_allowed_values(allowed_values, col_spec, global_string_norm) -> set:
        if allowed_values is None:
            return set()

        allowed_series = pd.Series(allowed_values, dtype="string")
        has_bracketed = any("[" in str(a) or "]" in str(a) for a in allowed_values)
        col_norm = col_spec.get("normalization", {})
        enforce_brackets = (
            bool(col_norm.get("enforce_brackets"))
            if col_norm.get("enforce_brackets") is not None
            else bool(global_string_norm.get("enforce_brackets", False))
        )

        if enforce_brackets or has_bracketed:
            allowed_series = normalize_vocabulary_series(allowed_series)

        return {str(a).strip().lower() for a in allowed_series.dropna()}

    def _normalize_when_block(when: dict) -> None:
        """
        Normalize a {col: {op,value}} (new) or {col: "legacy"} (old) when/apply_only_if dict.
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
                if "[" in mk and "]" in mk:
                    nk = normalize_token(mk)
                else:
                    nk = mk.strip().lower()
            else:
                nk = mk
            new_mapping[nk] = mv
        return new_mapping

    def _normalize_bins(bins: dict) -> None:
        """
        Normalize 'bins' structure:
          bins:
            name: { when: {op,value}, penalty: ... }
        We normalize string 'value' only (usually bins are numeric, so mostly no-op).
        """
        if not isinstance(bins, dict):
            return
        for _, b in bins.items():
            if not isinstance(b, dict):
                continue
            w = b.get("when")
            if isinstance(w, dict):
                if isinstance(w.get("value"), str):
                    w["value"] = normalize_token(w["value"])

    def _normalize_corrected_if(ci: dict) -> None:
        """
        corrected_if:
          flag_col: "C12"
          flag_value: "[Tilt corrected]"
        Normalize flag_value (token) and strip flag_col.
        """
        if not isinstance(ci, dict):
            return
        if ci.get("flag_col") is not None:
            ci["flag_col"] = str(ci["flag_col"]).strip()
        if isinstance(ci.get("flag_value"), str):
            ci["flag_value"] = normalize_token(ci["flag_value"])

    def _normalize_conditional_rules(rules: list) -> None:
        """
        conditional_rules:
          - when_all:
              C43_tc_method: {op: "==", value: "[Probe - pulse technique]"}
              ...
            bonus: 0.1
        Normalize string token values under when_all.*.value
        """
        if not isinstance(rules, list):
            return
        for r in rules:
            if not isinstance(r, dict):
                continue
            when_all = r.get("when_all")
            if isinstance(when_all, dict):
                _normalize_when_block(when_all)

    def _normalize_quality_token_key(k: str) -> str:
        """
        Quality-schema tokens must NOT be run through normalize_vocabulary_series.
        We only do deterministic string cleanup to ensure matching.
        """
        return (
            str(k)
            .replace("\u00a0", " ")
            .replace("\r", " ")
            .replace("\n", " ")
            .strip()
            .lower()
        )

    # -----------------------------
    # Normalize columns/core allowed
    # -----------------------------
    for section in ("columns", "core"):
        for _, col_spec in schema.get(section, {}).items():
            allowed = col_spec.get("allowed")
            if allowed is not None:
                col_spec["allowed"] = list(normalize_allowed_values(allowed, col_spec, global_string_norm))

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
    # hf_quality:v1 — normalize M-score logic blocks
    # -----------------------------
    m_score = schema.get("m_score")
    if isinstance(m_score, dict):

        # =================================================
        # Borehole logic
        # =================================================
        borehole_logic = m_score.get("borehole_logic")
        if isinstance(borehole_logic, dict):

            # ---- Temperature ----
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

                                # method lists
                                for mk in ("methods_any_of", "C32_methods_any_of"):
                                    mv = rule.get(mk)
                                    if isinstance(mv, list):
                                        rule[mk] = normalize_token_list(mv)

                                # other string fields inside rule (if any)
                                for rk, rv in list(rule.items()):
                                    if isinstance(rv, str):
                                        rule[rk] = normalize_token(rv)

            # ---- Conductivity ----
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

        # =================================================
        # Marine logic (probe-sensing)
        # =================================================
        marine_logic = m_score.get("marine_logic")
        if isinstance(marine_logic, dict):

            # ---- Temperature (blocks/bins + corrected_if) ----
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

            # ---- Conductivity (blocks/mappings + conditional_rules) ----
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

                _normalize_conditional_rules(c.get("conditional_rules"))

        # =================================================
        # P-FLAGS — normalize the same way as other bracketed tokens
        # =================================================
        p_flags = m_score.get("p_flags")
        if isinstance(p_flags, dict):
            # fields: just strip
            fields = p_flags.get("fields")
            if isinstance(fields, dict):
                for k, v in fields.items():
                    if isinstance(v, str):
                        fields[k] = v.strip()

            # letters: just strip
            letters = p_flags.get("letters")
            if isinstance(letters, dict):
                for k, v in letters.items():
                    if isinstance(v, str):
                        letters[k] = v.strip()

            # encoding keys: normalize like other bracketed tokens (with hyphens)
            encoding = p_flags.get("encoding")
            if isinstance(encoding, dict):
                new_enc = {}
                for k, v in encoding.items():
                    if isinstance(k, str):
                        # Use normalize_token (same as other bracketed values)
                        nk = normalize_token(k)
                        new_enc[nk] = v
                    else:
                        # Handle empty string key
                        new_enc[k] = v
                p_flags["encoding"] = new_enc

    return schema

    return schema
