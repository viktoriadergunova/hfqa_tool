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


def normalize_schema(schema: dict) -> dict:
    """
    Normalize 'allowed' values and enforce normalization rules in the schema.
    Returns a new, normalized schema dict.
    """
    schema = copy.deepcopy(schema)  # don't mutate original
    global_string_norm = schema.get("normalization", {})

    # -----------------------------
    # Normalize columns/core allowed
    # -----------------------------
    for section in ("columns", "core"):
        for col_name, col_spec in schema.get(section, {}).items():
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
    # hf_quality:v1 — normalize ONLY borehole_logic (no splitting on ';')
    # -----------------------------
    m_score = schema.get("m_score")
    if isinstance(m_score, dict):
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

                        # when: supports {} or {col: {op,value}} (new) or {col: "oldexpr"} (legacy)
                        when = case.get("when")
                        if isinstance(when, dict):
                            for wk, wv in when.items():
                                if isinstance(wv, dict):
                                    # op unchanged
                                    if isinstance(wv.get("value"), str):
                                        wv["value"] = normalize_token(wv["value"])
                                elif isinstance(wv, str):
                                    # legacy fallback
                                    when[wk] = normalize_token(wv)

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
                                for rk, rv in rule.items():
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

                        # C_field is a column name like "C42" -> only strip, no vocab normalization
                        if block.get("C_field") is not None:
                            block["C_field"] = str(block["C_field"]).strip()

                        # apply_only_if: supports {col: {op,value}} (new) or {col: "oldexpr"} (legacy)
                        apply_only_if = block.get("apply_only_if")
                        if isinstance(apply_only_if, dict):
                            for ak, av in apply_only_if.items():
                                if isinstance(av, dict):
                                    if isinstance(av.get("value"), str):
                                        av["value"] = normalize_token(av["value"])
                                    # op unchanged
                                elif isinstance(av, str):
                                    apply_only_if[ak] = normalize_token(av)

                        # mapping keys: normalize bracketed tokens via vocab; otherwise just strip/lower
                        mapping = block.get("mapping")
                        if isinstance(mapping, dict):
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
                            block["mapping"] = new_mapping

    return schema
