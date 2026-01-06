import copy
import pandas as pd
from etl.normalization import normalize_vocabulary_series


def norm_vocab(x) -> str:
    """Normalize a single vocabulary token using the global vocab normalizer."""
    s = normalize_vocabulary_series(pd.Series([x], dtype="string"))
    return str(s.iloc[0])


def as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def is_missing_token(x) -> bool:
    if pd.isna(x):
        return True
    x = str(x).strip()
    return x == "" or x.lower() in {"nan", "none", "null"} or x in {
        "[unspecified]", "[unspecified ]"
    }


def any_method_matches(top: str, bottom: str, allowed: list[str]) -> bool:
    return (top in allowed) or (bottom in allowed)


def worst_penalty_from_rules(rules: dict) -> float:
    penalties = []
    for r in rules.values():
        if isinstance(r, dict) and "penalty" in r:
            penalties.append(float(r["penalty"]))
    return min(penalties) if penalties else 0.0


# =====================================================
# QC-SCHEMA NORMALIZATION (M-SCORE ONLY)
# =====================================================

def _norm_list(lst):
    return [norm_vocab(x) for x in lst]


def _norm_mapping_keys(d: dict) -> dict:
    return {norm_vocab(k): v for k, v in d.items()}


def normalize_qc_schema_for_m_score(qc_schema: dict) -> dict:
    """
    Normalize ONLY tokens used for M-score matching:
      - methods_any_of lists
      - conductivity mapping KEYS
      - special tokens in expressions ([SUR], [Literature/unspecified])

    Safe to call multiple times (deep copy).
    """
    out = copy.deepcopy(qc_schema)

    m = out.get("m_score", {})
    bh = m.get("borehole_logic", {})

    # -------------------------
    # Temperature methods
    # -------------------------
    temp = bh.get("temperature", {})
    for _, case in temp.get("cases", {}).items():
        rules = case.get("rules", {})
        for _, rule in rules.items():
            if "methods_any_of" in rule:
                rule["methods_any_of"] = _norm_list(rule["methods_any_of"])

        # normalize [SUR] inside "when" expressions
        when = case.get("when", {})
        for k, expr in list(when.items()):
            if isinstance(expr, str) and "[SUR]" in expr:
                when[k] = expr.replace("[SUR]", norm_vocab("[SUR]"))

    # -------------------------
    # Conductivity mappings
    # -------------------------
    cond = bh.get("conductivity", {})
    blocks = cond.get("blocks", {})

    for name in ["location", "source_type", "saturation", "pT_conditions"]:
        if name in blocks and "mapping" in blocks[name]:
            blocks[name]["mapping"] = _norm_mapping_keys(
                blocks[name]["mapping"]
            )

    # normalize [Literature/unspecified] in apply_only_if
    num_block = blocks.get("number_of_conductivities", {})
    apply_only_if = num_block.get("apply_only_if", {})
    for k, expr in list(apply_only_if.items()):
        if isinstance(expr, str) and "[Literature/unspecified]" in expr:
            apply_only_if[k] = expr.replace(
                "[Literature/unspecified]",
                norm_vocab("[Literature/unspecified]")
            )

    return out
