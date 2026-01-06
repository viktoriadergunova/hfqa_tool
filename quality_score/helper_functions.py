import pandas as pd
from etl.normalization import normalize_vocabulary_series


def as_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def is_missing_token(x) -> bool:
    if pd.isna(x):
        return True
    x = str(x).strip()
    return x == "" or x.lower() in {"nan", "none", "null"} or x in {"[unspecified]", "[Unspecified]"}


def any_method_matches(top: str, bottom: str, allowed: list[str]) -> bool:
    return (top in allowed) or (bottom in allowed)


def worst_penalty_from_rules(rules: dict) -> float:
    penalties = []
    for _, r in rules.items():
        if isinstance(r, dict) and "penalty" in r:
            penalties.append(float(r["penalty"]))
    return min(penalties) if penalties else 0.0


def norm_vocab(df: pd.DataFrame, col: str) -> pd.Series:
    # small wrapper so modules don't import normalize_vocabulary_series everywhere
    return normalize_vocabulary_series(df[col])
