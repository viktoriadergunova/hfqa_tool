import pandas as pd
import numpy as np

from etl.normalization import normalize_vocabulary_series


def apply_conditional_rules(df: pd.DataFrame, cond_cfg: dict) -> pd.DataFrame:
    df = df.copy()
    rules = cond_cfg.get("conditional_rules", [])

    if "row_type" in df.columns:
        data_mask = df["row_type"] == "data"
    else:
        data_mask = pd.Series(True, index=df.index)

    for rule in rules:
        kind = rule.get("kind", "allowed_subset")

        if kind == "allowed_subset":
            df = _apply_allowed_subset_rule(df, data_mask, rule)

        elif kind == "required_if":
            df = _apply_required_if_rule(df, data_mask, rule)

        elif kind == "imply_category":
            df = _apply_imply_category_rule(df, data_mask, rule)

        elif kind == "pt_function_combo":
            df = _apply_pt_function_combo_rule(df, data_mask, rule)

        else:
            print(f"WARNING: Unknown conditional rule type: {kind}")

    return df


# -------------------------------------------------------------------
# Hilfsfunktionen zur Normalisierung von Tokens und Spalten
# -------------------------------------------------------------------
def _norm_token(val: str) -> str:
    """
    Einen einzelnen YAML-Token so normalisieren,
    wie es normalize_vocabulary_series mit den Daten macht.
    """
    if val is None:
        return ""
    s = str(val)
    if "[" in s and "]" in s:
        series = pd.Series([s], dtype="string")
        norm = normalize_vocabulary_series(series).iloc[0]
        return str(norm).strip().lower()
    else:
        return s.strip().lower()


def _norm_token_list(vals) -> list:
    if not vals:
        return []
    series = pd.Series(list(vals), dtype="string")
    norm = normalize_vocabulary_series(series)
    return [str(v).strip().lower() for v in norm.dropna()]


def _norm_when_series(df: pd.DataFrame, col_name: str, value_for_mode) -> pd.Series:
    """
    Liefert eine normalisierte String-Serie für "when"-Vergleiche.

    Wenn der Vergleichswert wie ein Vokabel-Token aussieht ([...]),
    wird die Spalte ebenfalls mit normalize_vocabulary_series behandelt,
    sonst nur .str.strip().str.lower().
    """
    s = df[col_name].astype("string")

    if value_for_mode is not None and "[" in str(value_for_mode) and "]" in str(value_for_mode):
        # Vokabel-Spalte → gleiche Normalisierung wie bei allowed/pt-Tokens
        s_norm = normalize_vocabulary_series(s)
        return s_norm.astype("string").str.strip().str.lower()
    else:
        # einfache String-Normalisierung
        return s.str.strip().str.lower()


# -------------------------------------------------------------------
# 1) allowed_subset  (Whitelist mit optional Multi-Choice)
# -------------------------------------------------------------------
def _apply_allowed_subset_rule(df, data_mask, rule):
    when = rule.get("when", {})
    target = rule.get("target", {})
    target_col = target.get("column")
    allowed = target.get("allowed", [])
    sep = target.get("separator", ";")
    multi = target.get("multi_choice", False)

    if target_col not in df.columns:
        return df

    # WHEN-Bedingung
    when_col = when.get("column")
    if when_col not in df.columns:
        return df

    mode = when.get("mode")
    value = when.get("value")

    # Spalte für WHEN-Seite normalisieren (inkl. Vokabel-Logik)
    col_str = _norm_when_series(df, when_col, value)
    norm_value = _norm_token(value) if value is not None else None

    if mode == "equals":
        if norm_value is None:
            return df
        cond = col_str == norm_value
    elif mode == "contains":
        if norm_value is None:
            return df
        cond = col_str.str.contains(norm_value, na=False, regex=False)
    else:
        return df

    cond &= data_mask

    # allowed-Liste mit derselben Vokabel-Logik normalisieren
    allowed_norm = _norm_token_list(allowed)
    allowed_set = set(allowed_norm)

    invalid_mask = pd.Series(False, index=df.index)

    # Zielspalte (Daten) sind global schon normalisiert; hier nur noch .str.lower
    s_target = df[target_col].astype("string")

    if multi:
        # Multi-Entry: jede Komponente muss in allowed_set liegen
        for i, cell in s_target.loc[cond].items():
            if pd.isna(cell):
                continue
            parts = [p.strip().lower() for p in str(cell).split(sep) if p.strip()]
            for p in parts:
                if p not in allowed_set:
                    invalid_mask.loc[i] = True
    else:
        invalid_mask.loc[cond] = ~s_target.loc[cond].str.strip().str.lower().isin(allowed_set)

    df[f"{target_col}__cond_{rule['name']}"] = invalid_mask
    return df


# -------------------------------------------------------------------
# 2) required_if  (macht Zielspalten bedingt mandatory)
# -------------------------------------------------------------------
def _apply_required_if_rule(df, data_mask, rule):
    when = rule.get("when", {})
    requires = rule.get("require", [])

    when_col = when.get("column")
    mode = when.get("mode")

    if when_col not in df.columns:
        return df

    # Wir normalisieren die WHEN-Spalte als Vokabel-Spalte,
    # weil hier typischerweise [Probing (...)]-Tokens stehen.
    col_str = _norm_when_series(df, when_col, None)

    if mode == "contains_any":
        vals = when.get("values", [])
        vals_norm = set(_norm_token_list(vals))

        def _contains_any(x):
            if pd.isna(x):
                return False
            xs = str(x).strip().lower()
            # Multi-Entry-Logik (falls P12 mehrere Methoden hätte)
            parts = [p.strip() for p in xs.split(";") if p.strip()]
            return any(_norm_token(p) in vals_norm for p in parts)

        wmask = col_str.apply(_contains_any)
    else:
        return df

    wmask &= data_mask

    for req in requires:
        col = req.get("column")
        if col not in df.columns:
            continue

        missing_mask = wmask & df[col].isna()
        df[f"{col}__cond_{rule['name']}"] = missing_mask

    return df


# -------------------------------------------------------------------
# 3) imply_category  (Reverse-Regeln wie C22/C23 → P12 probing)
# -------------------------------------------------------------------
def _apply_imply_category_rule(df, data_mask, rule):
    when = rule.get("when", {})
    req = rule.get("require", {})

    # when: any_column_nonempty
    cols_when = when.get("columns", [])
    if not cols_when:
        return df

    wmask = pd.Series(False, index=df.index)
    for c in cols_when:
        if c in df.columns:
            wmask |= df[c].notna()

    wmask &= data_mask

    # require: z.B. P12 contains_any(...)
    target_col = req.get("column")
    values = req.get("values", [])
    mode = req.get("mode")

    if target_col not in df.columns:
        return df

    # Ziel-Spalte ähnlich wie oben als Vokabel-Spalte normalisieren
    col_str = _norm_when_series(df, target_col, None)
    vals_norm = set(_norm_token_list(values))

    if mode == "contains_any":
        def _contains_any(x):
            if pd.isna(x):
                return False
            xs = str(x).strip().lower()
            parts = [p.strip() for p in xs.split(";") if p.strip()]
            return any(_norm_token(p) in vals_norm for p in parts)

        ok_mask = col_str.apply(_contains_any)
    else:
        return df

    df[f"{target_col}__cond_{rule['name']}"] = wmask & (~ok_mask)
    return df


# -------------------------------------------------------------------
# 4) pt_function_combo  (Fall A/B Logik für Corrected in-situ (pT))
# -------------------------------------------------------------------
def _apply_pt_function_combo_rule(df, data_mask, rule):
    when = rule.get("when", {})
    target = rule.get("target", {})
    params = rule.get("params", {})

    when_col = when.get("column")
    if when_col not in df.columns:
        return df

    value = when.get("value")
    mode = when.get("mode")

    col_str = _norm_when_series(df, when_col, value)
    norm_value = _norm_token(value) if value is not None else None

    if mode == "contains":
        if norm_value is None:
            return df
        wmask = col_str.str.contains(norm_value, na=False, regex=False)
    elif mode == "equals":
        if norm_value is None:
            return df
        wmask = col_str == norm_value
    else:
        return df

    wmask &= data_mask

    col = target.get("column")
    sep = target.get("separator", ";")
    if col not in df.columns:
        return df

    # Tokenlisten aus YAML normalisieren wie Vokabeln
    pt_tokens = set(_norm_token_list(params.get("pt_tokens", [])))
    p_tokens = set(_norm_token_list(params.get("p_tokens", [])))
    t_tokens = set(_norm_token_list(params.get("t_tokens", [])))
    generic = set(_norm_token_list(params.get("generic_tokens", [])))

    universe = pt_tokens | p_tokens | t_tokens | generic

    invalid = pd.Series(False, index=df.index)

    s_target = df[col].astype("string")

    for i, cell in s_target.loc[wmask].items():
        if pd.isna(cell):
            invalid[i] = True
            continue

        parts = [p.strip().lower() for p in str(cell).split(sep) if p.strip()]
        parts_set = set(parts)

        # 1. Alle Tokens müssen überhaupt bekannt sein
        if not parts_set <= universe:
            invalid[i] = True
            continue

        # 2. Fall A: mind. ein pT-Token ODER ein generic-Token → ok
        if parts_set & pt_tokens or parts_set & generic:
            continue  # gültig

        # 3. Fall B: keine pT/generic → dann mind. ein p- und ein T-Token
        has_p = bool(parts_set & p_tokens)
        has_t = bool(parts_set & t_tokens)

        if not (has_p and has_t):
            invalid[i] = True

    df[f"{col}__cond_{rule['name']}"] = invalid

    return df
