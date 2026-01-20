# quality_score/apply_p_flags.py
from __future__ import annotations

import pandas as pd
from etl.normalization import normalize_vocabulary_series


def calculate_p_flags(df: pd.DataFrame, qc_schema: dict) -> pd.Series:
    """
    Generate 7-character p-flag codes (paper style) from perturbation columns.
    
    Returns: Series of strings like "SxxxCxh" or "-------"
    """
    p_cfg = qc_schema.get("m_score", {}).get("p_flags", {})
    if not p_cfg:
        return pd.Series(["-" * 7] * len(df), index=df.index, dtype="string")
    
    order = p_cfg.get("order", [])
    fields = p_cfg.get("fields", {})
    letters = p_cfg.get("letters", {})
    encoding = p_cfg.get("encoding", {})
    
    if not order or not fields or not letters or not encoding:
        return pd.Series(["-" * 7] * len(df), index=df.index, dtype="string")
    
    def normalize_p_value(val) -> str:
        """
        Normalize the same way as schema keys (with vocabulary normalization).
        """
        if val is None or pd.isna(val):
            return ""
        
        s = str(val)
        if s.strip() == "":
            return ""
        
        # If it has brackets, use vocabulary normalization (same as schema)
        if "[" in s and "]" in s:
            series = pd.Series([s], dtype="string")
            norm = normalize_vocabulary_series(series).iloc[0]
            return str(norm).strip().lower()
        else:
            return s.strip().lower()
    
    def encode_flag(process_name: str, flag_value) -> str:
        """Encode a single flag position."""
        col = fields.get(process_name)
        letter = letters.get(process_name)
        
        if not col or not letter:
            return "-"
        
        # Normalize using the SAME method as schema normalization
        flag_norm = normalize_p_value(flag_value)
        
        # Match against encoding (keys already normalized in schema)
        action = encoding.get(flag_norm, "-")
        
        if action == "UPPER":
            return letter.upper()
        elif action == "LOWER":
            return letter.lower()
        elif action in ("X", "x", "-"):
            return action
        else:
            return "-"
    
    # Build p-flag strings
    p_flags = []
    for idx in df.index:
        code = ""
        for process in order:
            col = fields.get(process)
            if col and col in df.columns:
                val = df.loc[idx, col]
            else:
                val = None
            code += encode_flag(process, val)
        p_flags.append(code)
    
    return pd.Series(p_flags, index=df.index, dtype="string")