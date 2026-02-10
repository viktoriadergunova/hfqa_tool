import pandas as pd
import pytest

from quality_score.apply_p_flags import calculate_p_flags

@pytest.fixture
def qc_schema_min_pflags():
    # minimal qc schema containing only p_flags
    return {
        "m_score": {
            "p_flags": {
                "order": [
                    "sedimentation",
                    "erosion",
                    "topography_bathymetry",
                    "paleoclimate",
                    "surface_bottom_water_variation",
                    "convection",
                    "heat_refraction",
                ],
                "fields": {
                    "sedimentation": "C13",
                    "erosion": "C14",
                    "topography_bathymetry": "C15",
                    "paleoclimate": "C16",
                    "surface_bottom_water_variation": "C17",
                    "convection": "C18",
                    "heat_refraction": "C19",
                },
                "letters": {
                    "sedimentation": "S",
                    "erosion": "E",
                    "topography_bathymetry": "T",
                    "paleoclimate": "P",
                    "surface_bottom_water_variation": "V",
                    "convection": "C",
                    "heat_refraction": "R",
                },

                "encoding": {
                    "[present-and-corrected]": "UPPER",
                    "[present-and-not-corrected]": "LOWER",
                    "[present-not-significant]": "X",
                    "[not-recognized]": "x",
                    "[unspecified]": "-",
                },
            }
        }
    }


def test_p_flags_happy_path_all_actions(qc_schema_min_pflags):
    df = pd.DataFrame(
        {
            "C13": ["[not-recognized]"],       # -> x
            "C14": "[present-and-not-corrected]",   # -> e
            "C15": "[present-and-corrected]",     # -> T
            "C16": "[present-not-significant]",     # -> X
            "C17": "[unspecified]",                 # -> -
            "C18": None,                            # -> -
            "C19": ""                              # -> -
        }
    )

    out = calculate_p_flags(df, qc_schema_min_pflags)
    assert out.iloc[0] == "xeTX---"  # length 7

    
def test_p_flags_normalizes_bracketed_tokens_whitespace_and_case(qc_schema_min_pflags):
    df = pd.DataFrame(
        {
            "C13": [" [PRESENT   and   corrected] "],  # should normalize and match
            "C14": ["[Present and NOT corrected]"],     # normalize and match
            "C15": ["[Unspecified]"],                   # normalize to [unspecified]
            "C16": [pd.NA],
            "C17": [None],
            "C18": [""],
            "C19": ["   "],
        }
    )

    out = calculate_p_flags(df, qc_schema_min_pflags)
    assert out.iloc[0] == "Se-----"


def test_p_flags_non_bracketed_values_do_not_match_encoding(qc_schema_min_pflags):
    df = pd.DataFrame(
        {
            "C13": ["present and corrected"],  # no brackets => won't match encoding keys
            "C14": ["present and not corrected"],
            "C15": ["present not significant"],
            "C16": ["not recognized"],
            "C17": ["unspecified"],
            "C18": [None],
            "C19": [None],
        }
    )

    out = calculate_p_flags(df, qc_schema_min_pflags)
    assert out.iloc[0] == "-------"


def test_p_flags_missing_p_flags_config_returns_all_dashes():
    df = pd.DataFrame({"C13": ["[Present and corrected]"]})
    qc_schema = {"m_score": {}}  # no p_flags
    out = calculate_p_flags(df, qc_schema)
    assert out.iloc[0] == "-------"


def test_p_flags_missing_field_column_in_df_is_dash(qc_schema_min_pflags):
    # Drop C14 column entirely -> erosion position must be "-"
    df = pd.DataFrame(
        {
            "C13": ["[Present and corrected]"],  # S
            # "C14" missing
            "C15": ["[Present not significant]"],  # X
            "C16": ["[Not recognized]"],           # x
            "C17": ["[unspecified]"],              # -
            "C18": ["[unspecified]"],              # -
            "C19": ["[unspecified]"],              # -
        }
    )

    out = calculate_p_flags(df, qc_schema_min_pflags)
    assert out.iloc[0] == "S-Xx---"
