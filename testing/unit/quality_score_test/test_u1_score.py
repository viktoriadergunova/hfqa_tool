import pandas as pd

from quality_score.apply_u_quality_score import calculate_u_score


def _qc_schema_u(value_col="C1", unc_col="C2", U1=5.0, U2=15.0, U3=25.0):
    return {
        "u_score": {
            "calculation": {"value_col": value_col, "uncertainty_col": unc_col},
            "thresholds": {"U1": U1, "U2": U2, "U3": U3, "U4": 100.0, "Ux": "missing"},
        }
    }


def test_u_score_schema_column_names_respected():
    schema = _qc_schema_u(value_col="VAL", unc_col="UNC", U1=5.0, U2=15.0, U3=25.0)
    df = pd.DataFrame({"VAL": [100.0], "UNC": [20.0]})  # cov=20 => U3
    assert calculate_u_score(df, schema).iloc[0] == "U3"


def test_u_score_U1_boundary():
    schema = _qc_schema_u(U1=5.0, U2=15.0, U3=25.0)
    df = pd.DataFrame({"C1": [100.0], "C2": [4.9]})  # cov=4.9 < 5 => U1
    assert calculate_u_score(df, schema).iloc[0] == "U1"


def test_u_score_U2_range():
    schema = _qc_schema_u(U1=5.0, U2=15.0, U3=25.0)
    df = pd.DataFrame({"C1": [100.0], "C2": [10.0]})  # cov=10 => U2
    assert calculate_u_score(df, schema).iloc[0] == "U2"


def test_u_score_U3_range():
    schema = _qc_schema_u(U1=5.0, U2=15.0, U3=25.0)
    df = pd.DataFrame({"C1": [100.0], "C2": [20.0]})  # cov=20 => U3
    assert calculate_u_score(df, schema).iloc[0] == "U3"


def test_u_score_U4_range():
    schema = _qc_schema_u(U1=5.0, U2=15.0, U3=25.0)
    df = pd.DataFrame({"C1": [100.0], "C2": [30.0]})  # cov=30 => U4
    assert calculate_u_score(df, schema).iloc[0] == "U4"


def test_u_score_Ux_for_zero_or_missing():
    schema = _qc_schema_u(U1=5.0, U2=15.0, U3=25.0)

    df0 = pd.DataFrame({"C1": [0.0], "C2": [10.0]})  # div by zero => inf => Ux
    assert calculate_u_score(df0, schema).iloc[0] == "Ux"

    dfm = pd.DataFrame({"C1": [None], "C2": [10.0]})  # missing => Ux
    assert calculate_u_score(dfm, schema).iloc[0] == "Ux"
