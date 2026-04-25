import numpy as np
import pandas as pd

from model_track.stability import PSICalculator


def test_psi_numerical_stable():
    # Setup identical distributions
    data = np.random.normal(0, 1, 1000)
    df_ref = pd.DataFrame({"feat": data})
    df_curr = pd.DataFrame({"feat": data})

    calc = PSICalculator(n_bins=10)
    calc.fit(df_ref, features=["feat"])
    summary = calc.transform(df_curr)

    assert summary.iloc[0]["psi"] < 0.01
    assert summary.iloc[0]["status"] == "Stable"


def test_psi_numerical_drift():
    # Setup drifted distributions
    df_ref = pd.DataFrame({"feat": np.random.normal(0, 1, 1000)})
    df_curr = pd.DataFrame({"feat": np.random.normal(2, 1, 1000)})  # Shift mean

    calc = PSICalculator(n_bins=10)
    calc.fit(df_ref, features=["feat"])
    summary = calc.transform(df_curr)

    assert summary.iloc[0]["psi"] > 0.25
    assert summary.iloc[0]["status"] == "Unstable"


def test_psi_categorical_stable():
    data = ["A"] * 50 + ["B"] * 50
    df_ref = pd.DataFrame({"cat": data})
    df_curr = pd.DataFrame({"cat": data})

    calc = PSICalculator()
    calc.fit(df_ref, features=["cat"])
    summary = calc.transform(df_curr)

    assert summary.iloc[0]["psi"] < 0.01
    assert summary.iloc[0]["status"] == "Stable"


def test_psi_categorical_drift():
    df_ref = pd.DataFrame({"cat": ["A"] * 90 + ["B"] * 10})
    df_curr = pd.DataFrame({"cat": ["A"] * 10 + ["B"] * 90})

    calc = PSICalculator()
    calc.fit(df_ref, features=["cat"])
    summary = calc.transform(df_curr)

    assert summary.iloc[0]["psi"] > 0.25
    assert summary.iloc[0]["status"] == "Unstable"


def test_flag_unstable():
    calc = PSICalculator()
    calc.psi_results_ = {"f1": 0.05, "f2": 0.30}
    unstable = calc.flag_unstable(threshold=0.25)
    assert unstable == ["f2"]


def test_psi_missing_column():
    df_ref = pd.DataFrame({"f1": [1, 2, 3]})
    df_curr = pd.DataFrame({"other": [1, 2, 3]})  # f1 missing

    calc = PSICalculator()
    calc.fit(df_ref, features=["f1"])
    summary = calc.transform(df_curr)

    assert summary.empty
    assert calc.psi_results_ == {}


def test_psi_from_empty_context():
    from model_track.context import ProjectContext

    ctx = ProjectContext()
    calc = PSICalculator.from_context(ctx)
    assert calc.reference_stats_ is None


def test_psi_transform_empty_data():
    df_ref = pd.DataFrame({"f1": [1, 2, 3]})
    df_curr = pd.DataFrame({"f1": []})

    calc = PSICalculator()
    calc.fit(df_ref, features=["f1"])
    summary = calc.transform(df_curr)

    assert summary.empty
