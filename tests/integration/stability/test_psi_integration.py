import pandas as pd

from model_track.context import ProjectContext
from model_track.stability import PSICalculator


def test_psi_context_integration():
    df_train = pd.DataFrame({"age": [20, 30, 40, 50, 60] * 20})
    df_test = pd.DataFrame({"age": [21, 31, 41, 51, 61] * 20})

    # 1. Fit and save to context
    calc = PSICalculator(n_bins=5)
    calc.fit(df_train, features=["age"])

    ctx = ProjectContext()
    calc.to_context(ctx)

    # 2. Load from context in "another session"
    new_calc = PSICalculator.from_context(ctx)
    summary = new_calc.transform(df_test)

    assert "age" in new_calc.psi_results_
    assert summary.iloc[0]["psi"] < 0.10
    assert summary.iloc[0]["status"] == "Stable"
