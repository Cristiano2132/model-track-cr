import numpy as np
import pandas as pd
import pytest

from model_track.stability import MulticlassPSI


def test_multiclass_psi_stable():
    # Setup identical distributions for 3 classes
    np.random.seed(42)
    df_ref = pd.DataFrame(
        {
            "proba_A": np.random.uniform(0, 1, 1000),
            "proba_B": np.random.uniform(0, 1, 1000),
            "proba_C": np.random.uniform(0, 1, 1000),
            "pred": np.random.choice(["A", "B", "C"], 1000),
        }
    )

    # Normalize probabilities
    sums = df_ref[["proba_A", "proba_B", "proba_C"]].sum(axis=1)
    for col in ["proba_A", "proba_B", "proba_C"]:
        df_ref[col] = df_ref[col] / sums

    df_curr = df_ref.copy()

    calc = MulticlassPSI(n_bins=10)
    calc.fit(df_ref, proba_cols=["proba_A", "proba_B", "proba_C"], pred_col="pred")
    summary = calc.transform(df_curr)

    assert len(summary) == 4
    for _, row in summary.iterrows():
        assert row["psi"] < 0.01
        assert row["status"] == "Stable"

    psi_dict = calc.get_psi_dict()
    assert len(psi_dict) == 4
    for key in ["proba_A", "proba_B", "proba_C", "pred"]:
        assert key in psi_dict
        assert psi_dict[key] < 0.01


def test_multiclass_psi_drift():
    # Setup drifted distributions
    np.random.seed(42)
    df_ref = pd.DataFrame(
        {
            "proba_A": np.random.uniform(0, 1, 1000),
            "proba_B": np.random.uniform(0, 1, 1000),
            "proba_C": np.random.uniform(0, 1, 1000),
            "pred": np.random.choice(["A", "B", "C"], 1000, p=[0.33, 0.33, 0.34]),
        }
    )

    # Drifted current data
    df_curr = pd.DataFrame(
        {
            "proba_A": np.random.uniform(0.5, 1, 1000),  # Shifted
            "proba_B": np.random.uniform(0, 0.5, 1000),  # Shifted
            "proba_C": np.random.uniform(0, 1, 1000),
            "pred": np.random.choice(
                ["A", "B", "C"], 1000, p=[0.8, 0.1, 0.1]
            ),  # Drifted predictions
        }
    )

    calc = MulticlassPSI(n_bins=10)
    calc.fit(df_ref, proba_cols=["proba_A", "proba_B", "proba_C"], pred_col="pred")
    summary = calc.transform(df_curr)

    assert len(summary) == 4

    # Extract results
    status_dict = summary.set_index("feature")["status"].to_dict()

    assert status_dict["proba_A"] == "Unstable"
    assert status_dict["proba_B"] == "Unstable"
    assert status_dict["pred"] == "Unstable"


def test_multiclass_psi_no_pred_col():
    df_ref = pd.DataFrame(
        {"proba_A": np.random.uniform(0, 1, 100), "proba_B": np.random.uniform(0, 1, 100)}
    )

    calc = MulticlassPSI(n_bins=10)
    calc.fit(df_ref, proba_cols=["proba_A", "proba_B"])
    summary = calc.transform(df_ref)

    assert len(summary) == 2
    assert "proba_A" in summary["feature"].values
    assert "proba_B" in summary["feature"].values


def test_multiclass_psi_transform_not_fitted():
    calc = MulticlassPSI()
    with pytest.raises(ValueError, match="MulticlassPSI must be fitted"):
        calc.transform(pd.DataFrame({"proba_A": [0.1, 0.2]}))
