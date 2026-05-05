import pandas as pd
import pytest

from model_track.stability import RegressionPSI


def test_regression_psi_basic():
    """Test RegressionPSI basic functionality with continuous data."""
    df_ref = pd.DataFrame({"score": [10.5, 12.1, 15.3, 14.2, 11.8, 13.5]})
    df_cur = pd.DataFrame({"score": [10.2, 12.5, 15.0, 14.1, 11.9, 13.2]})  # Similar distribution
    df_drift = pd.DataFrame({"score": [20.5, 22.1, 25.3, 24.2, 21.8, 23.5]})  # Drifted

    rpsi = RegressionPSI(n_bins=3)
    rpsi.fit(df_ref, "score")

    # Transform similar
    rpsi.transform(df_cur)
    psi_similar = rpsi.get_psi()

    # Transform drifted
    rpsi.transform(df_drift)
    psi_drifted = rpsi.get_psi()

    assert psi_similar < 0.25, f"Expected stable PSI, got {psi_similar}"
    assert psi_drifted > 0.25, f"Expected unstable PSI, got {psi_drifted}"
    assert rpsi.score_col_ == "score"


def test_regression_psi_missing_column():
    """Test RegressionPSI behavior when score column is missing or not fitted."""
    rpsi = RegressionPSI(n_bins=3)
    df_cur = pd.DataFrame({"other": [1, 2, 3]})

    with pytest.raises(ValueError, match="ModelPSI must be fitted"):
        rpsi.transform(df_cur)

    df_ref = pd.DataFrame({"score": [10.5, 12.1, 15.3]})
    rpsi.fit(df_ref, "score")

    # No crash, but should handle missing score col gracefully in transform
    # Wait, the parent class PSICalculator ignores missing columns during transform.
    # But since it's the score column, it should just return empty or 0.
    res = rpsi.transform(df_cur)
    assert res.empty or "score" not in res["feature"].values
