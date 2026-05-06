import numpy as np
import pandas as pd
import pytest

from model_track.stats.regression_selection import RegressionSelector


def test_regression_selector_invalid_method():
    with pytest.raises(ValueError, match="method must be 'pearson' or 'spearman'"):
        RegressionSelector(method="invalid")


def test_regression_selector_zero_var_in_target_corr():
    # If a feature becomes constant during target corr, it returns NaN
    # We can force NaN correlation by making feature have zero variance, but that's caught in Stage 0
    # However, if target has zero variance, corr is NaN for all features.
    df = pd.DataFrame({"target": [1, 1, 1, 1], "feat1": [1, 2, 3, 4], "feat2": [2, 3, 4, 5]})
    sel = RegressionSelector()
    sel.fit(df, "target")
    assert "feat1" in sel.dropped_features_
    assert sel.dropped_features_["feat1"] == "zero_variance"


def test_regression_selector_pair_correlation_empty():
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4],
            "feat1": [1, 1, 1, 1],  # dropped in stage 0
        }
    )
    sel = RegressionSelector()
    sel.fit(df, "target")
    # _filter_by_pair_correlation is called with empty list
    assert sel.selected_features_ == []


def test_regression_selector_f2_in_to_drop():
    # To hit `if f2 in to_drop: continue`, feat3 must be dropped by feat1,
    # and then feat2 must process feat3.
    df = pd.DataFrame(
        {
            "target": [1, 2, 3, 4],
            "feat1": [1, 2, 3, 4],
            "feat2": [
                4,
                3,
                2,
                1,
            ],  # Negatively correlated, will not be dropped if we use abs() ? Wait, abs() makes it 1.0!
            "feat3": [1, 2, 3, 4],
        }
    )
    # Make feat2 have low correlation with feat1 so it doesn't get dropped by feat1
    df["feat2"] = [1, 0, 1, 0]  # corr with feat1 is 0
    sel = RegressionSelector(correlation_threshold=0.5)
    sel.fit(df, "target")
    # feat1 drops feat3.
    # feat2 processes feat3, but feat3 is already in to_drop, hitting the continue line.
    assert "feat1" in sel.selected_features_
    assert "feat2" in sel.selected_features_
    assert "feat3" in sel.dropped_features_


def test_regression_selector_vif_zero_var():
    # To cover np.var(y) == 0 in VIF, we need a feature that is NOT constant globally,
    # but becomes constant when we dropna() for VIF computation, OR we just mock it.
    df = pd.DataFrame({"target": [1, 2, 3, 4], "feat1": [1, 2, 3, 4], "feat2": [1, 2, 1, 2]})
    sel = RegressionSelector()
    # Let's mock np.var inside _compute_vif_scores
    original_var = np.var

    def mock_var(*args, **kwargs):
        return 0

    import model_track.stats.regression_selection

    model_track.stats.regression_selection.np.var = mock_var
    try:
        sel.fit(df, "target")
        # Since all variances are 0, VIF becomes inf for all, and they get dropped one by one
    finally:
        model_track.stats.regression_selection.np.var = original_var


def test_regression_selector_transform():
    df = pd.DataFrame({"target": [1, 2, 3, 4], "feat1": [1, 2, 3, 4], "feat2": [1, 1, 1, 1]})
    sel = RegressionSelector()
    sel.fit(df, "target")
    transformed = sel.transform(df)
    assert "feat2" not in transformed.columns
    assert "feat1" in transformed.columns
