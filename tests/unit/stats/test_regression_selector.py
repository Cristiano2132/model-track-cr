import numpy as np
import pandas as pd
import pytest

from model_track.stats.regression_selection import RegressionSelector


@pytest.fixture
def clean_regression_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "target": np.random.normal(0, 1, n),
            "feat1": np.random.normal(0, 1, n),  # low correlation
            "feat2": np.random.normal(0, 1, n),  # low correlation
        }
    )
    # Make feat3 highly correlated with target
    df["feat3"] = df["target"] * 0.8 + np.random.normal(0, 0.2, n)
    # Make feat4 somewhat correlated with target
    df["feat4"] = df["target"] * 0.5 + np.random.normal(0, 0.5, n)
    return df


def test_regression_selector_basic(clean_regression_data: pd.DataFrame) -> None:
    """Test basic functionality with default parameters."""
    selector = RegressionSelector(min_correlation=0.2)

    # We expect feat1 and feat2 to drop (low correlation).
    # feat3 and feat4 have high target correlation and are not collinear enough to drop each other.
    selector.fit(clean_regression_data, target="target")

    assert "feat3" in selector.selected_features_
    assert "feat4" in selector.selected_features_

    assert "feat1" in selector.dropped_features_
    assert selector.dropped_features_["feat1"] == "low_target_correlation"


def test_regression_selector_perfect_correlation() -> None:
    """Test that perfectly correlated pairs cause one to drop."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "target": np.random.normal(0, 1, 100),
        }
    )
    # Both highly correlated with target, but perfectly correlated with each other
    df["feat_a"] = df["target"] * 0.9 + np.random.normal(0, 0.1, 100)
    df["feat_b"] = df["feat_a"] * 1.0  # Perfect correlation

    selector = RegressionSelector(correlation_threshold=0.99)
    selector.fit(df, target="target")

    # One of them should be kept, one dropped.
    assert len(selector.selected_features_) == 1
    assert "high_pair_correlation" in selector.dropped_features_.values()


def test_regression_selector_high_vif() -> None:
    """Test that high multicollinearity is caught by VIF."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    # x3 is a linear combination of x1 and x2
    x3 = x1 + x2 + np.random.normal(0, 0.01, n)

    df = pd.DataFrame(
        {"target": x1 * 0.5 + x2 * 0.5 + np.random.normal(0, 0.1, n), "x1": x1, "x2": x2, "x3": x3}
    )

    selector = RegressionSelector(vif_threshold=5.0)
    selector.fit(df, target="target")

    # x1, x2, x3 are highly multicollinear. VIF should drop at least one.
    assert len(selector.selected_features_) < 3
    assert "high_vif" in selector.dropped_features_.values()


def test_regression_selector_constant_column() -> None:
    """Test that constant columns are dropped safely."""
    df = pd.DataFrame(
        {"target": [1, 2, 3, 4, 5], "feat_const": [1, 1, 1, 1, 1], "feat_good": [1, 2, 3, 4, 5]}
    )

    selector = RegressionSelector()
    selector.fit(df, target="target")

    assert "feat_good" in selector.selected_features_
    assert "feat_const" not in selector.selected_features_
    assert selector.dropped_features_["feat_const"] == "zero_variance"


def test_regression_selector_spearman() -> None:
    """Test using spearman correlation."""
    np.random.seed(42)
    n = 100
    x = np.random.uniform(1, 10, n)
    df = pd.DataFrame(
        {
            "target": x**3,  # Non-linear relationship
            "feat_exp": np.exp(x),
        }
    )

    # Pearson might be slightly lower due to non-linearity,
    # but Spearman should be very high. We just ensure it runs without error.
    selector = RegressionSelector(method="spearman")
    selector.fit(df, target="target")

    assert "feat_exp" in selector.selected_features_


def test_regression_selector_summary(clean_regression_data: pd.DataFrame) -> None:
    """Test the summary dataframe output."""
    selector = RegressionSelector(min_correlation=0.2)
    selector.fit(clean_regression_data, target="target")

    summary_df = selector.summary()

    assert isinstance(summary_df, pd.DataFrame)
    assert list(summary_df.columns) == ["feature", "target_corr", "vif", "selected", "drop_reason"]

    # Check if feat1 (low corr) is correctly summarized
    feat1_row = summary_df[summary_df["feature"] == "feat1"].iloc[0]
    assert not feat1_row["selected"]
    assert feat1_row["drop_reason"] == "low_target_correlation"
    assert pd.isna(feat1_row["vif"])
