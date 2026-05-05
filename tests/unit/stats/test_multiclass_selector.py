import numpy as np
import pandas as pd
import pytest

from model_track.stats import MulticlassSelector


@pytest.fixture
def multiclass_df():
    """Create a 3-class dataset with clear feature behaviors."""
    np.random.seed(42)
    n = 1000

    # Target: A, B, C
    target = np.random.choice(["A", "B", "C"], size=n)

    # f_strong_a: High predictive power for A
    f_strong_a = np.where(target == "A", 1, 0)
    # Add noise
    f_strong_a = np.where(np.random.rand(n) < 0.2, 1 - f_strong_a, f_strong_a)

    # f_strong_b: High predictive power for B
    f_strong_b = np.where(target == "B", 1, 0)
    f_strong_b = np.where(np.random.rand(n) < 0.2, 1 - f_strong_b, f_strong_b)

    # f_weak: Noise
    f_weak = np.random.choice([0, 1], size=n)

    # f_corr_a: Correlated with f_strong_a but slightly worse
    f_corr_a = f_strong_a.copy()
    f_corr_a = np.where(np.random.rand(n) < 0.1, 1 - f_corr_a, f_corr_a)

    df = pd.DataFrame(
        {
            "target": target,
            "f_strong_a": f_strong_a.astype(str),
            "f_strong_b": f_strong_b.astype(str),
            "f_weak": f_weak.astype(str),
            "f_corr_a": f_corr_a.astype(str),
        }
    )
    return df


def test_multiclass_selector_max_strategy(multiclass_df):
    selector = MulticlassSelector(
        classes=["A", "B", "C"], iv_threshold=0.1, iv_strategy="max", cramers_threshold=0.7
    )
    selector.fit(
        multiclass_df, target="target", features=["f_strong_a", "f_strong_b", "f_weak", "f_corr_a"]
    )

    assert "f_strong_a" in selector.selected_features_
    assert "f_strong_b" in selector.selected_features_
    assert "f_weak" in selector.dropped_features_
    assert "f_corr_a" in selector.dropped_features_

    summary = selector.iv_summary()
    assert "iv_A" in summary.columns
    assert "iv_B" in summary.columns
    assert "iv_C" in summary.columns
    assert "max_iv" in summary.columns
    assert "selected" in summary.columns


def test_multiclass_selector_all_strategy(multiclass_df):
    # In this dataset, no feature is predictive for ALL classes simultaneously (OvR)
    # So "all" strategy should drop everything if threshold is high enough
    selector = MulticlassSelector(classes=["A", "B", "C"], iv_threshold=1.0, iv_strategy="all")
    selector.fit(multiclass_df, target="target", features=["f_strong_a", "f_strong_b"])

    # f_strong_a has high IV for A, but low for B/C usually
    # If it's very low for any class, it's dropped
    assert len(selector.selected_features_) < 2


def test_multiclass_selector_transform(multiclass_df):
    selector = MulticlassSelector(classes=["A", "B", "C"], iv_threshold=0.1)
    selector.fit(multiclass_df, target="target", features=["f_strong_a", "f_weak"])

    df_transformed = selector.transform(multiclass_df)
    assert "f_weak" not in df_transformed.columns
    assert "f_strong_a" in df_transformed.columns


def test_multiclass_selector_empty_features(multiclass_df):
    selector = MulticlassSelector(classes=["A", "B", "C"])
    selector.fit(multiclass_df, target="target", features=[])
    assert selector.selected_features_ == []


def test_multiclass_selector_not_fitted():
    selector = MulticlassSelector(classes=["A", "B", "C"])
    with pytest.raises(RuntimeError, match="must be fitted first"):
        selector.iv_summary()
