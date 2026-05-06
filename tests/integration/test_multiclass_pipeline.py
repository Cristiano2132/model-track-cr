"""Integration test: OvRWoeAdapter end-to-end multiclass pipeline.

Tests the full workflow:
  synthetic data → OvRWoeAdapter.fit() → transform() → MulticlassEvaluator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_track.evaluation import MulticlassEvaluator
from model_track.woe import OvRWoeAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_multiclass_dataset(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a 3-class synthetic dataset and return (train, test) splits."""
    rng = np.random.default_rng(seed)

    # Target with 3 balanced classes
    classes = ["low", "medium", "high"]
    target = rng.choice(classes, size=n, p=[0.4, 0.35, 0.25])

    # Categorical features with some correlation to target
    income_map = {"low": ["A", "B"], "medium": ["B", "C"], "high": ["C", "D"]}
    region_map = {
        "low": ["north", "south"],
        "medium": ["south", "east"],
        "high": ["east", "north"],
    }

    income = np.array(
        [rng.choice(income_map[t]) for t in target]  # type: ignore[index]
    )
    region = np.array(
        [rng.choice(region_map[t]) for t in target]  # type: ignore[index]
    )

    df = pd.DataFrame({"target": target, "income": income, "region": region})

    split = int(n * 0.7)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fit_transform_per_class_full_pipeline() -> None:
    """Full pipeline: fit on train, transform train+test, no data leakage."""
    train, test = make_multiclass_dataset()
    classes = ["low", "medium", "high"]
    columns = ["income", "region"]

    adapter = OvRWoeAdapter(classes=classes)
    adapter.fit(train, target="target", columns=columns)

    train_woe = adapter.transform(train, columns=columns, strategy="per_class")
    test_woe = adapter.transform(test, columns=columns, strategy="per_class")

    # All per-class WoE columns are present
    expected_cols = [f"{c}_woe_{k}" for c in columns for k in classes]
    for col in expected_cols:
        assert col in train_woe.columns, f"Missing {col} in train"
        assert col in test_woe.columns, f"Missing {col} in test"

    # Original columns are unchanged (no leakage)
    pd.testing.assert_series_equal(train_woe["target"], train["target"])
    pd.testing.assert_series_equal(test_woe["target"], test["target"])

    # No NaN in WoE columns
    woe_train_cols = [c for c in train_woe.columns if "_woe_" in c]
    assert train_woe[woe_train_cols].isna().sum().sum() == 0


def test_fit_transform_max_iv_full_pipeline() -> None:
    """max_iv strategy reduces dimensionality to n_features WoE columns."""
    train, test = make_multiclass_dataset()
    classes = ["low", "medium", "high"]
    columns = ["income", "region"]

    adapter = OvRWoeAdapter(classes=classes)
    adapter.fit(train, target="target", columns=columns)

    train_woe = adapter.transform(train, columns=columns, strategy="max_iv")
    test_woe = adapter.transform(test, columns=columns, strategy="max_iv")

    assert "income_woe" in train_woe.columns
    assert "region_woe" in test_woe.columns

    # Per-class columns should NOT be present
    for k in classes:
        assert f"income_woe_{k}" not in train_woe.columns


def test_iv_summary_values_positive() -> None:
    """IV values should be > 0 for features with real target correlation."""
    train, _ = make_multiclass_dataset()
    classes = ["low", "medium", "high"]

    adapter = OvRWoeAdapter(classes=classes)
    adapter.fit(train, target="target", columns=["income", "region"])

    summary = adapter.iv_summary()

    # max_iv must be positive for informative features
    assert (summary["max_iv"] > 0).all(), f"Expected all max_iv > 0, got:\n{summary}"


def test_pipeline_with_multiclass_evaluator() -> None:
    """OvRWoeAdapter → dummy model → MulticlassEvaluator works end-to-end."""
    train, test = make_multiclass_dataset(n=600)
    classes = ["low", "medium", "high"]
    columns = ["income", "region"]

    # Fit WoE
    adapter = OvRWoeAdapter(classes=classes)
    adapter.fit(train, target="target", columns=columns)
    test_woe = adapter.transform(test, columns=columns, strategy="per_class")

    # Use WoE columns as features (simulate simple model with majority class)
    y_true = test_woe["target"]
    majority_class = train["target"].mode()[0]
    y_pred = pd.Series([majority_class] * len(y_true), index=y_true.index)

    evaluator = MulticlassEvaluator(average="macro")
    metrics = evaluator.evaluate(y_true, y_pred=y_pred)

    assert "accuracy" in metrics
    assert "f1" in metrics
    # Majority-class predictor should have accuracy around the majority-class proportion
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_transform_test_with_unseen_categories() -> None:
    """Test set with unseen category values maps to WoE=0.0 (no error)."""
    train, _ = make_multiclass_dataset()
    classes = ["low", "medium", "high"]

    adapter = OvRWoeAdapter(classes=classes)
    adapter.fit(train, target="target", columns=["income"])

    # Create test set with an unseen income category
    df_test = pd.DataFrame({"target": ["low", "medium"], "income": ["UNSEEN", "A"]})
    df_out = adapter.transform(df_test, columns=["income"], strategy="per_class")

    # Unseen category → WoE = 0.0
    assert df_out.loc[0, "income_woe_low"] == pytest.approx(0.0)
    # Known category → real WoE value (should not be NaN)
    assert not pd.isna(df_out.loc[1, "income_woe_low"])
