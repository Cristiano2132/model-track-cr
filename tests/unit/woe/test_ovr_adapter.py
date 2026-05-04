"""Unit tests for OvRWoeAdapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_track.woe import OvRWoeAdapter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def three_class_df() -> pd.DataFrame:
    """Synthetic 3-class dataset with two categorical features."""
    rng = np.random.default_rng(42)
    n = 300
    target = rng.choice(["low", "medium", "high"], size=n, p=[0.4, 0.35, 0.25])
    income = rng.choice(["A", "B", "C"], size=n)
    region = rng.choice(["north", "south", "east"], size=n)
    return pd.DataFrame({"target": target, "income": income, "region": region})


@pytest.fixture()
def fitted_adapter(three_class_df: pd.DataFrame) -> OvRWoeAdapter:
    adapter = OvRWoeAdapter(classes=["low", "medium", "high"])
    adapter.fit(three_class_df, target="target", columns=["income", "region"])
    return adapter


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_empty_classes_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        OvRWoeAdapter(classes=[])


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------


def test_fit_creates_one_calculator_per_class(
    fitted_adapter: OvRWoeAdapter,
) -> None:
    assert set(fitted_adapter.calculators_.keys()) == {"low", "medium", "high"}


def test_fit_iv_per_class_keys(fitted_adapter: OvRWoeAdapter) -> None:
    assert set(fitted_adapter.iv_per_class_.keys()) == {"low", "medium", "high"}
    for cls_iv in fitted_adapter.iv_per_class_.values():
        assert set(cls_iv.keys()) == {"income", "region"}


def test_fit_iv_non_negative(fitted_adapter: OvRWoeAdapter) -> None:
    for cls_iv in fitted_adapter.iv_per_class_.values():
        for iv_val in cls_iv.values():
            assert iv_val >= 0.0


def test_fit_missing_target_raises(three_class_df: pd.DataFrame) -> None:
    adapter = OvRWoeAdapter(classes=["low", "medium", "high"])
    with pytest.raises(ValueError, match="Target column"):
        adapter.fit(three_class_df, target="nonexistent", columns=["income"])


def test_fit_empty_columns_raises(three_class_df: pd.DataFrame) -> None:
    adapter = OvRWoeAdapter(classes=["low", "medium", "high"])
    with pytest.raises(ValueError, match="must not be empty"):
        adapter.fit(three_class_df, target="target", columns=[])


# ---------------------------------------------------------------------------
# transform() — per_class
# ---------------------------------------------------------------------------


def test_transform_per_class_columns(
    fitted_adapter: OvRWoeAdapter,
    three_class_df: pd.DataFrame,
) -> None:
    df_out = fitted_adapter.transform(
        three_class_df, columns=["income", "region"], strategy="per_class"
    )
    expected_cols = [
        "income_woe_low",
        "income_woe_medium",
        "income_woe_high",
        "region_woe_low",
        "region_woe_medium",
        "region_woe_high",
    ]
    for col in expected_cols:
        assert col in df_out.columns, f"Missing column: {col}"


def test_transform_per_class_no_nan(
    fitted_adapter: OvRWoeAdapter,
    three_class_df: pd.DataFrame,
) -> None:
    df_out = fitted_adapter.transform(
        three_class_df, columns=["income", "region"], strategy="per_class"
    )
    woe_cols = [c for c in df_out.columns if "_woe_" in c]
    assert df_out[woe_cols].isna().sum().sum() == 0


def test_transform_per_class_original_cols_unchanged(
    fitted_adapter: OvRWoeAdapter,
    three_class_df: pd.DataFrame,
) -> None:
    df_out = fitted_adapter.transform(
        three_class_df, columns=["income", "region"], strategy="per_class"
    )
    pd.testing.assert_series_equal(df_out["income"], three_class_df["income"])
    pd.testing.assert_series_equal(df_out["region"], three_class_df["region"])


# ---------------------------------------------------------------------------
# transform() — max_iv
# ---------------------------------------------------------------------------


def test_transform_max_iv_columns(
    fitted_adapter: OvRWoeAdapter,
    three_class_df: pd.DataFrame,
) -> None:
    df_out = fitted_adapter.transform(
        three_class_df, columns=["income", "region"], strategy="max_iv"
    )
    assert "income_woe" in df_out.columns
    assert "region_woe" in df_out.columns
    # Should NOT have per-class columns
    assert "income_woe_low" not in df_out.columns


def test_transform_max_iv_no_nan(
    fitted_adapter: OvRWoeAdapter,
    three_class_df: pd.DataFrame,
) -> None:
    df_out = fitted_adapter.transform(
        three_class_df, columns=["income", "region"], strategy="max_iv"
    )
    woe_cols = ["income_woe", "region_woe"]
    assert df_out[woe_cols].isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# transform() — error cases
# ---------------------------------------------------------------------------


def test_unfitted_raises_on_transform(three_class_df: pd.DataFrame) -> None:
    adapter = OvRWoeAdapter(classes=["low", "medium", "high"])
    with pytest.raises(RuntimeError, match="fitted"):
        adapter.transform(three_class_df, columns=["income"])


def test_invalid_strategy_raises(
    fitted_adapter: OvRWoeAdapter,
    three_class_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Unknown strategy"):
        fitted_adapter.transform(
            three_class_df,
            columns=["income"],
            strategy="unknown",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Unseen categories
# ---------------------------------------------------------------------------


def test_unseen_categories_fillna_zero(fitted_adapter: OvRWoeAdapter) -> None:
    """Unseen category values should map to 0.0 (neutral WoE)."""
    df_new = pd.DataFrame(
        {
            "target": ["low", "medium"],
            "income": ["UNSEEN_CAT", "A"],
            "region": ["north", "UNSEEN_REGION"],
        }
    )
    df_out = fitted_adapter.transform(df_new, columns=["income", "region"], strategy="per_class")
    # Rows with unseen categories should have 0.0 WoE
    assert df_out.loc[0, "income_woe_low"] == pytest.approx(0.0)
    assert df_out.loc[1, "region_woe_low"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# iv_summary()
# ---------------------------------------------------------------------------


def test_iv_summary_shape(fitted_adapter: OvRWoeAdapter) -> None:
    summary = fitted_adapter.iv_summary()
    assert summary.index.name == "feature"
    assert list(summary.index) == ["income", "region"]
    expected_cols = {"iv_low", "iv_medium", "iv_high", "max_iv"}
    assert set(summary.columns) == expected_cols


def test_iv_summary_max_iv_is_max(fitted_adapter: OvRWoeAdapter) -> None:
    summary = fitted_adapter.iv_summary()
    iv_cols = [c for c in summary.columns if c.startswith("iv_")]
    for feature in summary.index:
        computed_max = summary.loc[feature, iv_cols].max()
        assert summary.loc[feature, "max_iv"] == pytest.approx(float(computed_max))


def test_iv_summary_unfitted_raises() -> None:
    adapter = OvRWoeAdapter(classes=["A", "B"])
    with pytest.raises(RuntimeError, match="fitted"):
        adapter.iv_summary()


# ---------------------------------------------------------------------------
# Two-class OvR is equivalent to binary WoE
# ---------------------------------------------------------------------------


def test_two_classes_per_class_produces_two_columns() -> None:
    """With 2 classes, per_class strategy gives 2 WoE columns per feature."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "target": rng.choice(["yes", "no"], size=n),
            "cat": rng.choice(["X", "Y", "Z"], size=n),
        }
    )
    adapter = OvRWoeAdapter(classes=["yes", "no"])
    adapter.fit(df, target="target", columns=["cat"])
    df_out = adapter.transform(df, columns=["cat"], strategy="per_class")
    assert "cat_woe_yes" in df_out.columns
    assert "cat_woe_no" in df_out.columns
