"""Unit tests for DecisionTable."""

import numpy as np
import pandas as pd
import pytest

from model_track.evaluation import DecisionTable

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def perfect_model_df() -> pd.DataFrame:
    """Perfect model: all positives have score 1.0, all negatives 0.0."""
    return pd.DataFrame(
        {
            "target": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "proba": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


@pytest.fixture()
def realistic_df() -> pd.DataFrame:
    """Realistic scenario with spread-out scores."""
    return pd.DataFrame(
        {
            "target": [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            "proba": [0.95, 0.85, 0.70, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05],
        }
    )


@pytest.fixture()
def random_model_df() -> pd.DataFrame:
    """Random model: scores uncorrelated with target."""
    rng = np.random.default_rng(42)
    n = 200
    target = rng.integers(0, 2, size=n)
    proba = rng.uniform(0, 1, size=n)
    return pd.DataFrame({"target": target, "proba": proba})


# ---------------------------------------------------------------------------
# Tests: generate()
# ---------------------------------------------------------------------------


class TestDecisionTableGenerate:
    def test_default_capture_levels(self, realistic_df: pd.DataFrame) -> None:
        dt = DecisionTable()
        table = dt.generate(realistic_df, target="target", proba="proba")
        assert isinstance(table, pd.DataFrame)
        assert not table.empty

    def test_custom_capture_levels(self, realistic_df: pd.DataFrame) -> None:
        levels = [0.30, 0.50, 0.90]
        dt = DecisionTable(capture_levels=levels)
        table = dt.generate(realistic_df, target="target", proba="proba")
        assert len(table) <= len(levels)

    def test_output_columns(self, realistic_df: pd.DataFrame) -> None:
        dt = DecisionTable(capture_levels=[0.50])
        table = dt.generate(realistic_df, target="target", proba="proba")
        expected_cols = [
            "target_capture_pct",
            "orders_declined_pct",
            "actual_tpr_pct",
            "fnr_pct",
            "hit_rate_pct",
            "cutoff",
        ]
        assert list(table.columns) == expected_cols

    def test_tpr_plus_fnr_equals_100(self, realistic_df: pd.DataFrame) -> None:
        dt = DecisionTable()
        table = dt.generate(realistic_df, target="target", proba="proba")
        for _, row in table.iterrows():
            assert abs(row["actual_tpr_pct"] + row["fnr_pct"] - 100.0) < 0.01

    def test_perfect_model_capture(self, perfect_model_df: pd.DataFrame) -> None:
        dt = DecisionTable(capture_levels=[1.0])
        table = dt.generate(perfect_model_df, target="target", proba="proba")
        row = table.iloc[0]
        assert row["actual_tpr_pct"] == 100.0
        assert row["orders_declined_pct"] == 30.0  # 3 out of 10

    def test_perfect_model_hit_rate(self, perfect_model_df: pd.DataFrame) -> None:
        dt = DecisionTable(capture_levels=[1.0])
        table = dt.generate(perfect_model_df, target="target", proba="proba")
        row = table.iloc[0]
        assert row["hit_rate_pct"] == 100.0  # all declined are positives

    def test_capture_levels_are_monotonic(self, realistic_df: pd.DataFrame) -> None:
        dt = DecisionTable()
        table = dt.generate(realistic_df, target="target", proba="proba")
        if len(table) > 1:
            tpr_values = table["actual_tpr_pct"].tolist()
            assert tpr_values == sorted(tpr_values)

    def test_decline_rate_range(self, random_model_df: pd.DataFrame) -> None:
        dt = DecisionTable()
        table = dt.generate(random_model_df, target="target", proba="proba")
        assert (table["orders_declined_pct"] >= 0).all()
        assert (table["orders_declined_pct"] <= 100).all()

    def test_cutoff_range(self, random_model_df: pd.DataFrame) -> None:
        dt = DecisionTable()
        table = dt.generate(random_model_df, target="target", proba="proba")
        assert (table["cutoff"] >= 0).all()
        assert (table["cutoff"] <= 1).all()


# ---------------------------------------------------------------------------
# Tests: cutoff_for_capture() and decline_rate_for_capture()
# ---------------------------------------------------------------------------


class TestDecisionTableInterpolation:
    def test_cutoff_for_capture_returns_float(self, realistic_df: pd.DataFrame) -> None:
        dt = DecisionTable(capture_levels=[0.30, 0.50, 0.70, 1.00])
        dt.generate(realistic_df, target="target", proba="proba")
        cutoff = dt.cutoff_for_capture(0.50)
        assert isinstance(cutoff, float)

    def test_decline_rate_for_capture_returns_float(self, realistic_df: pd.DataFrame) -> None:
        dt = DecisionTable(capture_levels=[0.30, 0.50, 0.70, 1.00])
        dt.generate(realistic_df, target="target", proba="proba")
        rate = dt.decline_rate_for_capture(0.50)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 100.0

    def test_cutoff_requires_generate(self) -> None:
        dt = DecisionTable()
        with pytest.raises(RuntimeError, match="generate"):
            dt.cutoff_for_capture(0.50)

    def test_decline_rate_requires_generate(self) -> None:
        dt = DecisionTable()
        with pytest.raises(RuntimeError, match="generate"):
            dt.decline_rate_for_capture(0.50)


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------


class TestDecisionTableValidation:
    def test_invalid_capture_level_zero(self) -> None:
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            DecisionTable(capture_levels=[0.0])

    def test_invalid_capture_level_negative(self) -> None:
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            DecisionTable(capture_levels=[-0.1])

    def test_invalid_capture_level_above_one(self) -> None:
        with pytest.raises(ValueError, match="\\(0, 1\\]"):
            DecisionTable(capture_levels=[1.5])

    def test_all_same_class_raises(self) -> None:
        df = pd.DataFrame({"target": [0, 0, 0], "proba": [0.1, 0.2, 0.3]})
        dt = DecisionTable()
        with pytest.raises(ValueError, match="at least 2"):
            dt.generate(df, target="target", proba="proba")

    def test_no_positive_cases_raises(self) -> None:
        df = pd.DataFrame({"target": [0, 0, 0, -1], "proba": [0.1, 0.2, 0.3, 0.4]})
        dt = DecisionTable()
        with pytest.raises(ValueError, match="no positive cases"):
            dt.generate(df, target="target", proba="proba")


# ---------------------------------------------------------------------------
# Tests: plot()
# ---------------------------------------------------------------------------


class TestDecisionTablePlot:
    def test_plot_requires_generate(self) -> None:
        dt = DecisionTable()
        with pytest.raises(RuntimeError, match="generate"):
            dt.plot()

    def test_plot_returns_axes(self, realistic_df: pd.DataFrame) -> None:
        pytest.importorskip("matplotlib")
        dt = DecisionTable(capture_levels=[0.30, 0.50, 1.00])
        dt.generate(realistic_df, target="target", proba="proba")
        import matplotlib

        matplotlib.use("Agg")
        ax = dt.plot()
        assert ax is not None
