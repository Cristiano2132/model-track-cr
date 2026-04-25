"""Unit tests for BinaryEvaluator."""

import pandas as pd
import pytest

from model_track.evaluation import BinaryEvaluator


@pytest.fixture
def perfect_data() -> tuple[pd.Series, pd.Series]:
    """Perfect model: scores perfectly separate classes."""
    y_true = pd.Series([0, 0, 0, 1, 1, 1])
    y_proba = pd.Series([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
    return y_true, y_proba


@pytest.fixture
def random_data() -> tuple[pd.Series, pd.Series]:
    """Random model: scores are unrelated to classes."""
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    y_proba = pd.Series([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    return y_true, y_proba


@pytest.fixture
def mixed_data() -> tuple[pd.Series, pd.Series]:
    """Realistic model with imperfect separation."""
    y_true = pd.Series([0, 1, 0, 1, 1, 0, 0, 1])
    y_proba = pd.Series([0.1, 0.9, 0.2, 0.8, 0.7, 0.4, 0.3, 0.6])
    return y_true, y_proba


class TestBinaryEvaluatorMetrics:
    def test_perfect_model_auc(self, perfect_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = perfect_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert metrics["auc"] == pytest.approx(1.0)

    def test_perfect_model_ks(self, perfect_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = perfect_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert metrics["ks"] == pytest.approx(1.0)

    def test_perfect_model_gini(self, perfect_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = perfect_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert metrics["gini"] == pytest.approx(1.0)

    def test_gini_equals_2auc_minus_1(self, mixed_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = mixed_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert metrics["gini"] == pytest.approx(2 * metrics["auc"] - 1, abs=1e-10)

    def test_metric_keys_present(self, mixed_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = mixed_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert set(metrics.keys()) == {"auc", "ks", "gini", "brier_score", "log_loss"}

    def test_auc_bounds(self, mixed_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = mixed_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert 0.0 <= metrics["auc"] <= 1.0

    def test_ks_bounds(self, mixed_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = mixed_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert 0.0 <= metrics["ks"] <= 1.0

    def test_brier_score_bounds(self, mixed_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = mixed_data
        ev = BinaryEvaluator()
        metrics = ev.evaluate(y_true, y_proba)
        assert 0.0 <= metrics["brier_score"] <= 1.0

    def test_raises_for_multiclass_target(self) -> None:
        y_true = pd.Series([0, 1, 2, 1, 0])
        y_proba = pd.Series([0.1, 0.8, 0.9, 0.7, 0.2])
        ev = BinaryEvaluator()
        with pytest.raises(ValueError, match="2 unique classes"):
            ev.evaluate(y_true, y_proba)


class TestBinaryEvaluatorReport:
    def test_report_without_date_col(self, mixed_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = mixed_data
        df = pd.DataFrame({"target": y_true, "score": y_proba})
        ev = BinaryEvaluator()
        report = ev.report(df, "target", "score")
        assert len(report) == 1
        assert report.iloc[0]["period"] == "overall"

    def test_report_with_date_col(self) -> None:
        df = pd.DataFrame(
            {
                "target": [0, 1, 0, 1, 1, 0],
                "score": [0.1, 0.9, 0.2, 0.8, 0.7, 0.3],
                "month": ["jan", "jan", "jan", "feb", "feb", "feb"],
            }
        )
        ev = BinaryEvaluator()
        report = ev.report(df, "target", "score", date_col="month")
        assert len(report) == 2
        assert set(report["period"]) == {"jan", "feb"}

    def test_report_columns(self, mixed_data: tuple[pd.Series, pd.Series]) -> None:
        y_true, y_proba = mixed_data
        df = pd.DataFrame({"target": y_true, "score": y_proba})
        ev = BinaryEvaluator()
        report = ev.report(df, "target", "score")
        assert list(report.columns) == ["period", "auc", "ks", "gini", "brier_score", "log_loss"]

    def test_report_skips_periods_with_single_class(self) -> None:
        df = pd.DataFrame(
            {
                "target": [0, 0, 0, 1, 1, 1],
                "score": [0.1, 0.2, 0.3, 0.8, 0.9, 0.7],
                "month": ["jan", "jan", "jan", "feb", "feb", "feb"],  # jan has only class 0
            }
        )
        ev = BinaryEvaluator()
        report = ev.report(df, "target", "score", date_col="month")
        # jan should be skipped (only class 0), feb should also be skipped (only class 1)
        assert len(report) == 0
