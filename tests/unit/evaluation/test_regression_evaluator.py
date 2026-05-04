import numpy as np
import pandas as pd
import pytest

from model_track.evaluation import RegressionEvaluator


def test_regression_evaluator_evaluate_basic():
    """Test basic regression metrics."""
    evaluator = RegressionEvaluator()
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.1, 1.9, 3.2])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] > 0.9
    assert metrics["mape"] > 0
    assert metrics["median_ae"] > 0


def test_regression_evaluator_perfect_prediction():
    """Test metrics with zero error."""
    evaluator = RegressionEvaluator()
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 3.0])

    metrics = evaluator.evaluate(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["mape"] == 0.0


def test_regression_evaluator_mape_zeros():
    """Test MAPE warning with zeros in y_true."""
    evaluator = RegressionEvaluator()
    y_true = pd.Series([0.0, 1.0, 2.0])
    y_pred = pd.Series([0.1, 0.9, 2.1])

    with pytest.warns(UserWarning, match="y_true contains zeros"):
        metrics = evaluator.evaluate(y_true, y_pred)
        assert np.isinf(metrics["mape"]) or metrics["mape"] > 1000  # sklearn behavior


def test_regression_evaluator_report():
    """Test temporal report."""
    df = pd.DataFrame(
        {
            "target": [1.0, 2.0, 1.5, 2.5],
            "pred": [1.1, 1.9, 1.4, 2.6],
            "month": ["jan", "jan", "feb", "feb"],
        }
    )
    evaluator = RegressionEvaluator()
    report = evaluator.report(df, target="target", pred_col="pred", date_col="month")

    assert len(report) == 2
    assert "rmse" in report.columns
    assert "period" in report.columns


def test_regression_evaluator_prediction_interval():
    """Test prediction interval coverage."""
    evaluator = RegressionEvaluator()
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_lower = pd.Series([0.5, 1.5, 2.5, 3.5])
    y_upper = pd.Series([1.5, 2.5, 3.5, 4.5])

    coverage = evaluator.prediction_interval_coverage(y_true, y_lower, y_upper)
    assert coverage == 1.0

    y_upper_fail = pd.Series([0.9, 2.5, 3.5, 4.5])
    coverage_fail = evaluator.prediction_interval_coverage(y_true, y_lower, y_upper_fail)
    assert coverage_fail == 0.75


def test_regression_evaluator_residual_plot():
    """Test residual plot (lazy import check)."""
    evaluator = RegressionEvaluator()
    y_true = pd.Series([1.0, 2.0])
    y_pred = pd.Series([1.1, 1.9])

    # Mocking plt to avoid needing a GUI in tests
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    returned_ax = evaluator.residual_plot(y_true, y_pred, ax=ax)
    assert returned_ax is ax
    plt.close(fig)
