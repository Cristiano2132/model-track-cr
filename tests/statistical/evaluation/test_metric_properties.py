import numpy as np
import pandas as pd
import pytest

from model_track.evaluation import BinaryEvaluator, MulticlassEvaluator, RegressionEvaluator


@pytest.fixture
def reg_data():
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100))
    y_pred = y_true + np.random.normal(0, 0.1, 100)
    return y_true, y_pred


def test_regression_monotonicity(reg_data):
    """If errors increase, metrics must reflect that."""
    y_true, y_pred = reg_data
    evaluator = RegressionEvaluator()

    m1 = evaluator.evaluate(y_true, y_pred)

    # Increase error
    y_pred_bad = y_pred + 1.0
    m2 = evaluator.evaluate(y_true, y_pred_bad)

    assert m2["rmse"] > m1["rmse"]
    assert m2["mae"] > m1["mae"]
    assert m2["r2"] < m1["r2"]


def test_regression_r2_limit():
    """R2 must be <= 1.0."""
    evaluator = RegressionEvaluator()
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.1, 1.9, 3.2])

    m = evaluator.evaluate(y_true, y_pred)
    assert m["r2"] <= 1.0

    # Perfect
    m_perf = evaluator.evaluate(y_true, y_true)
    assert m_perf["r2"] == 1.0


def test_binary_auc_limit():
    """AUC must be between 0 and 1."""
    evaluator = BinaryEvaluator()
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8])

    m = evaluator.evaluate(y_true, y_score)
    assert 0 <= m["auc"] <= 1.0


def test_multiclass_micro_f1_is_accuracy():
    """In multiclass, micro-averaged F1 equals accuracy."""
    np.random.seed(42)
    y_true = pd.Series(np.random.randint(0, 3, 100))
    y_pred = pd.Series(np.random.randint(0, 3, 100))

    evaluator = MulticlassEvaluator(average="micro")
    m = evaluator.evaluate(y_true, y_pred=y_pred)

    assert pytest.approx(m["f1"]) == m["accuracy"]


def test_regression_scale_invariance_failure():
    """RMSE and MAE are NOT scale invariant (they scale with data)."""
    evaluator = RegressionEvaluator()
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.1, 1.9, 3.1])

    m1 = evaluator.evaluate(y_true, y_pred)

    # Scale data by 10
    m2 = evaluator.evaluate(y_true * 10, y_pred * 10)

    assert pytest.approx(m2["rmse"]) == m1["rmse"] * 10
    assert pytest.approx(m2["mae"]) == m1["mae"] * 10
    # R2 SHOULD be invariant to scale
    assert pytest.approx(m2["r2"]) == m1["r2"]
