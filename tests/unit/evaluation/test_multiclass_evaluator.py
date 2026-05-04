import numpy as np
import pandas as pd
import pytest

from model_track.evaluation import MulticlassEvaluator


def test_multiclass_evaluator_evaluate_basic():
    """Test basic evaluation with 3 classes."""
    evaluator = MulticlassEvaluator(average="macro")
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 2, 0, 1, 2])  # Perfect prediction

    # Probabilities for perfect prediction
    y_proba = pd.DataFrame(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        columns=[0, 1, 2],
    )

    metrics = evaluator.evaluate(y_true, y_proba=y_proba, y_pred=y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["log_loss"] < 0.1


def test_multiclass_evaluator_infer_pred():
    """Test inference of y_pred from y_proba."""
    evaluator = MulticlassEvaluator()
    y_true = pd.Series([0, 1, 2])
    y_proba = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])

    metrics = evaluator.evaluate(y_true, y_proba=y_proba)
    assert metrics["accuracy"] == 1.0


def test_multiclass_evaluator_binary_target_raises():
    """Test that binary target raises ValueError."""
    evaluator = MulticlassEvaluator()
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 1])

    with pytest.raises(ValueError, match="expects > 2 unique classes"):
        evaluator.evaluate(y_true, y_pred=y_pred)


def test_multiclass_evaluator_no_inputs_raises():
    """Test that providing neither y_proba nor y_pred raises ValueError."""
    evaluator = MulticlassEvaluator()
    y_true = pd.Series([0, 1, 2])

    with pytest.raises(ValueError, match="Either y_proba or y_pred must be provided"):
        evaluator.evaluate(y_true)


def test_multiclass_evaluator_averaging():
    """Test different averaging methods."""
    y_true = pd.Series([0, 0, 1, 1, 2, 2])
    y_pred = pd.Series([0, 1, 1, 1, 2, 0])

    macro_ev = MulticlassEvaluator(average="macro")
    micro_ev = MulticlassEvaluator(average="micro")

    macro_metrics = macro_ev.evaluate(y_true, y_pred=y_pred)
    micro_metrics = micro_ev.evaluate(y_true, y_pred=y_pred)

    assert macro_metrics["precision"] != micro_metrics["precision"]


def test_multiclass_evaluator_report_basic():
    """Test report generation."""
    df = pd.DataFrame(
        {
            "target": [0, 1, 2, 0, 1, 2],
            "pred": [0, 1, 2, 0, 2, 1],
            "month": ["jan", "jan", "jan", "feb", "feb", "feb"],
        }
    )
    evaluator = MulticlassEvaluator()
    report = evaluator.report(df, target="target", pred_col="pred", date_col="month")

    assert len(report) == 2
    assert "accuracy" in report.columns
    assert "period" in report.columns


def test_multiclass_evaluator_confusion_matrix():
    """Test confusion matrix method."""
    evaluator = MulticlassEvaluator()
    y_true = pd.Series([0, 1, 2])
    y_pred = pd.Series([0, 2, 2])
    cm = evaluator.confusion_matrix(y_true, y_pred)

    assert cm.loc[0, 0] == 1
    assert cm.loc[1, 2] == 1
    assert cm.loc[2, 2] == 1


def test_multiclass_evaluator_invalid_average():
    """Test invalid average parameter."""
    with pytest.raises(ValueError, match="Unsupported average method"):
        MulticlassEvaluator(average="invalid")


def test_multiclass_evaluator_evaluate_df_proba():
    """Test evaluate with y_proba as DataFrame."""
    evaluator = MulticlassEvaluator()
    y_true = pd.Series([0, 1, 2])
    y_proba = pd.DataFrame([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], columns=[0, 1, 2])
    metrics = evaluator.evaluate(y_true, y_proba=y_proba)
    assert metrics["accuracy"] == 1.0
    assert "log_loss" in metrics


def test_multiclass_evaluator_report_global():
    """Test report without date_col."""
    df = pd.DataFrame(
        {
            "target": [0, 1, 2, 0, 1, 2],
            "pred": [0, 1, 2, 1, 1, 2],
        }
    )
    evaluator = MulticlassEvaluator()
    report = evaluator.report(df, target="target", pred_col="pred")
    assert len(report) == 1
    assert report.loc[0, "period"] == "overall"


def test_multiclass_evaluator_report_empty_or_small():
    """Test report with empty data or periods with few classes."""
    df = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],  # Only 2 classes, should be skipped
            "pred": [0, 1, 1, 1],
            "month": ["jan", "jan", "feb", "feb"],
        }
    )
    evaluator = MulticlassEvaluator()
    report = evaluator.report(df, target="target", pred_col="pred", date_col="month")
    assert len(report) == 0  # Both months skipped because classes <= 2


def test_multiclass_evaluator_report_with_scores():
    """Test report with score_cols to cover log_loss branch."""
    df = pd.DataFrame(
        {
            "target": ["A", "B", "C", "A", "B", "C"],
            "A": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "B": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            "C": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "month": ["jan", "jan", "jan", "feb", "feb", "feb"],
        }
    )
    evaluator = MulticlassEvaluator()
    report = evaluator.report(df, target="target", score_cols=["A", "B", "C"], date_col="month")
    assert "log_loss" in report.columns


def test_multiclass_evaluator_classification_report():
    """Test classification_report method."""
    evaluator = MulticlassEvaluator()
    y_true = pd.Series([0, 1, 2, 0, 1, 2])
    y_pred = pd.Series([0, 1, 2, 0, 1, 1])
    report = evaluator.classification_report(y_true, y_pred)
    assert isinstance(report, str)
    assert "precision" in report
