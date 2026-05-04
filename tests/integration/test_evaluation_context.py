import pandas as pd
import pytest

from model_track.evaluation import BinaryEvaluator, MulticlassEvaluator, RegressionEvaluator


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "target_bin": [0, 1, 0, 1, 0, 1],
            "score_bin": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
            "target_multi": [0, 1, 2, 0, 1, 2],
            "pred_multi": [0, 1, 2, 0, 2, 1],
            "target_reg": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
            "pred_reg": [11.0, 19.0, 31.0, 9.0, 21.0, 29.0],
            "date": ["2023-01", "2023-01", "2023-01", "2023-02", "2023-02", "2023-02"],
        }
    )


def test_evaluation_integration_with_context(sample_df):
    """Test using evaluators with ProjectContext and saving metadata."""
    # 1. Binary Evaluation
    bin_ev = BinaryEvaluator()
    bin_report = bin_ev.report(
        sample_df, target="target_bin", score_col="score_bin", date_col="date"
    )

    # 2. Multiclass Evaluation
    multi_ev = MulticlassEvaluator()
    multi_report = multi_ev.report(
        sample_df, target="target_multi", pred_col="pred_multi", date_col="date"
    )

    # 3. Regression Evaluation
    reg_ev = RegressionEvaluator()
    reg_report = reg_ev.report(sample_df, target="target_reg", pred_col="pred_reg", date_col="date")

    # Check reports
    assert len(bin_report) == 2
    assert len(multi_report) == 2
    assert len(reg_report) == 2

    # Simulate saving to context (if we had a metadata storage pattern in Context)
    # For now, we verify that the evaluators work seamlessly with data typically found in a Context pipeline
    assert "ks" in bin_report.columns
    assert "f1" in multi_report.columns
    assert "rmse" in reg_report.columns
