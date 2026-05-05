import pandas as pd
import pytest

from model_track.context import ProjectContext
from model_track.stability import ModelPSI, StabilityReport


def test_model_psi_basic():
    """Test ModelPSI basic functionality."""
    df_ref = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4, 0.5]})
    df_cur = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4, 0.9]})  # Slight shift

    mpsi = ModelPSI(n_bins=5)
    mpsi.fit(df_ref, "score")
    mpsi.transform(df_cur)

    psi = mpsi.get_psi()
    assert psi >= 0
    assert mpsi.score_col_ == "score"


def test_stability_report_no_context():
    """Test report generation without context (using direct data)."""
    df_ref = pd.DataFrame({"f1": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
    df_cur = pd.DataFrame({"f1": [1, 2, 10], "score": [0.8, 0.9, 0.9]})

    report = StabilityReport()
    # Need to fit calculators if no context
    report.feature_psi_.fit(df_ref, ["f1"])
    report.score_psi_.fit(df_ref, "score")

    results = report.run(df_cur, features=["f1"], score_col="score")

    assert len(results) == 2
    assert "f1" in results["name"].values
    assert "score" in results["name"].values

    summary = report.summary()
    assert "overall_status" in summary
    assert summary["metrics"]["total_items"] == 2


def test_stability_report_with_context():
    """Test report integration with ProjectContext."""
    ctx = ProjectContext()
    df_ref = pd.DataFrame({"f1": [1, 2, 3], "score": [0.1, 0.2, 0.3]})

    # Fit and save to context
    from model_track.stability import PSICalculator

    psi_calc = PSICalculator().fit(df_ref, ["f1"])
    psi_calc.to_context(ctx)

    # Also save score ref
    mpsi = ModelPSI().fit(df_ref, "score")
    # Manually adding score to reference_stats since to_context uses the same dict
    ctx.reference_stats.update(mpsi.reference_stats_)

    report = StabilityReport(context=ctx)
    df_cur = pd.DataFrame({"f1": [1, 2, 3], "score": [0.1, 0.2, 0.3]})

    results = report.run(df_cur, score_col="score")
    assert len(results) == 2
    assert report.summary()["overall_status"] == "Stable"

    # Extra checks for coverage
    assert report.is_healthy() is True
    assert "STABLE" in report.summary_text()

    # Heatmap test
    ax = report.plot_drift_heatmap()
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.close()


def test_stability_report_edge_cases():
    """Test edge cases for coverage."""
    report = StabilityReport()

    # Not run summary
    assert "NOT RUN" in report.summary_text()
    assert report.is_healthy() is False

    with pytest.raises(ValueError, match="Run the report before plotting"):
        report.plot_drift_heatmap()

    # Unstable scenario summary_text
    df_ref = pd.DataFrame({"f1": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
    df_cur = pd.DataFrame({"f1": [10, 20, 30], "score": [0.8, 0.9, 0.95]})

    report.feature_psi_.fit(df_ref, ["f1"])
    report.score_psi_.fit(df_ref, "score")
    report.run(df_cur, features=["f1"], score_col="score")

    text = report.summary_text()
    assert "UNSTABLE" in text
    assert "Unstable Features: f1" in text
    assert "Unstable Scores: score" in text


def test_stability_report_from_context_method():
    ctx = ProjectContext()
    report = StabilityReport.from_context(ctx)
    assert report.context is ctx


def test_stability_report_generate_alias():
    report = StabilityReport()
    df_ref = pd.DataFrame({"f1": [1, 2, 3]})
    report.feature_psi_.fit(df_ref, ["f1"])
    results = report.generate(df_ref, features=["f1"])
    assert not results.empty


def test_stability_report_monitor_status():
    report = StabilityReport()
    report.results_["data"] = pd.DataFrame(
        [{"type": "feature", "name": "f1", "psi": 0.15, "status": "Monitor"}]
    )
    summary = report.summary()
    assert summary["overall_status"] == "Monitor"


def test_stability_report_run_exception_handling():
    # Trigger exception in run
    report = StabilityReport()
    # No fit, so transform will fail for ModelPSI
    # But it catches (ValueError, KeyError)
    df = pd.DataFrame({"score": [1, 2, 3]})
    results = report.run(df, score_col="score")
    # Should not crash, and results should be empty or not contain "score"
    if not results.empty:
        assert "score" not in results["name"].values
    else:
        assert results.empty


def test_stability_report_run_exception_trigger():
    report = StabilityReport()
    # Mock score_psi_.transform to raise ValueError
    from unittest.mock import MagicMock

    report.score_psi_.transform = MagicMock(side_effect=ValueError("Test Error"))
    df = pd.DataFrame({"score": [1, 2, 3]})
    results = report.run(df, score_col="score")
    # This should hit the 'except' block in run()
    assert "score" not in results["name"].values if not results.empty else True


def test_stability_report_multiclass():
    """Test StabilityReport with multiclass score columns."""
    df_ref = pd.DataFrame({"proba_A": [0.8, 0.7, 0.6], "proba_B": [0.2, 0.3, 0.4], "f1": [1, 2, 3]})
    df_cur = pd.DataFrame({"proba_A": [0.1, 0.2, 0.3], "proba_B": [0.9, 0.8, 0.7], "f1": [1, 2, 3]})

    report = StabilityReport()
    # Fit
    report.feature_psi_.fit(df_ref, ["f1"])
    report.multiclass_psi_.fit(df_ref, proba_cols=["proba_A", "proba_B"])

    results = report.run(df_cur, features=["f1"], score_col=["proba_A", "proba_B"])

    assert len(results) == 3
    names = results["name"].tolist()
    assert "f1" in names
    assert "proba_A" in names
    assert "proba_B" in names

    # Verify heatmap works with multiple scores
    ax = report.plot_drift_heatmap()
    assert ax is not None
    import matplotlib.pyplot as plt

    plt.close()


def test_stability_report_multiclass_exception_handling():
    """Test exception handling in multiclass score processing."""
    report = StabilityReport()
    from unittest.mock import MagicMock

    report.multiclass_psi_.transform = MagicMock(side_effect=ValueError("Multiclass Error"))

    df = pd.DataFrame({"proba_A": [0.1, 0.2], "proba_B": [0.9, 0.8]})
    results = report.run(df, score_col=["proba_A", "proba_B"])

    assert results.empty or "proba_A" not in results["name"].values


def test_stability_report_multiclass_with_context():
    """Test StabilityReport multiclass integration with ProjectContext."""
    ctx = ProjectContext()
    df_ref = pd.DataFrame({"proba_A": [0.8, 0.7, 0.6], "proba_B": [0.2, 0.3, 0.4]})

    # Save to context
    from model_track.stability import MulticlassPSI

    mpsi = MulticlassPSI().fit(df_ref, proba_cols=["proba_A", "proba_B"])
    ctx.reference_stats = {}
    ctx.reference_stats.update(mpsi.reference_stats_)

    report = StabilityReport(context=ctx)
    df_cur = pd.DataFrame({"proba_A": [0.8, 0.7, 0.6], "proba_B": [0.2, 0.3, 0.4]})

    results = report.run(df_cur, score_col=["proba_A", "proba_B"])
    assert len(results) == 2
    assert "proba_A" in results["name"].values
    assert "proba_B" in results["name"].values


def test_stability_report_regression():
    """Test StabilityReport regression integration with ProjectContext."""
    from model_track.base import TaskType

    ctx = ProjectContext()
    ctx.task_type = TaskType.REGRESSION

    df_ref = pd.DataFrame({"score": [10.5, 12.1, 15.3, 14.2], "f1": [1, 2, 3, 4]})

    # Save to context
    from model_track.stability import PSICalculator, RegressionPSI

    psi_calc = PSICalculator().fit(df_ref, ["f1"])
    rpsi = RegressionPSI().fit(df_ref, "score")

    ctx.reference_stats = {}
    ctx.reference_stats.update(psi_calc.reference_stats_)
    ctx.reference_stats.update(rpsi.reference_stats_)

    report = StabilityReport(context=ctx)
    df_cur = pd.DataFrame({"score": [10.2, 12.5, 15.0, 14.1], "f1": [1, 2, 3, 4]})

    results = report.run(df_cur, features=["f1"], score_col="score")

    assert len(results) == 2
    assert "score" in results["name"].values
    assert "f1" in results["name"].values

    # Verify the report indeed used the RegressionPSI instance
    # The reference stats should be loaded into regression_psi_
    assert report.regression_psi_.score_col_ == "score"
    assert report.regression_psi_.get_psi() >= 0
