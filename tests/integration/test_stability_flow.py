import pandas as pd

from model_track.context import ProjectContext
from model_track.stability import ModelPSI, PSICalculator, StabilityReport


def test_full_stability_monitoring_flow():
    """
    Integration test:
    1. Reference data -> Fit calculators -> Save to Context
    2. Current data (shifted) -> Load Context -> Run StabilityReport
    3. Verify detection.
    """
    # 1. Reference Data (Baseline)
    df_ref = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45] * 20,
            "income": [2000, 3000, 4000, 5000, 6000] * 20,
            "score": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
        }
    )

    ctx = ProjectContext()

    # Fit features
    psi_calc = PSICalculator(n_bins=5).fit(df_ref, ["age", "income"])
    psi_calc.to_context(ctx)

    # Fit score
    mpsi = ModelPSI(n_bins=5).fit(df_ref, "score")
    ctx.reference_stats.update(mpsi.reference_stats_)

    # 2. Current Data (Shifted)
    df_cur = pd.DataFrame(
        {
            "age": [50, 60, 70, 80, 90] * 20,  # DRAGGED UP
            "income": [2000, 3000, 4000, 5000, 6000] * 20,  # STABLE
            "score": [0.8, 0.9, 0.95, 0.99, 0.99] * 20,  # DRAGGED UP
        }
    )

    # 3. Execution
    report = StabilityReport(context=ctx)
    results = report.run(df_cur, score_col="score")
    summary = report.summary()

    # 4. Validations
    assert summary["overall_status"] == "Unstable"

    # Verify feature status
    age_status = results.loc[results["name"] == "age", "status"].iloc[0]
    income_status = results.loc[results["name"] == "income", "status"].iloc[0]
    score_status = results.loc[results["name"] == "score", "status"].iloc[0]

    assert age_status == "Unstable"
    assert income_status == "Stable"
    assert score_status == "Unstable"

    assert "age" in summary["unstable_features"]
    assert "score" in summary["unstable_scores"]
    assert "income" not in summary["unstable_features"]
