import numpy as np
import pandas as pd

from model_track.binning.tree_binner import TreeBinner
from model_track.preprocessing.types import TypeDetector
from model_track.woe.calculator import WoeCalculator
from model_track.woe.stability import WoeStability


def test_full_modeling_pipeline():
    """
    Integration Test: Validates that multiple components work together
    in a realistic sequence.
    """
    # 1. Create synthetic data
    df = pd.DataFrame(
        {
            "id": range(100),
            "period": pd.date_range("2023-01-01", periods=100, freq="D"),
            "income": np.random.normal(5000, 1000, 100),
            "target": np.random.choice([0, 1], size=100),
        }
    )

    # 2. Type Detection
    detector = TypeDetector(target="target", id_cols=["id"], datetime_cols=["period"])
    types = detector.detect(df)
    assert "income" in types["numerical"]

    # 3. Binning
    binner = TreeBinner(max_depth=2)
    binner.fit(df, column="income", target="target")
    df["income_cat"] = binner.transform(df, column="income")

    # 4. WOE Calculation
    calc = WoeCalculator()
    calc.fit(df, target="target", columns=["income_cat"])
    df_woe = calc.transform(df, columns=["income_cat"])
    assert "income_cat_woe" in df_woe.columns

    # 5. Stability Analysis
    stability = WoeStability(date_col="period")
    matrix = stability.calculate_stability_matrix(
        df=df_woe, feature_col="income_cat", target_col="target"
    )

    # Final Sanity Check
    assert not matrix.empty
    assert len(matrix.columns) > 0
