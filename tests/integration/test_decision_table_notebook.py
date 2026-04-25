"""Integration test: DecisionTable matches the original notebook logic exactly."""

import pandas as pd
import pytest

from model_track.evaluation import DecisionTable


def original_generate_fraud_capture_table(
    df: pd.DataFrame, target_col: str, proba_col: str
) -> pd.DataFrame:
    """The original logic straight from ieee_fraud_eda_baseline.ipynb."""
    # 1. Sort by descending score
    df_sorted = (
        df[[target_col, proba_col]]
        .sort_values(by=proba_col, ascending=False)
        .reset_index(drop=True)
    )

    total_frauds = df_sorted[target_col].sum()
    total_orders = len(df_sorted)

    # 2. Cumulative calculations
    df_sorted["cumulative_frauds_caught"] = df_sorted[target_col].cumsum()
    df_sorted["tpr_acum"] = df_sorted["cumulative_frauds_caught"] / total_frauds

    # 3. Desired capture levels
    target_capture = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    rows = []
    for target in target_capture:
        mask = df_sorted["tpr_acum"] >= target
        if not mask.any():
            continue

        idx = int(mask.idxmax())
        record = df_sorted.iloc[idx]

        declined_orders_count = idx + 1
        total_orders_pct = (declined_orders_count / total_orders) * 100

        actual_tpr = record["tpr_acum"] * 100
        fnr = 100 - actual_tpr
        cutoff = record[proba_col]

        hit_rate = (record["cumulative_frauds_caught"] / declined_orders_count) * 100

        rows.append(
            {
                "target_capture_pct": int(target * 100),
                "orders_declined_pct": round(total_orders_pct, 2),
                "actual_tpr_pct": round(actual_tpr, 2),
                "fnr_pct": round(fnr, 2),
                "hit_rate_pct": round(hit_rate, 2),
                "cutoff": round(cutoff, 4),
            }
        )

    return pd.DataFrame(rows)


@pytest.fixture
def test_df() -> pd.DataFrame:
    """A slightly larger, randomized dataset to ensure edge cases match."""
    import numpy as np

    rng = np.random.default_rng(123)
    n = 1000
    target = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    # Give proba a correlation with target so it's somewhat realistic
    proba = np.clip(rng.normal(loc=target * 0.5, scale=0.2) + 0.3, 0, 1)

    return pd.DataFrame({"target": target, "proba": proba})


def test_decision_table_matches_notebook_exactly(test_df: pd.DataFrame) -> None:
    # Generate via original function
    df_original = original_generate_fraud_capture_table(
        test_df, target_col="target", proba_col="proba"
    )

    # Generate via new class
    dt = DecisionTable()
    df_new = dt.generate(test_df, target="target", proba="proba")

    # Assert exact match
    pd.testing.assert_frame_equal(df_original, df_new)
