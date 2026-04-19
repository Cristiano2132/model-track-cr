import time

import numpy as np
import pandas as pd

from model_track.woe.calculator import WoeCalculator


def test_woe_calculator_performance():
    """
    Benchmark: Verifies that WoeCalculator handles 100k rows efficiently.
    Target complexity should be near O(n).
    """
    # 1. Prepare 100k rows
    n_rows = 100_000
    df = pd.DataFrame(
        {
            "feature": np.random.choice(["A", "B", "C", "D", "E"], size=n_rows),
            "target": np.random.choice([0, 1], size=n_rows),
        }
    )

    calc = WoeCalculator()

    # 2. Measure fit time
    start_fit = time.time()
    calc.fit(df, target="target", columns=["feature"])
    end_fit = time.time()

    fit_duration = end_fit - start_fit

    # 3. Measure transform time
    start_trans = time.time()
    df_transformed = calc.transform(df, columns=["feature"])
    end_trans = time.time()

    trans_duration = end_trans - start_trans

    # 4. Assertions (loose bounds for CI stability)
    # 100k rows should be processed in less than 1 second for these operations
    assert fit_duration < 1.0, f"Fit took too long: {fit_duration:.4f}s"
    assert trans_duration < 1.0, f"Transform took too long: {trans_duration:.4f}s"
    assert len(df_transformed) == n_rows
