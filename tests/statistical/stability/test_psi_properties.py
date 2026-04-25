import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from model_track.stability import PSICalculator


@settings(deadline=None)
@given(
    data=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=50,
        max_size=200,
    )
)
def test_psi_always_non_negative(data):
    df = pd.DataFrame({"feat": data})
    # Add some noise to create a second distribution
    df_noisy = df + np.random.normal(0, 0.1, size=(len(df), 1))

    calc = PSICalculator(n_bins=5)
    calc.fit(df, features=["feat"])
    summary = calc.transform(df_noisy)

    # PSI mathematically cannot be negative
    assert summary.iloc[0]["psi"] >= -1e-9  # Float tolerance


@settings(deadline=None)
@given(data=st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=20, max_size=100))
def test_psi_identical_is_zero(data):
    df = pd.DataFrame({"feat": data})

    calc = PSICalculator(n_bins=5)
    calc.fit(df, features=["feat"])
    summary = calc.transform(df)

    # For identical data, PSI should be 0 (within epsilon tolerance)
    assert summary.iloc[0]["psi"] < 1e-5
