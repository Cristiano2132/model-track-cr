import numpy as np
import pandas as pd
import pytest

from model_track.binning.quantile_binner import QuantileBinner


def test_quantile_binner_basic():
    df = pd.DataFrame({"age": np.linspace(0, 100, 101)})
    binner = QuantileBinner(n_bins=4)
    binner.fit(df, column="age")

    # Edges should be [0, 25, 50, 75, 100]
    # self.bins should be [25, 50, 75]
    assert len(binner.bins) == 3
    assert binner.bins == [25.0, 50.0, 75.0]

    binned = binner.transform(df, column="age")
    assert binned.nunique() == 4
    assert "N/A" not in binned.values


def test_quantile_binner_with_nulls():
    df = pd.DataFrame({"age": [10, 20, 30, 40, np.nan]})
    binner = QuantileBinner(n_bins=2)
    binner.fit(df, column="age")

    # Edges: [10, 25, 40] -> self.bins: [25]
    assert binner.bins == [25.0]

    binned = binner.transform(df, column="age")
    assert binned.iloc[4] == "N/A"
    assert binned.nunique() == 3  # (10, 25], (25, 40], N/A


def test_quantile_binner_duplicates():
    # Many zeros will cause duplicate edges
    df = pd.DataFrame({"score": [0, 0, 0, 0, 0, 10, 20, 30]})
    binner = QuantileBinner(n_bins=4)
    binner.fit(df, column="score")

    # Should not crash and should drop duplicates
    binned = binner.transform(df, column="score")
    assert binned.dtype == object


def test_quantile_binner_not_fitted():
    binner = QuantileBinner()
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(RuntimeError, match="must be fitted"):
        binner.transform(df, column="a")


def test_quantile_binner_empty_df():
    df = pd.DataFrame({"a": []})
    binner = QuantileBinner()
    binner.fit(df, column="a")
    assert binner.bins == []

    binned = binner.transform(df, column="a")
    assert binned.empty


def test_quantile_binner_single_value():
    # Only one unique value
    df = pd.DataFrame({"a": [1, 1, 1, 1]})
    binner = QuantileBinner(n_bins=4)
    binner.fit(df, column="a")
    assert binner.bins == []

    binned = binner.transform(df, column="a")
    assert binned.nunique() == 1
    assert binned.iloc[0] == "(-inf, inf]"
