import pandas as pd
import pytest

from model_track.binning.bin_applier import BinApplier
from model_track.context import ProjectContext


def test_bin_applier_basic():
    bins_map = {"age": [25.0, 50.0, 75.0]}
    applier = BinApplier(bins_map)

    df = pd.DataFrame({"age": [10, 30, 60, 90, None]})
    transformed = applier.apply(df)

    assert transformed["age"].iloc[0] == "(-inf, 25.0]"
    assert transformed["age"].iloc[1] == "(25.0, 50.0]"
    assert transformed["age"].iloc[4] == "N/A"


def test_bin_applier_from_context():
    ctx = ProjectContext()
    ctx.bins_map = {"score": [0.5]}

    applier = BinApplier.from_context(ctx)
    df = pd.DataFrame({"score": [0.1, 0.9]})
    transformed = applier.apply(df)

    assert transformed["score"].iloc[0] == "(-inf, 0.5]"
    assert transformed["score"].iloc[1] == "(0.5, inf]"


def test_bin_applier_missing_column():
    # 'a' is NOT in bins_map
    applier = BinApplier({"other_col": [1]})
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="Column 'a' not found"):
        applier.apply(df, columns=["a"])


def test_bin_applier_selective_columns():
    applier = BinApplier({"a": [1], "b": [1]})
    df = pd.DataFrame({"a": [0.5, 1.5], "b": [0.5, 1.5], "c": [10, 20]})

    # Apply only to 'a'
    transformed = applier.apply(df, columns=["a"])

    assert transformed["a"].dtype == object
    assert (
        transformed["b"].dtype == float
    )  # 'b' remains untouched because we specified columns=['a']
    assert transformed["c"].iloc[0] == 10
