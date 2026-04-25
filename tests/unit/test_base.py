import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from model_track.base import (
    BinaryAdapter,
    MulticlassAdapter,
    RegressionAdapter,
    TaskAdapter,
    TaskType,
)


def test_binary_adapter_validation():
    adapter = BinaryAdapter()

    # Valid binary
    adapter.validate_target(pd.Series([0, 1, 0, 1]))
    # Valid unary (considered a subset of binary/multiclass)
    adapter.validate_target(pd.Series([1, 1, 1]))

    # Invalid binary (multiclass)
    with pytest.raises(ValueError, match="Binary task requires <= 2 unique classes"):
        adapter.validate_target(pd.Series([0, 1, 2]))

    assert adapter.task_type == TaskType.BINARY
    assert adapter.positive_class() == 1
    assert "auc" in adapter.default_metrics()


def test_multiclass_adapter_validation():
    adapter = MulticlassAdapter()

    # Valid multiclass
    adapter.validate_target(pd.Series([0, 1, 2]))

    # Also valid (technically a subset)
    adapter.validate_target(pd.Series([0, 1]))

    assert adapter.task_type == TaskType.MULTICLASS
    assert adapter.positive_class() is None
    assert "macro_auc" in adapter.default_metrics()


def test_regression_adapter_validation():
    adapter = RegressionAdapter()

    # Valid regression
    adapter.validate_target(pd.Series([1.5, 2.3, 4.0]))

    # Invalid regression (categorical)
    with pytest.raises(ValueError, match="Regression task requires numeric target"):
        adapter.validate_target(pd.Series(["a", "b", "c"]))

    assert adapter.task_type == TaskType.REGRESSION
    assert "rmse" in adapter.default_metrics()


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_regression_adapter_pbt(values):
    """PBT: Ensure RegressionAdapter accepts any valid numeric list."""
    adapter = RegressionAdapter()
    adapter.validate_target(pd.Series(values))


@given(st.lists(st.text(), min_size=1))
def test_regression_adapter_fails_on_text_pbt(values):
    """PBT: Ensure RegressionAdapter fails on non-numeric strings."""
    adapter = RegressionAdapter()
    # If the list is empty it might pass, but min_size=1 ensures it's checked
    with pytest.raises(ValueError, match="Regression task requires numeric target"):
        adapter.validate_target(pd.Series(values))


def test_task_adapter_protocol():
    # Verify our adapters satisfy the Protocol
    assert isinstance(BinaryAdapter(), TaskAdapter)
    assert isinstance(MulticlassAdapter(), TaskAdapter)
    assert isinstance(RegressionAdapter(), TaskAdapter)
