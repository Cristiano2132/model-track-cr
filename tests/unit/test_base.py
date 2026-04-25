import pandas as pd
import pytest

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


def test_task_adapter_protocol():
    # Verify our adapters satisfy the Protocol
    assert isinstance(BinaryAdapter(), TaskAdapter)
    assert isinstance(MulticlassAdapter(), TaskAdapter)
    assert isinstance(RegressionAdapter(), TaskAdapter)
