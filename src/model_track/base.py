from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import pandas as pd


class TaskType(Enum):
    """Supported modeling task types."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


@runtime_checkable
class TaskAdapter(Protocol):
    """
    Protocol defining the interface for task-specific adapters.
    Used by evaluators and tuners to adapt behavior to the task type.
    """

    task_type: TaskType

    def validate_target(self, y: pd.Series) -> None:
        """Validate if the target series is compatible with this task."""
        ...

    def default_metrics(self) -> list[str]:
        """Return the default metrics for this task."""
        ...

    def positive_class(self) -> Any:
        """Return the positive class identifier (relevant for BINARY)."""
        ...


class BinaryAdapter:
    """Adapter for binary classification tasks."""

    task_type = TaskType.BINARY

    def __init__(self, pos_class: Any = 1):
        self.pos_class = pos_class

    def validate_target(self, y: pd.Series) -> None:
        if y.nunique() > 2:
            raise ValueError(f"Binary task requires <= 2 unique classes, found {y.nunique()}")

    def default_metrics(self) -> list[str]:
        return ["auc", "ks", "gini", "brier_score"]

    def positive_class(self) -> Any:
        return self.pos_class


class MulticlassAdapter:
    """Adapter for multiclass classification tasks."""

    task_type = TaskType.MULTICLASS

    def __init__(self, classes: list[Any] | None = None):
        self.classes = classes

    def validate_target(self, y: pd.Series) -> None:
        if y.nunique() <= 2:
            # Although 2 classes is technically multiclass, usually BinaryAdapter is preferred.
            # We allow it but warn or just pass if the user explicitly chose Multiclass.
            pass

    def default_metrics(self) -> list[str]:
        return ["macro_auc", "kappa", "accuracy"]

    def positive_class(self) -> Any:
        return None


class RegressionAdapter:
    """Adapter for regression tasks."""

    task_type = TaskType.REGRESSION

    def validate_target(self, y: pd.Series) -> None:
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError("Regression task requires numeric target.")

    def default_metrics(self) -> list[str]:
        return ["rmse", "mae", "r2", "mape"]

    def positive_class(self) -> Any:
        return None


class BaseTransformer(ABC):
    """
    Base interface for all model-track transformers.
    Defines the fit and transform contract to ensure modularity.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: str | None = None) -> "BaseTransformer":
        """
        Train the transformer.

        Args:
            df: Input DataFrame to train on.
            target: Target variable name for supervised transformers.

        Returns:
            BaseTransformer: The fitted transformer instance.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the data.

        Args:
            df: Input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass

    def fit_transform(self, df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """
        Helper method to perform both fit and transform in sequence.

        Args:
            df: Input DataFrame.
            target: Target variable name.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        return self.fit(df, target).transform(df)
