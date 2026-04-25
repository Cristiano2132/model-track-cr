"""Base class for all model evaluators in model-track."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from model_track.base import TaskType


class BaseEvaluator(ABC):
    """
    Abstract base class for all task-specific evaluators.

    Subclasses must implement ``evaluate()`` and ``report()``, which together
    form the standard evaluation contract used across binary, multiclass, and
    regression tasks.

    Attributes:
        task_type: The ``TaskType`` this evaluator is designed for.

    Example:
        >>> # Use a concrete subclass, e.g. BinaryEvaluator
        >>> from model_track.evaluation import BinaryEvaluator
        >>> evaluator = BinaryEvaluator()
        >>> evaluator.task_type.value
        'binary'
    """

    task_type: TaskType

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        """
        Compute task-specific metrics for a single dataset split.

        Returns:
            dict[str, float]: Metric name → value mapping.
        """

    @abstractmethod
    def report(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Compute metrics per time period and return a summary DataFrame.

        Returns:
            pd.DataFrame: One row per period with metric columns.
        """
