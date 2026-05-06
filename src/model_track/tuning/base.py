from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from model_track.base import TaskAdapter


class BaseTuner(ABC):
    """
    Abstract base class for hyperparameter tuners.

    Tuners are responsible for searching the best hyperparameter space
    for a given task and dataset.
    """

    def __init__(
        self,
        task: TaskAdapter,
        param_bounds: dict[str, Any] | None = None,
        cv_folds: int = 3,
        random_state: int = 42,
    ):
        self.task = task
        self.param_bounds = param_bounds or {}
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None

    @abstractmethod
    def tune(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Perform hyperparameter tuning.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            dict: The best hyperparameters found.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_model(self) -> Any:
        """
        Return a model instance initialized with the best parameters.

        Returns:
            Any: Model instance.
        """
        pass  # pragma: no cover
