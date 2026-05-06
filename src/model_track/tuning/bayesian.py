import warnings
from abc import abstractmethod
from typing import Any

import pandas as pd
from sklearn.model_selection import cross_val_score

from model_track.tuning.base import BaseTuner

try:
    from bayes_opt import BayesianOptimization

    HAS_BAYES = True
except ImportError:
    HAS_BAYES = False


class BayesianTuner(BaseTuner):
    """
    Hyperparameter tuner using Bayesian Optimization.

    Wraps the `bayesian-optimization` library to perform a global search
    over the hyperparameter space.
    """

    def __init__(
        self,
        task: Any,
        param_bounds: dict[str, Any] | None = None,
        cv_folds: int = 3,
        n_iter: int = 15,
        init_points: int = 5,
        random_state: int = 42,
    ):
        super().__init__(
            task=task,
            param_bounds=param_bounds,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        self.n_iter = n_iter
        self.init_points = init_points

        if not HAS_BAYES:
            raise ImportError(
                "'bayesian-optimization' is required for BayesianTuner. "
                "Install it with 'pip install model-track-cr[tuning]'."
            )

    def tune(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """
        Perform Bayesian optimization.

        Args:
            X: Feature matrix.
            y: Target vector.

        Returns:
            dict: The best hyperparameters found.
        """
        # 1. Decide scoring metric
        scoring = self._get_scoring_metric()

        # 2. Define objective function
        def objective_function(**params: Any) -> float:
            # Round integer parameters if any
            processed_params = self._process_params(params)
            model = self._create_model(processed_params)

            # bayesian-optimization MAXIMIZES the output
            # cross_val_score returns scores where higher is better for 'neg_...' metrics too
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=scoring, n_jobs=-1)
            return float(scores.mean())

        # 3. Run optimizer
        optimizer = BayesianOptimization(
            f=objective_function,
            pbounds=self.param_bounds,
            random_state=self.random_state,
            verbose=2,
        )

        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        self.best_params_ = self._process_params(optimizer.max["params"])
        self.best_score_ = float(optimizer.max["target"])

        return self.best_params_

    def get_model(self) -> Any:
        """Return a model instance initialized with the best parameters."""
        if self.best_params_ is None:
            raise ValueError("Tuner has not been fitted yet. Call .tune() first.")
        return self._create_model(self.best_params_)

    def _get_scoring_metric(self) -> str:
        """Map internal task metrics to sklearn scoring strings."""
        primary_metric = self.task.default_metrics()[0]
        mapping = {
            "auc": "roc_auc",
            "macro_auc": "roc_auc_ovr",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "accuracy": "accuracy",
        }
        return mapping.get(primary_metric, primary_metric)

    def _process_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Process parameters from optimizer (e.g. round integers).
        Override in subclasses for specific model needs.
        """
        return params

    @abstractmethod
    def _create_model(self, params: dict[str, Any]) -> Any:
        """Factory method to create the model instance with given params."""
        pass  # pragma: no cover
