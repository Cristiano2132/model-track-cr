from typing import Any

from model_track.base import TaskType
from model_track.tuning.bayesian import BayesianTuner

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


class LGBMTuner(BayesianTuner):
    """
    Specialized Bayesian Tuner for LightGBM models.

    Provides sensible default parameter bounds and automatically
    handles model instantiation based on the task type.
    """

    DEFAULT_BOUNDS = {
        "num_leaves": (20, 300),
        "max_depth": (3, 15),
        "learning_rate": (0.01, 0.3),
        "feature_fraction": (0.5, 1.0),
        "bagging_fraction": (0.5, 1.0),
        "min_child_samples": (10, 100),
        "lambda_l1": (0, 10),
        "lambda_l2": (0, 10),
    }

    def __init__(
        self,
        task: Any,
        param_bounds: dict[str, Any] | None = None,
        cv_folds: int = 3,
        n_iter: int = 15,
        init_points: int = 5,
        random_state: int = 42,
    ):
        bounds = param_bounds if param_bounds is not None else self.DEFAULT_BOUNDS
        super().__init__(
            task=task,
            param_bounds=bounds,
            cv_folds=cv_folds,
            n_iter=n_iter,
            init_points=init_points,
            random_state=random_state,
        )

        if not HAS_LGBM:
            raise ImportError(
                "'lightgbm' is required for LGBMTuner. "
                "Install it with 'pip install model-track-cr[tuning]'."
            )

    def _process_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Convert float parameters to integers where needed by LightGBM."""
        processed = params.copy()
        for key in ["num_leaves", "max_depth", "min_child_samples"]:
            if key in processed:
                processed[key] = int(processed[key])
        return processed

    def _create_model(self, params: dict[str, Any]) -> Any:
        """Create LGBMClassifier or LGBMRegressor based on task type."""
        if self.task.task_type == TaskType.REGRESSION:
            return lgb.LGBMRegressor(random_state=self.random_state, verbose=-1, **params)

        # Binary or Multiclass
        objective = "binary" if self.task.task_type == TaskType.BINARY else "multiclass"
        return lgb.LGBMClassifier(
            objective=objective, random_state=self.random_state, verbose=-1, **params
        )
