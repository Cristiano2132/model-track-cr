"""
Tuning module for hyperparameter optimization.
Provides Bayesian optimization wrappers for ML models.
"""

from typing import Any

from model_track.tuning.base import BaseTuner

__all__ = ["BaseTuner", "BayesianTuner", "LGBMTuner"]

try:
    from model_track.tuning.bayesian import BayesianTuner
    from model_track.tuning.lgbm import LGBMTuner
except ImportError:  # pragma: no cover
    # Fallback classes that raise error when instantiated

    class BayesianTuner:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "'BayesianTuner' requires 'tuning' extra dependencies. "
                "Install them with 'pip install model-track-cr[tuning]'."
            )

    class LGBMTuner:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "'LGBMTuner' requires 'tuning' extra dependencies. "
                "Install them with 'pip install model-track-cr[tuning]'."
            )
