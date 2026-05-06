import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from model_track.base import BinaryAdapter, RegressionAdapter
from model_track.tuning import LGBMTuner
from model_track.tuning.bayesian import HAS_BAYES
from model_track.tuning.lgbm import HAS_LGBM

# Markers for optional dependencies
requires_tuning = pytest.mark.skipif(
    not (HAS_BAYES and HAS_LGBM),
    reason="Tuning dependencies (lightgbm, bayesian-optimization) not installed",
)


@requires_tuning
def test_integration_lgbm_tuning_binary():
    # 1. Setup data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    ser_y = pd.Series(y)

    # 2. Tune
    tuner = LGBMTuner(task=BinaryAdapter(), n_iter=3, init_points=2, random_state=42)
    # Small iterations for speed in integration tests
    best_params = tuner.tune(df_X, ser_y)

    assert isinstance(best_params, dict)
    assert "num_leaves" in best_params

    # 3. Use tuned model
    model = tuner.get_model()
    model.fit(df_X, ser_y)

    preds = model.predict_proba(df_X)[:, 1]
    assert len(preds) == 100


@requires_tuning
def test_integration_lgbm_tuning_regression():
    # 1. Setup data
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    df_X = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    ser_y = pd.Series(y)

    # 2. Tune
    tuner = LGBMTuner(task=RegressionAdapter(), n_iter=3, init_points=2, random_state=42)
    tuner.tune(df_X, ser_y)

    assert tuner.best_score_ < 0  # negative RMSE

    # 3. Use tuned model
    model = tuner.get_model()
    model.fit(df_X, ser_y)

    preds = model.predict(df_X)
    assert len(preds) == 100
