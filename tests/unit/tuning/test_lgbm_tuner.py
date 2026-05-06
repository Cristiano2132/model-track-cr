from unittest.mock import patch

import pandas as pd
import pytest

from model_track.base import BinaryAdapter, MulticlassAdapter, RegressionAdapter
from model_track.tuning import LGBMTuner
from model_track.tuning.bayesian import HAS_BAYES
from model_track.tuning.lgbm import HAS_LGBM

# Markers for optional dependencies
requires_tuning = pytest.mark.skipif(
    not (HAS_BAYES and HAS_LGBM),
    reason="Tuning dependencies (lightgbm, bayesian-optimization) not installed",
)


@pytest.fixture
def sample_data():
    X = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    return X, y


@requires_tuning
def test_lgbm_tuner_initialization():
    tuner = LGBMTuner(task=BinaryAdapter())
    assert tuner.cv_folds == 3
    assert "num_leaves" in tuner.param_bounds


@requires_tuning
def test_lgbm_tuner_process_params():
    tuner = LGBMTuner(task=BinaryAdapter())
    raw_params = {"num_leaves": 31.5, "max_depth": 5.9, "learning_rate": 0.1}
    processed = tuner._process_params(raw_params)
    assert processed["num_leaves"] == 31
    assert processed["max_depth"] == 5
    assert processed["learning_rate"] == 0.1


@requires_tuning
@patch("model_track.tuning.bayesian.BayesianOptimization")
def test_lgbm_tuner_mocked_tune(mock_bayes, sample_data):
    X, y = sample_data
    # Mock optimizer.max
    mock_instance = mock_bayes.return_value
    mock_instance.max = {"params": {"num_leaves": 31, "max_depth": 5}, "target": 0.85}

    tuner = LGBMTuner(task=BinaryAdapter(), n_iter=1, init_points=1)
    best_params = tuner.tune(X, y)

    assert best_params["num_leaves"] == 31
    assert tuner.best_score_ == 0.85
    mock_instance.maximize.assert_called_once()


@requires_tuning
def test_lgbm_tuner_create_model_binary():
    tuner = LGBMTuner(task=BinaryAdapter())
    model = tuner._create_model({"num_leaves": 20})
    from lightgbm import LGBMClassifier

    assert isinstance(model, LGBMClassifier)
    assert model.objective == "binary"


@requires_tuning
def test_lgbm_tuner_create_model_multiclass():
    tuner = LGBMTuner(task=MulticlassAdapter())
    model = tuner._create_model({"num_leaves": 20})
    from lightgbm import LGBMClassifier

    assert isinstance(model, LGBMClassifier)
    assert model.objective == "multiclass"


@requires_tuning
def test_lgbm_tuner_create_model_regression():
    tuner = LGBMTuner(task=RegressionAdapter())
    model = tuner._create_model({"num_leaves": 20})
    from lightgbm import LGBMRegressor

    assert isinstance(model, LGBMRegressor)


@requires_tuning
def test_lgbm_tuner_get_scoring_metric():
    assert LGBMTuner(task=BinaryAdapter())._get_scoring_metric() == "roc_auc"
    assert LGBMTuner(task=MulticlassAdapter())._get_scoring_metric() == "roc_auc_ovr"
    assert (
        LGBMTuner(task=RegressionAdapter())._get_scoring_metric() == "neg_root_mean_squared_error"
    )


@requires_tuning
def test_tuner_error_if_not_tuned():
    tuner = LGBMTuner(task=BinaryAdapter())
    with pytest.raises(ValueError, match="Tuner has not been fitted yet"):
        tuner.get_model()


def test_lgbm_tuner_no_lgbm_error():
    with patch("model_track.tuning.lgbm.HAS_LGBM", False):
        # In CI, if bayesian-optimization is also missing, it might raise that error first
        # because LGBMTuner calls super().__init__ which checks HAS_BAYES.
        with pytest.raises(ImportError, match="is required"):
            LGBMTuner(task=BinaryAdapter())


def test_bayesian_tuner_no_bayes_error():
    from model_track.tuning.bayesian import BayesianTuner

    class DummyTuner(BayesianTuner):
        def _create_model(self, params):
            return None

    with patch("model_track.tuning.bayesian.HAS_BAYES", False):
        with pytest.raises(ImportError, match="'bayesian-optimization' is required"):
            DummyTuner(task=BinaryAdapter())
