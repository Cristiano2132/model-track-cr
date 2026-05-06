from unittest.mock import Mock

from model_track.tuning.bayesian import BayesianTuner


def test_base_tuner_process_params():
    class DummyTuner(BayesianTuner):
        def _create_model(self, params):
            return None

    tuner = DummyTuner(task=Mock())
    assert tuner._process_params({"a": 1}) == {"a": 1}
