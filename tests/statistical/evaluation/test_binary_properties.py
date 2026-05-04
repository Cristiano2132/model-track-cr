"""Property-based statistical tests for BinaryEvaluator using Hypothesis."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from model_track.evaluation import BinaryEvaluator


@given(
    n=st.integers(min_value=10, max_value=200),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=50)
@pytest.mark.filterwarnings("ignore:ks_2samp:RuntimeWarning")
def test_auc_always_in_bounds(n: int, seed: int) -> None:
    """AUC must always be in [0.0, 1.0]."""
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.integers(0, 2, size=n))
    # Ensure at least one of each class
    y_true.iloc[0] = 0
    y_true.iloc[1] = 1
    y_proba = pd.Series(rng.uniform(0, 1, size=n))
    ev = BinaryEvaluator()
    metrics = ev.evaluate(y_true, y_proba)
    assert 0.0 <= metrics["auc"] <= 1.0


@given(
    n=st.integers(min_value=10, max_value=200),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=50)
@pytest.mark.filterwarnings("ignore:ks_2samp:RuntimeWarning")
def test_ks_always_in_bounds(n: int, seed: int) -> None:
    """KS statistic must always be in [0.0, 1.0]."""
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.integers(0, 2, size=n))
    y_true.iloc[0] = 0
    y_true.iloc[1] = 1
    y_proba = pd.Series(rng.uniform(0, 1, size=n))
    ev = BinaryEvaluator()
    metrics = ev.evaluate(y_true, y_proba)
    assert 0.0 <= metrics["ks"] <= 1.0


@given(
    n=st.integers(min_value=10, max_value=200),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=50)
@pytest.mark.filterwarnings("ignore:ks_2samp:RuntimeWarning")
def test_gini_equals_2auc_minus_1(n: int, seed: int) -> None:
    """Gini must always equal 2*AUC - 1."""
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.integers(0, 2, size=n))
    y_true.iloc[0] = 0
    y_true.iloc[1] = 1
    y_proba = pd.Series(rng.uniform(0, 1, size=n))
    ev = BinaryEvaluator()
    metrics = ev.evaluate(y_true, y_proba)
    assert metrics["gini"] == pytest.approx(2 * metrics["auc"] - 1, abs=1e-10)


@given(
    n=st.integers(min_value=10, max_value=200),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=50)
@pytest.mark.filterwarnings("ignore:ks_2samp:RuntimeWarning")
def test_brier_score_always_in_bounds(n: int, seed: int) -> None:
    """Brier score must always be in [0.0, 1.0]."""
    rng = np.random.default_rng(seed)
    y_true = pd.Series(rng.integers(0, 2, size=n))
    y_true.iloc[0] = 0
    y_true.iloc[1] = 1
    y_proba = pd.Series(rng.uniform(0, 1, size=n))
    ev = BinaryEvaluator()
    metrics = ev.evaluate(y_true, y_proba)
    assert 0.0 <= metrics["brier_score"] <= 1.0
