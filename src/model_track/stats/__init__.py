"""
Stats module for model-track.
Provides statistical metrics and feature selection algorithms.

Example:
    >>> from model_track.stats import StatisticalSelector
    >>> selector = StatisticalSelector(iv_threshold=0.1)
    >>> selector.fit(df, target="target", features=["feat1", "feat2"])
    >>> selected_df = selector.transform(df)
"""

from model_track.stats.metrics import compute_cramers_v, compute_iv
from model_track.stats.selection import StatisticalSelector

__all__ = ["compute_cramers_v", "compute_iv", "StatisticalSelector"]
