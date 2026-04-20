"""
model-track: A professional library for model tracking, feature engineering, and statistical analysis.

This package provides tools for:
- Data auditing and schema comparison.
- Memory optimization for large datasets.
- Supervised and unsupervised binning.
- Weight of Evidence (WoE) calculation and stability analysis.
- Statistical feature selection.

Example:
    >>> from model_track.preprocessing import DataOptimizer
    >>> from model_track.stats import StatisticalSelector
    >>> # Quick optimization
    >>> df = DataOptimizer.reduce_mem_usage(df)
    >>> # Statistical selection
    >>> selector = StatisticalSelector()
    >>> df_selected = selector.fit_transform(df, target="target", features=df.columns)
"""

__version__ = "0.1.0"

from model_track.base import BaseTransformer
from model_track.context import ProjectContext

__all__ = ["BaseTransformer", "ProjectContext"]
