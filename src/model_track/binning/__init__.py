"""
Binning module for model-track.
Provides supervised and unsupervised methods to discretize continuous variables.

Example:
    >>> from model_track.binning import TreeBinner
    >>> binner = TreeBinner(max_depth=3)
    >>> binner.fit(df, column="age", target="survived")
    >>> df["age_binned"] = binner.transform(df, column="age")
"""

from model_track.binning.bin_applier import BinApplier
from model_track.binning.quantile_binner import QuantileBinner
from model_track.binning.tree_binner import TreeBinner

__all__ = ["TreeBinner", "QuantileBinner", "BinApplier"]
