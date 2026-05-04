"""
WoE (Weight of Evidence) module for model-track.
Tools for WoE calculation, stability analysis, and category mapping.

Example:
    >>> from model_track.woe import WoeCalculator, WoeStability, OvRWoeAdapter
    >>> # Calculate WoE
    >>> calc = WoeCalculator()
    >>> calc.fit(df, target="target", columns=["category_col"])
    >>> df_woe = calc.transform(df, columns=["category_col"])
    >>> # Analyze stability
    >>> ws = WoeStability(date_col="period")
    >>> matrix = ws.calculate_stability_matrix(df, "category_col", "target")
"""

from model_track.woe.calculator import WoeCalculator
from model_track.woe.ovr_adapter import OvRWoeAdapter
from model_track.woe.stability import CategoryMapper, WoeStability

__all__ = ["WoeCalculator", "WoeStability", "CategoryMapper", "OvRWoeAdapter"]
