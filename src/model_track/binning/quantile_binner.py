import numpy as np
import pandas as pd

from model_track.base import BaseTransformer


class QuantileBinner(BaseTransformer):
    """
    Unsupervised binner that discretizes continuous variables into intervals of equal frequency.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bins: list[float] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, column: str = "") -> "QuantileBinner":  # type: ignore[override]
        """
        Learn the split points based on quantiles.

        Args:
            df: Input DataFrame.
            column: Name of the feature to be binned.

        Returns:
            QuantileBinner: The fitted instance.
        """
        data = df[column].dropna()
        if len(data) == 0:
            self.bins = []
            self._is_fitted = True
            return self

        # Generate quantiles
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.quantile(data, quantiles))

        # We keep only the internal split points for consistency with TreeBinner
        # TreeBinner stores thresholds. For N bins, we have N-1 thresholds.
        # np.quantile with 10 bins gives 11 edges.
        # We drop the first and last (which are replaced by -inf/inf in transform)
        if len(edges) > 2:
            self.bins = sorted(edges[1:-1].tolist())
        else:
            self.bins = []

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, column: str = "") -> pd.Series:
        """
        Apply the learned quantile bins and handle null values.

        Args:
            df: Input DataFrame.
            column: Name of the feature to transform.

        Returns:
            pd.Series: The binned feature as strings.
        """
        if not self._is_fitted:
            raise RuntimeError("The binner must be fitted before transforming.")

        # We add -inf and inf to cover the entire range
        split_points = [-float("inf")] + (self.bins or []) + [float("inf")]

        # Apply bins
        binned = pd.cut(df[column], bins=split_points, duplicates="drop")

        # Convert to string and treat nulls as 'N/A'
        return binned.astype(str).replace("nan", "N/A")
