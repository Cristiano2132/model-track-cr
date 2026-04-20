import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from model_track.base import BaseTransformer


class TreeBinner(BaseTransformer):
    """
    Supervised binner that uses decision trees to find optimal split points.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 100):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bins: list[float] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, column: str = "", target: str = "") -> "TreeBinner":  # type: ignore[override]
        """
        Learn the split points using a decision tree.

        Args:
            df: Input DataFrame.
            column: Name of the feature to be binned.
            target: Name of the target variable.

        Returns:
            TreeBinner: The fitted instance.
        """
        # Cleanup for training (decision trees don't natively handle NaNs)
        df_clean = df[[column, target]].dropna()

        x = df_clean[[column]]
        y = df_clean[target]

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            ccp_alpha=0.0,
        )
        tree.fit(x, y)

        # Extrair thresholds e ordenar
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
        self.bins = sorted(thresholds.tolist())
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, column: str = "") -> pd.Series:
        """
        Apply the learned bins and handle null values.

        Args:
            df: Input DataFrame.
            column: Name of the feature to transform.

        Returns:
            pd.Series: The binned feature as strings.
        """
        if not self._is_fitted:
            raise RuntimeError("The binner must be fitted before transforming.")

        # We add -inf and inf to cover the entire range (with fallback for empty list)
        split_points = [-float("inf")] + (self.bins or []) + [float("inf")]

        # Gerar os labels/bins
        binned = pd.cut(df[column], bins=split_points, duplicates="drop")

        # Convert to string and treat nulls as 'N/A'
        return binned.astype(str).replace("nan", "N/A")
