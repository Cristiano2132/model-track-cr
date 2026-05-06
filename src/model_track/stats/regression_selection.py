import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from model_track.base import BaseTransformer


class RegressionSelector(BaseTransformer):
    """
    Feature selector for regression tasks.
    Filters features based on:
    1. Minimum absolute correlation with the target.
    2. Maximum absolute correlation between feature pairs.
    3. Variance Inflation Factor (VIF) to reduce multicollinearity.
    """

    def __init__(
        self,
        method: str = "pearson",
        min_correlation: float = 0.05,
        correlation_threshold: float = 0.90,
        vif_threshold: float = 10.0,
    ):
        """
        Args:
            method: 'pearson' or 'spearman' for correlation calculation.
            min_correlation: Minimum absolute correlation with the target.
            correlation_threshold: Maximum allowed absolute correlation between pairs.
            vif_threshold: Maximum allowed VIF.
        """
        if method not in ("pearson", "spearman"):
            raise ValueError("method must be 'pearson' or 'spearman'")

        self.method = method
        self.min_correlation = min_correlation
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold

        self.selected_features_: list[str] = []
        self.dropped_features_: dict[str, str] = {}  # feature -> reason
        self.target_corr_: dict[str, float] = {}
        self.vif_results_: dict[str, float] = {}

    def fit(  # type: ignore[override]
        self, df: pd.DataFrame, target: str, features: list[str] | None = None
    ) -> "RegressionSelector":
        """
        Evaluate features and define which ones will be kept.

        Args:
            df: Input DataFrame.
            target: Target column name.
            features: List of features to evaluate. If None, all numeric columns except target.

        Returns:
            RegressionSelector: The fitted selector instance.
        """
        if features is None:
            features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]

        valid_features = [f for f in features if f in df.columns]

        non_constant = self._filter_zero_variance(df, valid_features)
        passing_target_corr = self._filter_by_target_correlation(df, target, non_constant)
        passing_pair_corr = self._filter_by_pair_correlation(df, passing_target_corr)
        self.selected_features_ = self._filter_by_vif(df, passing_pair_corr)

        return self

    # ------------------------------------------------------------------
    # Private filtering stages
    # ------------------------------------------------------------------

    def _filter_zero_variance(self, df: pd.DataFrame, features: list[str]) -> list[str]:
        """Stage 0: Remove constant features."""
        valid = []
        for f in features:
            if df[f].nunique(dropna=True) <= 1:
                self.dropped_features_[f] = "zero_variance"
            else:
                valid.append(f)
        return valid

    def _filter_by_target_correlation(
        self, df: pd.DataFrame, target: str, features: list[str]
    ) -> list[str]:
        """Stage 1: Remove features with low correlation to target."""
        target_corr: dict[str, float] = {}
        passing = []

        for f in features:
            corr = df[f].corr(df[target], method=self.method)
            if pd.isna(corr):
                self.dropped_features_[f] = "zero_variance"
                continue

            abs_corr = abs(corr)
            target_corr[f] = abs_corr

            if abs_corr >= self.min_correlation:
                passing.append(f)
            else:
                self.dropped_features_[f] = "low_target_correlation"

        self.target_corr_ = target_corr
        # Sort by target correlation descending so strongest features survive pair filter
        passing.sort(key=lambda x: self.target_corr_[x], reverse=True)
        return passing

    def _filter_by_pair_correlation(self, df: pd.DataFrame, features: list[str]) -> list[str]:
        """Stage 2: Remove features with high pairwise correlation."""
        if not features:
            return []

        corr_matrix = df[features].corr(method=self.method).abs()
        to_drop: set[str] = set()
        passing = []

        for i, f1 in enumerate(features):
            if f1 in to_drop:
                continue
            passing.append(f1)
            for f2 in features[i + 1 :]:
                if f2 in to_drop:
                    continue  # pragma: no cover
                if corr_matrix.loc[f1, f2] > self.correlation_threshold:
                    to_drop.add(f2)
                    self.dropped_features_[f2] = "high_pair_correlation"

        return passing

    def _compute_vif_scores(self, vif_data: pd.DataFrame, features: list[str]) -> dict[str, float]:
        """Compute VIF for each feature in the list."""
        vif_scores: dict[str, float] = {}
        for target_feature in features:
            predictors = [f for f in features if f != target_feature]
            X = vif_data[predictors].values
            y = vif_data[target_feature].values

            if np.var(y) == 0:
                vif_scores[target_feature] = float("inf")
                continue

            lr = LinearRegression()
            lr.fit(X, y)
            r2 = lr.score(X, y)
            vif_scores[target_feature] = float("inf") if r2 >= 0.99999 else 1.0 / (1.0 - r2)

        return vif_scores

    def _filter_by_vif(self, df: pd.DataFrame, features: list[str]) -> list[str]:
        """Stage 3: Iteratively remove features with highest VIF above threshold."""
        current = features.copy()
        vif_data = df[current].dropna()

        while len(current) > 1:
            vif_scores = self._compute_vif_scores(vif_data, current)
            max_feature = max(vif_scores, key=vif_scores.get)  # type: ignore
            max_vif = vif_scores[max_feature]

            if max_vif > self.vif_threshold:
                current.remove(max_feature)
                self.dropped_features_[max_feature] = "high_vif"
                self.vif_results_[max_feature] = max_vif
            else:
                for f in current:
                    self.vif_results_[f] = vif_scores[f]
                break

        if len(current) == 1:
            self.vif_results_[current[0]] = 1.0

        return current

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove discarded features from the DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with selected features only.
        """
        cols_to_drop = [f for f in self.dropped_features_ if f in df.columns]
        return df.drop(columns=cols_to_drop, errors="ignore")

    def summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing the selection process.
        """
        all_features = set(self.selected_features_) | set(self.dropped_features_)

        summary_data = []
        for f in all_features:
            is_selected = f in self.selected_features_
            reason = self.dropped_features_.get(f, "kept")
            target_corr = self.target_corr_.get(f, np.nan)
            vif = self.vif_results_.get(f, np.nan)

            summary_data.append(
                {
                    "feature": f,
                    "target_corr": target_corr,
                    "vif": vif,
                    "selected": is_selected,
                    "drop_reason": reason,
                }
            )

        summary_df = pd.DataFrame(summary_data).sort_values(
            by=["selected", "target_corr"], ascending=[False, False]
        )
        return summary_df.reset_index(drop=True)
