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

        # 0. Check for zero variance
        non_constant_features = []
        for f in valid_features:
            if df[f].nunique(dropna=True) <= 1:
                self.dropped_features_[f] = "zero_variance"
            else:
                non_constant_features.append(f)

        # 1. Target Correlation
        target_corr = {}
        features_passing_target_corr = []

        for f in non_constant_features:
            corr = df[f].corr(df[target], method=self.method)
            # Handle NaN correlation (e.g., constant in sample after dropna)
            if pd.isna(corr):
                self.dropped_features_[f] = "zero_variance"
                continue

            abs_corr = abs(corr)
            target_corr[f] = abs_corr

            if abs_corr >= self.min_correlation:
                features_passing_target_corr.append(f)
            else:
                self.dropped_features_[f] = "low_target_correlation"

        self.target_corr_ = target_corr

        # 2. Pairwise Correlation
        # Sort features by target correlation descending
        features_passing_target_corr.sort(key=lambda x: self.target_corr_[x], reverse=True)

        features_passing_pair_corr = []
        if features_passing_target_corr:
            # Calculate full correlation matrix for remaining features
            corr_matrix = df[features_passing_target_corr].corr(method=self.method).abs()

            to_drop_corr = set()
            for i, f1 in enumerate(features_passing_target_corr):
                if f1 in to_drop_corr:
                    continue

                features_passing_pair_corr.append(f1)

                for f2 in features_passing_target_corr[i + 1 :]:
                    if f2 in to_drop_corr:
                        continue

                    if corr_matrix.loc[f1, f2] > self.correlation_threshold:
                        to_drop_corr.add(f2)
                        self.dropped_features_[f2] = "high_pair_correlation"

        # 3. Iterative VIF
        current_features = features_passing_pair_corr.copy()

        # Prepare data for VIF (dropna to ensure LinearRegression works)
        vif_data = df[current_features].dropna()

        while len(current_features) > 1:
            vif_scores = {}
            for _i, target_feature in enumerate(current_features):
                predictor_features = [f for f in current_features if f != target_feature]

                X = vif_data[predictor_features].values
                y = vif_data[target_feature].values

                # Check for constant target in subset
                if np.var(y) == 0:
                    vif_scores[target_feature] = float("inf")
                    continue

                lr = LinearRegression()
                lr.fit(X, y)

                # R^2 score
                r2 = lr.score(X, y)

                # Handle edge cases where R^2 is 1.0 (perfect multicollinearity)
                if r2 >= 0.99999:
                    vif = float("inf")
                else:
                    vif = 1.0 / (1.0 - r2)

                vif_scores[target_feature] = vif

            max_vif_feature = max(vif_scores, key=lambda k: vif_scores[k])
            max_vif = vif_scores[max_vif_feature]

            if max_vif > self.vif_threshold:
                current_features.remove(max_vif_feature)
                self.dropped_features_[max_vif_feature] = "high_vif"
                self.vif_results_[max_vif_feature] = max_vif
            else:
                # All remaining features pass VIF threshold
                for f in current_features:
                    self.vif_results_[f] = vif_scores[f]
                break

        # If loop exits due to 1 feature left, calculate its VIF as 1.0
        if len(current_features) == 1:
            self.vif_results_[current_features[0]] = 1.0

        self.selected_features_ = current_features

        return self

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
