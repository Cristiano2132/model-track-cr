from typing import Any, cast

import numpy as np
import pandas as pd

from ..context import ProjectContext


class PSICalculator:
    """
    Population Stability Index (PSI) Calculator.
    Measures distribution shift between baseline (training) and current data.
    """

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-6):
        self.n_bins = n_bins
        self.epsilon = epsilon
        self.reference_stats_: dict[str, dict[str, Any]] = {}
        self.psi_results_: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, features: list[str]) -> "PSICalculator":
        """Learn reference distribution from baseline data."""
        self.reference_stats_ = {}
        for col in features:
            data = df[col].dropna()
            if pd.api.types.is_numeric_dtype(data):
                # Using quantiles for numerical features
                # Add -inf and inf to ensure all data is captured in transform
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                bins = np.unique(np.quantile(data, quantiles))
                if len(bins) > 1:
                    bins[0] = -np.inf
                    bins[-1] = np.inf

                counts, bin_edges = np.histogram(data, bins=bins)
                n_bins_count = len(counts)
                # Laplace smoothing
                dist = (counts + self.epsilon) / (len(data) + self.epsilon * n_bins_count)

                self.reference_stats_[col] = {
                    "type": "numerical",
                    "bins": bin_edges.tolist(),
                    "expected_dist": dist.tolist(),
                }
            else:
                # Using unique values for categorical features
                counts = data.value_counts(normalize=False)
                n_bins_count = len(counts)
                dist = (counts + self.epsilon) / (len(data) + self.epsilon * n_bins_count)

                self.reference_stats_[col] = {
                    "type": "categorical",
                    "values": counts.index.tolist(),
                    "expected_dist": dist.values.tolist(),
                }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PSI for each feature in the provided dataframe."""
        self.psi_results_ = {}
        for col, ref in self.reference_stats_.items():
            if col not in df.columns:
                continue

            current_data = df[col].dropna()
            if len(current_data) == 0:
                continue

            if ref["type"] == "numerical":
                bins = np.array(ref["bins"])
                actual_counts, _ = np.histogram(current_data, bins=bins)
            else:
                # Map current data to the same categorical values
                cat_counts = []
                values = cast(list[Any], ref["values"])
                for val in values:
                    cat_counts.append(int((current_data == val).sum()))
                actual_counts = np.array(cat_counts)

            # Normalize with Laplace smoothing
            total_current = len(current_data)
            n_bins_current = len(actual_counts)
            actual_dist = (actual_counts + self.epsilon) / (
                total_current + self.epsilon * n_bins_current
            )
            expected_dist = np.array(ref["expected_dist"])

            # PSI Calculation: (Actual% - Expected%) * ln(Actual% / Expected%)
            psi_val = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
            self.psi_results_[col] = float(psi_val)

        return self.summary()

    def summary(self) -> pd.DataFrame:
        """Returns a summary table of PSI results."""
        data = []
        for col, psi in self.psi_results_.items():
            if psi < 0.10:
                status = "Stable"
            elif psi < 0.25:
                status = "Monitor"
            else:
                status = "Unstable"

            data.append({"feature": col, "psi": psi, "status": status})

        return pd.DataFrame(data)

    def flag_unstable(self, threshold: float = 0.25) -> list[str]:
        """Returns feature names with PSI above threshold."""
        return [col for col, psi in self.psi_results_.items() if psi >= threshold]

    @classmethod
    def from_context(cls, ctx: ProjectContext) -> "PSICalculator":
        """Load reference stats from a ProjectContext."""
        calc = cls()
        calc.reference_stats_ = getattr(ctx, "reference_stats", {}).copy()
        return calc

    def to_context(self, ctx: ProjectContext) -> None:
        """Save reference stats to a ProjectContext."""
        ctx.reference_stats = self.reference_stats_


class ModelPSI(PSICalculator):
    """
    Specialized PSI Calculator for model scores/probabilities.
    Focuses on a single score column and typically uses fixed deciles.
    """

    def __init__(self, n_bins: int = 10, epsilon: float = 1e-6):
        super().__init__(n_bins=n_bins, epsilon=epsilon)
        self.score_col_: str | None = None

    def fit(self, df: pd.DataFrame, score_col: str) -> "ModelPSI":  # type: ignore[override]
        """Learn reference distribution for the score column."""
        self.score_col_ = score_col
        super().fit(df, [score_col])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate PSI for the score column."""
        if self.score_col_ is None:
            raise ValueError("ModelPSI must be fitted or loaded from context first.")
        return super().transform(df)

    def get_psi(self) -> float:
        """Returns the scalar PSI value for the score."""
        if not self.psi_results_ or self.score_col_ not in self.psi_results_:
            return 0.0
        return self.psi_results_[self.score_col_]
