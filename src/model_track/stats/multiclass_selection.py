from typing import Literal

import numpy as np
import pandas as pd

from model_track.base import BaseTransformer
from model_track.stats.metrics import compute_cramers_v
from model_track.woe.ovr_adapter import OvRWoeAdapter


class MulticlassSelector(BaseTransformer):
    """
    Feature selection for multiclass tasks using One-vs-Rest (OvR) IV and Cramer's V.

    Supports multiple IV strategies:
    - "max": Feature passes if max(IV across classes) >= threshold.
    - "mean": Feature passes if mean(IV across classes) >= threshold.
    - "all": Feature passes if all(IV per class) >= threshold.
    """

    def __init__(
        self,
        classes: list[str | int],
        iv_threshold: float = 0.10,
        iv_strategy: Literal["max", "mean", "all"] = "max",
        cramers_threshold: float = 0.85,
        sample_size: int | None = 50000,
    ):
        self.classes = classes
        self.iv_threshold = iv_threshold
        self.iv_strategy = iv_strategy
        self.cramers_threshold = cramers_threshold
        self.sample_size = sample_size

        self.iv_results_: dict[str, dict[str, float]] = {}
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []

    def fit(  # type: ignore[override]
        self, df: pd.DataFrame, target: str, features: list[str] | None = None
    ) -> "MulticlassSelector":
        """
        Evaluate features using OvR IV and Cramer's V.

        Args:
            df: Input DataFrame.
            target: Multiclass target column name.
            features: List of features to evaluate.

        Returns:
            MulticlassSelector: Fitted instance.
        """
        features = features or []
        df_sample = self._sample_data(df, target)

        valid_features = [f for f in features if f in df_sample.columns]
        if not valid_features:
            self.selected_features_ = []
            self.dropped_features_ = []
            return self

        summary = self._compute_iv_summary(df_sample, target, valid_features)
        strong_features = self._filter_by_iv(valid_features, summary)
        selected = self._filter_by_cramers_v(df_sample, strong_features, summary)

        self.selected_features_ = selected
        self.dropped_features_ = [f for f in valid_features if f not in selected]
        self.iv_results_ = summary.to_dict(orient="index")

        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_data(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Stratified sample if data exceeds sample_size."""
        if self.sample_size and len(df) > self.sample_size:
            frac = self.sample_size / len(df)
            return pd.concat(
                [
                    g.sample(frac=frac, random_state=42)
                    for _, g in df.groupby(target, observed=True, sort=False)
                ],
                axis=0,
            )
        return df

    def _compute_iv_summary(
        self, df: pd.DataFrame, target: str, features: list[str]
    ) -> pd.DataFrame:
        """Fit OvR adapter and return IV summary."""
        adapter = OvRWoeAdapter(classes=self.classes)
        adapter.fit(df, target=target, columns=features)
        return adapter.iv_summary()

    def _passes_iv(self, ivs: list[float]) -> bool:
        """Check if a feature's IV list passes the configured strategy."""
        if self.iv_strategy == "max":
            return max(ivs) >= self.iv_threshold
        if self.iv_strategy == "mean":
            return float(np.mean(ivs)) >= self.iv_threshold
        # "all"
        return all(iv >= self.iv_threshold for iv in ivs)

    def _filter_by_iv(self, features: list[str], summary: pd.DataFrame) -> list[str]:
        """Stage 1: Keep features that pass the IV threshold."""
        strong = []
        for feat in features:
            ivs = [summary.loc[feat, f"iv_{c}"] for c in self.classes]
            if self._passes_iv(ivs):
                strong.append(feat)

        # Sort by max_iv descending so strongest features survive correlation filter
        strong.sort(key=lambda x: float(summary.loc[x, "max_iv"]), reverse=True)
        return strong

    def _filter_by_cramers_v(
        self, df: pd.DataFrame, features: list[str], summary: pd.DataFrame
    ) -> list[str]:
        """Stage 2: Remove highly correlated features (Cramer's V) keeping highest IV."""
        to_drop: set[str] = set()
        for i, f1 in enumerate(features):
            if f1 in to_drop:
                continue
            for f2 in features[i + 1 :]:
                if f2 in to_drop:
                    continue
                if compute_cramers_v(df, f1, f2) > self.cramers_threshold:
                    to_drop.add(f2)

        return [f for f in features if f not in to_drop]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove dropped features."""
        return df.drop(
            columns=[f for f in self.dropped_features_ if f in df.columns], errors="ignore"
        )

    def iv_summary(self) -> pd.DataFrame:
        """Return summary of IV results and selection status."""
        if not self.iv_results_:
            raise RuntimeError("MulticlassSelector must be fitted first.")

        summary = pd.DataFrame.from_dict(self.iv_results_, orient="index")
        summary["selected"] = summary.index.isin(self.selected_features_)
        summary.index.name = "feature"
        return summary
