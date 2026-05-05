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
        df_sample = df

        if self.sample_size and len(df) > self.sample_size:
            frac = self.sample_size / len(df)
            df_sample = pd.concat(
                [
                    g.sample(frac=frac, random_state=42)
                    for _, g in df.groupby(target, observed=True, sort=False)
                ],
                axis=0,
            )

        valid_features = [f for f in features if f in df_sample.columns]
        if not valid_features:
            self.selected_features_ = []
            self.dropped_features_ = []
            return self

        # 1. Compute OvR IV
        adapter = OvRWoeAdapter(classes=self.classes)
        adapter.fit(df_sample, target=target, columns=valid_features)
        summary = adapter.iv_summary()

        strong_features = []
        for feat in valid_features:
            ivs = [summary.loc[feat, f"iv_{c}"] for c in self.classes]

            if self.iv_strategy == "max":
                pass_iv = max(ivs) >= self.iv_threshold
            elif self.iv_strategy == "mean":
                pass_iv = float(np.mean(ivs)) >= self.iv_threshold
            else:  # "all"
                pass_iv = all(iv >= self.iv_threshold for iv in ivs)

            if pass_iv:
                strong_features.append(feat)

        # 2. Sort by max_iv (higher IV "wins" in correlation filter)
        strong_features.sort(key=lambda x: float(summary.loc[x, "max_iv"]), reverse=True)

        # 3. Correlation Filter (Cramer's V)
        to_drop_corr = set()
        for i, f1 in enumerate(strong_features):
            if f1 in to_drop_corr:
                continue
            for f2 in strong_features[i + 1 :]:
                if f2 in to_drop_corr:
                    continue
                v = compute_cramers_v(df_sample, f1, f2)
                if v > self.cramers_threshold:
                    to_drop_corr.add(f2)

        self.selected_features_ = [f for f in strong_features if f not in to_drop_corr]
        self.dropped_features_ = [f for f in valid_features if f not in self.selected_features_]

        # Store for summary
        self.iv_results_ = summary.to_dict(orient="index")

        return self

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
