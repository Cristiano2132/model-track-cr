"""
OvRWoeAdapter — multiclass WoE via One-vs-Rest strategy.

For multiclass targets, WoE and IV are binary metrics by design.
This adapter trains one :class:`WoeCalculator` per class using the
One-vs-Rest (OvR) strategy: for each class ``k``, observations
belonging to ``k`` are treated as "events" (1) and all others as
"non-events" (0).

Example::

    from model_track.woe import OvRWoeAdapter

    adapter = OvRWoeAdapter(classes=["A", "B", "C"])
    adapter.fit(df, target="risk_tier", columns=["income_cat", "region"])

    # One WoE column per class per feature
    df_woe = adapter.transform(df, columns=["income_cat", "region"], strategy="per_class")
    # → income_cat_woe_A, income_cat_woe_B, income_cat_woe_C, region_woe_A, …

    # Single WoE column per feature (class with highest IV wins)
    df_woe = adapter.transform(df, columns=["income_cat", "region"], strategy="max_iv")
    # → income_cat_woe, region_woe

    iv_df = adapter.iv_summary()
    # feature | iv_A | iv_B | iv_C | max_iv
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from model_track.woe.calculator import WoeCalculator


class OvRWoeAdapter:
    """Multiclass WoE adapter using the One-vs-Rest (OvR) strategy.

    Trains one :class:`WoeCalculator` per class and provides two
    transform strategies:

    * ``"per_class"`` — produces ``n_classes × n_features`` WoE columns
      named ``{col}_woe_{k}``.
    * ``"max_iv"`` — produces ``n_features`` WoE columns named ``{col}_woe``
      using, for each feature, the class whose WoE mapping has the highest IV.

    Args:
        classes: Ordered list of class labels present in the target column.
    """

    def __init__(self, classes: list[str | int]) -> None:
        if not classes:
            raise ValueError("`classes` must not be empty.")
        self.classes: list[str | int] = classes
        self.calculators_: dict[str, WoeCalculator] = {}
        self.iv_per_class_: dict[str, dict[str, float]] = {}
        self._fitted_columns: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iv(
        df: pd.DataFrame,
        binary_target: str,
        feature: str,
    ) -> float:
        """Compute IV for a single feature against a binary target.

        Uses the same Laplace-smoothed statistics as :class:`WoeCalculator`.

        Args:
            df: DataFrame containing *feature* and *binary_target* columns.
            binary_target: Name of the binary (0/1) target column.
            feature: Feature column name (already cast to ``str``).

        Returns:
            float: Information Value for the feature.
        """
        stats = df.groupby(feature, observed=True)[binary_target].agg(["count", "sum"])
        stats.columns = pd.Index(["Total", "Bad"])
        stats["Good"] = stats["Total"] - stats["Bad"]

        total_bad: float = float(stats["Bad"].sum())
        total_good: float = float(stats["Good"].sum())

        perc_bad = (stats["Bad"] + 0.5) / (total_bad + 0.5)
        perc_good = (stats["Good"] + 0.5) / (total_good + 0.5)

        woe = np.log(perc_good / perc_bad)
        iv_series = (perc_good - perc_bad) * woe
        return float(iv_series.sum())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        columns: list[str],
    ) -> OvRWoeAdapter:
        """Fit one WoeCalculator per class using OvR binary targets.

        For each class ``k`` in :attr:`classes`, a temporary binary column
        ``target_k`` is created (``1`` if ``df[target] == k``, else ``0``)
        and a :class:`WoeCalculator` is trained on it.

        IV is computed for every (class, feature) pair and stored in
        :attr:`iv_per_class_`.

        Args:
            df: Training DataFrame.
            target: Name of the multiclass target column.
            columns: Feature columns to fit WoE on.

        Returns:
            OvRWoeAdapter: The fitted adapter (``self``).

        Raises:
            ValueError: If *columns* is empty or *target* not in *df*.
        """
        if not columns:
            raise ValueError("`columns` must not be empty.")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        self._fitted_columns = list(columns)
        self.calculators_ = {}
        self.iv_per_class_ = {}

        for cls in self.classes:
            key = str(cls)
            binary_target = f"__ovr_target_{key}__"

            # Build OvR binary view (no copy of the full DataFrame)
            binary_col = (df[target] == cls).astype(int)
            binary_series = pd.Series(binary_col.values, index=df.index, name=binary_target)

            calc = WoeCalculator()
            calc.fit(
                df.assign(**{binary_target: binary_series}),
                target=binary_target,
                columns=columns,
            )
            self.calculators_[key] = calc

            # Compute IV per feature for this class
            iv_for_class: dict[str, float] = {}
            tmp_df = df.assign(**{binary_target: binary_series})
            for col in columns:
                str_col = f"__str_{col}__"
                tmp_df = tmp_df.assign(**{str_col: tmp_df[col].astype(str).fillna("N/A")})
                iv_for_class[col] = self._compute_iv(tmp_df, binary_target, str_col)
            self.iv_per_class_[key] = iv_for_class

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        columns: list[str],
        strategy: Literal["per_class", "max_iv"] = "per_class",
    ) -> pd.DataFrame:
        """Apply fitted WoE mappings to the DataFrame.

        Args:
            df: Input DataFrame (train or test).
            columns: Feature columns to transform. Must be a subset of the
                columns used during :meth:`fit`.
            strategy: Output strategy.

                * ``"per_class"`` — one ``{col}_woe_{k}`` column per
                  class per feature.
                * ``"max_iv"`` — one ``{col}_woe`` column per feature,
                  using the class with the highest IV for that feature.

        Returns:
            pd.DataFrame: Copy of *df* with additional WoE columns.

        Raises:
            RuntimeError: If the adapter has not been fitted yet.
            ValueError: If *strategy* is not recognised.
        """
        if not self._is_fitted:
            raise RuntimeError("OvRWoeAdapter must be fitted before calling transform().")
        if strategy not in ("per_class", "max_iv"):
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'per_class' or 'max_iv'.")

        df_out = df.copy()

        if strategy == "per_class":
            for cls in self.classes:
                key = str(cls)
                calc = self.calculators_[key]
                # Transform produces {col}_woe columns; rename to {col}_woe_{k}
                df_tmp = calc.transform(df_out, columns=columns)
                for col in columns:
                    src = f"{col}_woe"
                    dst = f"{col}_woe_{key}"
                    if src in df_tmp.columns:
                        df_out[dst] = df_tmp[src]

        else:  # "max_iv"
            for col in columns:
                # Pick the class with highest IV for this feature
                best_cls = max(
                    self.classes,
                    key=lambda k: self.iv_per_class_.get(str(k), {}).get(col, 0.0),
                )
                key = str(best_cls)
                calc = self.calculators_[key]
                df_tmp = calc.transform(df_out, columns=[col])
                src = f"{col}_woe"
                if src in df_tmp.columns:
                    df_out[src] = df_tmp[src]

        return df_out

    def iv_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising IV per class per feature.

        Returns:
            pd.DataFrame: Index = feature name. Columns = ``iv_{k}`` for
            each class, plus ``max_iv`` (maximum IV across all classes for
            that feature).

        Raises:
            RuntimeError: If the adapter has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("OvRWoeAdapter must be fitted before calling iv_summary().")

        rows: dict[str, dict[str, float]] = {}
        for col in self._fitted_columns:
            row: dict[str, float] = {}
            for cls in self.classes:
                key = str(cls)
                iv_val = self.iv_per_class_.get(key, {}).get(col, 0.0)
                row[f"iv_{key}"] = iv_val
            row["max_iv"] = max(row.values())
            rows[col] = row

        summary = pd.DataFrame.from_dict(rows, orient="index")
        summary.index.name = "feature"
        return summary
