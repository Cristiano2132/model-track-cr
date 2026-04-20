import numpy as np
import pandas as pd

from model_track.base import BaseTransformer


class WoeCalculator(BaseTransformer):
    """Calculates and applies Weight of Evidence (WoE) for categorical variables."""

    def __init__(self) -> None:
        self.mapping_: dict[str, dict[str, float]] = {}
        self._is_fitted = False

    def _compute_mapping(self, df: pd.DataFrame, target: str, feature: str) -> dict[str, float]:
        """
        Calculate the WoE dictionary with Laplace Smoothing.

        Args:
            df: Input DataFrame.
            target: Target column name.
            feature: Feature column name.

        Returns:
            dict[str, float]: Mapping of category values to WoE values.
        """
        stats = df.groupby(feature, observed=True)[target].agg(["count", "sum"])
        stats.columns = ["Total", "Bad"]
        stats["Good"] = stats["Total"] - stats["Bad"]

        perc_bad = (stats["Bad"] + 0.5) / (stats["Bad"].sum() + 0.5)
        perc_good = (stats["Good"] + 0.5) / (stats["Good"].sum() + 0.5)

        return np.log(perc_good / perc_bad).to_dict()  # type: ignore[no-any-return]

    def fit(  # type: ignore[override]
        self, df: pd.DataFrame, target: str, columns: list[str] | None = None
    ) -> "WoeCalculator":
        """
        Fit the WoE mappings for the specified columns.

        Args:
            df: Input DataFrame.
            target: Target column name.
            columns: List of columns to calculate WoE for.

        Returns:
            WoeCalculator: The fitted calculator instance.
        """
        columns = columns or []
        for col in columns:
            # Ensure data is treated as strings for the mapping dictionary
            temp_series = df[col].astype(str).fillna("N/A")
            temp_df = pd.DataFrame({col: temp_series, target: df[target]})

            self.mapping_[col] = self._compute_mapping(temp_df, target, col)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Apply the calculated WoE mappings to the data.

        Args:
            df: Input DataFrame.
            columns: List of columns to transform.

        Returns:
            pd.DataFrame: DataFrame with additional _woe columns.
        """
        if not self._is_fitted:
            raise RuntimeError("WoeCalculator must be fitted before transforming.")

        columns = columns or []
        df_out = df.copy()
        for col in columns:
            if col in self.mapping_:
                temp_series = df_out[col].astype(str).fillna("N/A")
                new_col_name = f"{col}_woe"

                # Map directly with the dict; fillna(0.0) ensures neutral weight for unseen categories
                df_out[new_col_name] = temp_series.map(self.mapping_[col]).fillna(0.0)

        return df_out
