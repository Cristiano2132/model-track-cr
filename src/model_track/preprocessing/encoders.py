import pandas as pd

from model_track.base import BaseTransformer


class OrdinalEncoder(BaseTransformer):
    """
    Ordinal Encoder optimized for trees (LightGBM/XGBoost).
    Unseen categories at fit time receive a wildcard value (-1).
    """

    def __init__(self, unseen_value: int = -1):
        self.unseen_value = unseen_value
        self.mapping_: dict[str, dict[str, int]] = {}
        self._is_fitted = False

    def fit(
        self, df: pd.DataFrame, target: str | None = None, columns: list[str] | None = None
    ) -> "OrdinalEncoder":
        """
        Fit the encoder to the specified columns.

        Args:
            df: Input DataFrame.
            target: Ignored (kept for API consistency).
            columns: List of columns to encode.

        Returns:
            OrdinalEncoder: The fitted encoder instance.
        """
        if not columns:
            raise ValueError("A list of columns must be provided.")

        for col in columns:
            # Force conversion to string and treat nulls as a valid category
            series = df[col].astype(str).fillna("N/A")
            unique_values = series.unique()
            self.mapping_[col] = {val: i for i, val in enumerate(unique_values)}

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Apply the ordinal encoding to the data.

        Args:
            df: Input DataFrame.
            columns: List of columns to transform.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if not self._is_fitted:
            raise RuntimeError("The encoder must be fitted before transforming.")
        if not columns:
            raise ValueError("A list of columns must be provided.")

        df_out = df.copy()
        for col in columns:
            if col in self.mapping_:
                series = df_out[col].astype(str).fillna("N/A")
                # Map and fill what didn't exist in training with unseen_value
                df_out[col] = series.map(self.mapping_[col]).fillna(self.unseen_value).astype(int)

        return df_out
