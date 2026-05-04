import pandas as pd

from model_track.context import ProjectContext


class BinApplier:
    """
    Applies saved binning cut points to new data to ensure consistency.
    Typically used to apply bins learned during training to test or production data.
    """

    def __init__(self, bins_map: dict[str, list[float]] | None = None):
        self.bins_map = bins_map or {}

    @classmethod
    def from_context(cls, context: ProjectContext) -> "BinApplier":
        """
        Create a BinApplier using the bins_map from a ProjectContext.

        Args:
            context: The ProjectContext containing the saved bins.

        Returns:
            BinApplier: An instance configured with the context's bins.
        """
        return cls(bins_map=context.bins_map)

    def apply(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Apply saved bins to the specified columns of the DataFrame.

        Args:
            df: Input DataFrame.
            columns: List of columns to transform. If None, applies to all columns in bins_map.

        Returns:
            pd.DataFrame: A new DataFrame with the binned columns.
        """
        df_transformed = df.copy()
        cols_to_apply = columns if columns is not None else list(self.bins_map.keys())

        for col in cols_to_apply:
            if col not in self.bins_map:
                raise ValueError(f"Column '{col}' not found in the bins_map.")

            df_transformed[col] = self.apply_column(df[col], self.bins_map[col])

        return df_transformed

    @staticmethod
    def apply_column(series: pd.Series, bins: list[float]) -> pd.Series:
        """
        Apply a specific set of bins to a pandas Series.

        Args:
            series: Input Series.
            bins: List of internal split points.

        Returns:
            pd.Series: Binned Series as strings, with nulls as 'N/A'.
        """
        # We add -inf and inf to cover the entire range
        split_points = [-float("inf")] + sorted(bins) + [float("inf")]

        # Apply bins
        binned = pd.cut(series, bins=split_points, duplicates="drop")

        # Convert to string and treat nulls as 'N/A'
        return binned.astype(str).replace("nan", "N/A")
