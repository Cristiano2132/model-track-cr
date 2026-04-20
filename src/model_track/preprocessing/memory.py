import numpy as np
import pandas as pd


class DataOptimizer:
    """Memory optimizer for large-scale DataFrames."""

    @staticmethod
    def _downcast_numeric(series: pd.Series) -> pd.Series:
        """
        Downcast the Series type if possible to save memory.

        Args:
            series: Input Series.

        Returns:
            pd.Series: Optimized Series.
        """
        col_type = series.dtype

        if col_type == "object" or isinstance(col_type, pd.CategoricalDtype):
            return series.astype("category")

        c_min = series.min()
        c_max = series.max()

        if str(col_type).startswith("int"):
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                return series.astype(np.int8)
            if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                return series.astype(np.int16)
            if c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                return series.astype(np.int32)
            return series.astype(np.int64)

        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
            return series.astype(np.float32)
        return series.astype(np.float64)

    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """
        Reduce numeric types to save RAM and report the gain.

        Args:
            df: Input DataFrame.
            verbose: If True, prints optimization report.

        Returns:
            pd.DataFrame: Optimized DataFrame.
        """
        # Create a copy to ensure original immutability
        df = df.copy()

        start_mem = df.memory_usage().sum() / 1024**2

        for col in df.columns:
            df[col] = DataOptimizer._downcast_numeric(df[col])

        end_mem = df.memory_usage().sum() / 1024**2

        if verbose:
            diff = start_mem - end_mem
            pct = (diff / start_mem) * 100 if start_mem > 0 else 0
            print(f"📉 Initial Memory: {start_mem:.2f} MB")
            print(f"✅ Final Memory:   {end_mem:.2f} MB")
            print(f"🚀 Reduction:      {diff:.2f} MB ({pct:.1f}%)")

        return df
