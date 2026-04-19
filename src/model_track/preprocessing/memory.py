import numpy as np
import pandas as pd


class DataOptimizer:
    """Otimizador de memória para DataFrames de larga escala."""

    @staticmethod
    def _downcast_numeric(series: pd.Series) -> pd.Series:
        """Aplica downcast no tipo da Series se possível para economizar memória."""
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
        Reduz tipos numéricos para economizar RAM e reporta o ganho.
        """
        # Criamos uma cópia para garantir a imutabilidade do original
        df = df.copy()

        start_mem = df.memory_usage().sum() / 1024**2

        for col in df.columns:
            df[col] = DataOptimizer._downcast_numeric(df[col])

        end_mem = df.memory_usage().sum() / 1024**2

        if verbose:
            diff = start_mem - end_mem
            pct = (diff / start_mem) * 100 if start_mem > 0 else 0
            print(f"📉 Memória Inicial: {start_mem:.2f} MB")
            print(f"✅ Memória Final:   {end_mem:.2f} MB")
            print(f"🚀 Redução de:      {diff:.2f} MB ({pct:.1f}%)")

        return df
