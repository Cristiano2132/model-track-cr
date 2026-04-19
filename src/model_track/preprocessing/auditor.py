from typing import Any

import numpy as np
import pandas as pd


class DataAuditor:
    """Auditoria profunda de schemas e integridade estatística entre tabelas."""

    def __init__(self, target: str | None = None):
        self.target = target

    def compare_schemas(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame, tolerance: float = 1e-6
    ) -> dict[str, Any]:
        """
        Compara schemas, colunas exclusivas e integridade de valores em colunas comuns.
        """
        cols_a = set(df_a.columns)
        cols_b = set(df_b.columns)

        common_cols = [c for c in (cols_a & cols_b) if c != self.target]
        diff_value_cols = []

        for col in common_cols:
            is_num_a = pd.api.types.is_numeric_dtype(df_a[col])
            is_num_b = pd.api.types.is_numeric_dtype(df_b[col])

            if is_num_a and is_num_b:
                val_a = df_a[col].mean()
                val_b = df_b[col].mean()

                # Tratamento de NaNs para evitar furos na auditoria (Linhas 42-45)
                if pd.isna(val_a) and pd.isna(val_b):
                    continue
                elif pd.isna(val_a) or pd.isna(val_b):
                    diff_value_cols.append(col)
                elif abs(val_a - val_b) > tolerance:
                    diff_value_cols.append(col)
            else:
                if str(df_a[col].dtype) != str(df_b[col].dtype):
                    diff_value_cols.append(col)

        return {
            "only_in_a": list(cols_a - cols_b),
            "only_in_b": list(cols_b - cols_a),
            "diff_value_cols": diff_value_cols,
        }

    def _get_column_stats(self, col: str, series: pd.Series) -> dict[str, Any]:
        """Calcula estatísticas para uma única coluna."""
        n_unique = int(series.nunique())

        # Exemplos determinísticos
        if n_unique <= 10:
            unique_vals = sorted([str(x) for x in series.dropna().unique()])
            examples = ", ".join(unique_vals)
        else:
            examples = "Too many values..."

        # Estatísticas de Top Class
        mode_result = series.mode()
        top_val = mode_result.iloc[0] if not mode_result.empty else np.nan

        total_rows = len(series)
        top_pct = (
            (series.value_counts().max() / total_rows) * 100
            if total_rows > 0 and n_unique > 0
            else 0.0
        )

        # CORREÇÃO: Usar pd.api.types para evitar erro com CategoricalDtype
        is_numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
            series.dtype
        )

        stats = {
            "column_name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isnull().sum()),
            "pct_na": float((series.isnull().sum() / total_rows) * 100) if total_rows > 0 else 0.0,
            "n_distinct": n_unique,
            "min": series.min() if is_numeric else np.nan,
            "max": series.max() if is_numeric else np.nan,
            "top_class": top_val,
            "top_class_pct": float(top_pct),
            "unique_examples": examples,
        }
        return stats

    def get_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera sumário estatístico exaustivo das colunas."""
        summary_data = [self._get_column_stats(col, df[col]) for col in df.columns]
        return pd.DataFrame(summary_data).set_index("column_name")
