import numpy as np
import pandas as pd

from model_track.base import BaseTransformer


class WoeCalculator(BaseTransformer):
    """Calcula e aplica o Weight of Evidence (WoE) para variáveis categóricas."""

    def __init__(self) -> None:
        self.mapping_: dict[str, dict[str, float]] = {}
        self._is_fitted = False

    def _compute_mapping(self, df: pd.DataFrame, target: str, feature: str) -> dict[str, float]:
        """Calcula o dicionário WoE com Laplace Smoothing."""
        stats = df.groupby(feature, observed=True)[target].agg(["count", "sum"])
        stats.columns = ["Total", "Bad"]
        stats["Good"] = stats["Total"] - stats["Bad"]

        perc_bad = (stats["Bad"] + 0.5) / (stats["Bad"].sum() + 0.5)
        perc_good = (stats["Good"] + 0.5) / (stats["Good"].sum() + 0.5)

        return np.log(perc_good / perc_bad).to_dict()  # type: ignore[no-any-return]

    def fit(self, df: pd.DataFrame, target: str, columns: list[str]) -> "WoeCalculator":  # type: ignore[override]
        for col in columns:
            # Garante que os dados sejam tratados como string para o dicionário
            temp_series = df[col].astype(str).fillna("N/A")
            temp_df = pd.DataFrame({col: temp_series, target: df[target]})

            self.mapping_[col] = self._compute_mapping(temp_df, target, col)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:  # type: ignore[override]
        if not self._is_fitted:
            raise RuntimeError("WoeCalculator precisa ser fitado antes do transform.")

        df_out = df.copy()
        for col in columns:
            if col in self.mapping_:
                temp_series = df_out[col].astype(str).fillna("N/A")
                new_col_name = f"{col}_woe"

                # Mapeia com o dict direto e o fillna(0.0) garante o peso neutro para categorias não vistas
                df_out[new_col_name] = temp_series.map(self.mapping_[col]).fillna(0.0)

        return df_out
