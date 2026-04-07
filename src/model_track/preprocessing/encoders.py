import pandas as pd

from model_track.base import BaseTransformer


class OrdinalEncoder(BaseTransformer):
    """
    Codificador Ordinal otimizado para árvores (LightGBM/XGBoost).
    Categorias não vistas no momento do fit recebem um valor coringa (-1).
    """

    def __init__(self, unseen_value: int = -1):
        self.unseen_value = unseen_value
        self.mapping_: dict[str, dict[str, int]] = {}
        self._is_fitted = False

    def fit(
        self, df: pd.DataFrame, target: str | None = None, columns: list[str] | None = None
    ) -> "OrdinalEncoder":
        if not columns:
            raise ValueError("Uma lista de colunas deve ser fornecida.")

        for col in columns:
            # Força conversão para string e trata os nulos como uma categoria válida
            series = df[col].astype(str).fillna("N/A")
            unique_values = series.unique()
            self.mapping_[col] = {val: i for i, val in enumerate(unique_values)}

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("O encoder precisa ser fitado antes de transformar.")
        if not columns:
            raise ValueError("Uma lista de colunas deve ser fornecida.")

        df_out = df.copy()
        for col in columns:
            if col in self.mapping_:
                series = df_out[col].astype(str).fillna("N/A")
                # Mapeia e preenche o que não existia no treino com -1
                df_out[col] = series.map(self.mapping_[col]).fillna(self.unseen_value).astype(int)

        return df_out
