from typing import Any

import pandas as pd


class TypeDetector:
    """Detecta a natureza estatística e funcional das colunas de forma declarativa."""

    def __init__(
        self,
        target: str | None = None,
        id_cols: list[str] | None = None,
        datetime_cols: list[str] | None = None,  # <--- Parâmetro adicionado
        low_card_threshold: int = 15,
    ):
        self.target = target
        self.id_cols = id_cols or []
        self.datetime_cols = datetime_cols or []
        self.threshold = low_card_threshold

    def _classify_column(self, col: str, dtype: Any, n_unique: int, total_rows: int) -> str | None:
        """Classifica uma coluna de acordo com sua tipagem e cardinalidade."""
        if col == self.target or col in self.id_cols:
            return None

        # 1. DATETIME (Explicito pelo usuário OU tipo nativo do Pandas)
        if col in self.datetime_cols or pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        # 2. CATEGÓRICOS
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            if n_unique <= self.threshold:
                return "categorical_low"
            return "categorical_high"

        # 3. NUMÉRICOS
        if pd.api.types.is_numeric_dtype(dtype):
            # Alta cardinalidade em inteiros pode ser um ID mascarado
            if pd.api.types.is_integer_dtype(dtype) and n_unique > (total_rows * 0.05):
                return "id_like"
            # Baixa cardinalidade (ex: 0 e 1, flags)
            if n_unique <= self.threshold:
                return "categorical_low"
            # Numéricos contínuos reais
            return "numerical"

        return None

    def detect(self, df: pd.DataFrame) -> dict[str, list[str]]:
        feature_types: dict[str, list[str]] = {
            "datetime": [],
            "categorical_low": [],
            "categorical_high": [],
            "numerical": [],
            "id_like": [],
        }

        total_rows = len(df)
        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique(dropna=False)

            col_type = self._classify_column(col, dtype, n_unique, total_rows)
            if col_type:
                feature_types[col_type].append(col)

        return feature_types
