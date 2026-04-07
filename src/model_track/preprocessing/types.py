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

    def detect(self, df: pd.DataFrame) -> dict[str, list[str]]:
        feature_types: dict[str, list[str]] = {
            "datetime": [],
            "categorical_low": [],
            "categorical_high": [],
            "numerical": [],
            "id_like": [],
        }

        for col in df.columns:
            # Ignora target e IDs
            if col == self.target or col in self.id_cols:
                continue

            dtype = df[col].dtype
            n_unique = df[col].nunique(dropna=False)

            # 1. DATETIME (Explicito pelo usuário OU tipo nativo do Pandas)
            if col in self.datetime_cols or pd.api.types.is_datetime64_any_dtype(dtype):
                feature_types["datetime"].append(col)
                continue

            # 2. CATEGÓRICOS
            if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
                if n_unique <= self.threshold:
                    feature_types["categorical_low"].append(col)
                else:
                    feature_types["categorical_high"].append(col)
                continue

            # 3. NUMÉRICOS
            if pd.api.types.is_numeric_dtype(dtype):
                # Alta cardinalidade em inteiros pode ser um ID mascarado
                if pd.api.types.is_integer_dtype(dtype) and n_unique > (len(df) * 0.05):
                    feature_types["id_like"].append(col)
                # Baixa cardinalidade (ex: 0 e 1, flags)
                elif n_unique <= self.threshold:
                    feature_types["categorical_low"].append(col)
                # Numéricos contínuos reais
                else:
                    feature_types["numerical"].append(col)

        return feature_types
