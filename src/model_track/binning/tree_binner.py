import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from model_track.base import BaseTransformer


class TreeBinner(BaseTransformer):
    """
    Binarizador supervisionado que utiliza árvores de decisão para
    encontrar pontos de corte ótimos.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 100):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bins: list[float] | None = None
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, column: str = "", target: str = "") -> "TreeBinner":  # type: ignore[override]
        """Aprende os pontos de corte da árvore."""
        # Limpeza para o treino (a árvore não aceita NaNs nativamente)
        df_clean = df[[column, target]].dropna()

        x = df_clean[[column]]
        y = df_clean[target]

        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            ccp_alpha=0.0,
        )
        tree.fit(x, y)

        # Extrair thresholds e ordenar
        thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]
        self.bins = sorted(thresholds.tolist())
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, column: str = "") -> pd.Series:
        """Aplica os bins aprendidos e trata valores nulos."""
        if not self._is_fitted:
            raise RuntimeError("O binner deve ser 'fitado' antes de transformar.")

        # Adicionamos -inf e inf para cobrir todo o range (com fallback para lista vazia)
        split_points = [-float("inf")] + (self.bins or []) + [float("inf")]

        # Gerar os labels/bins
        binned = pd.cut(df[column], bins=split_points, duplicates="drop")

        # Converter para string e tratar nulos como 'N/A' conforme seu notebook
        return binned.astype(str).replace("nan", "N/A")
