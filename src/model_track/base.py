from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    """
    Interface base para todos os transformadores do model-track.
    Define o contrato fit e transform para garantir modularidade.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: str | None = None) -> "BaseTransformer":
        """Treina o transformador."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a transformação."""
        pass

    def fit_transform(self, df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """Atalho para executar o treinamento e a aplicação em sequência."""
        return self.fit(df, target).transform(df)
