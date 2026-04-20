from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    """
    Base interface for all model-track transformers.
    Defines the fit and transform contract to ensure modularity.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: str | None = None) -> "BaseTransformer":
        """
        Train the transformer.

        Args:
            df: Input DataFrame to train on.
            target: Target variable name for supervised transformers.

        Returns:
            BaseTransformer: The fitted transformer instance.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the transformation to the data.

        Args:
            df: Input DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        pass

    def fit_transform(self, df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """
        Helper method to perform both fit and transform in sequence.

        Args:
            df: Input DataFrame.
            target: Target variable name.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        return self.fit(df, target).transform(df)
