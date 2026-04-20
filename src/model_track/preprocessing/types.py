from typing import Any

import pandas as pd


class TypeDetector:
    """Detects the statistical and functional nature of columns in a declarative way."""

    def __init__(
        self,
        target: str | None = None,
        id_cols: list[str] | None = None,
        datetime_cols: list[str] | None = None,  # User-specified datetime columns
        low_card_threshold: int = 15,
    ):
        self.target = target
        self.id_cols = id_cols or []
        self.datetime_cols = datetime_cols or []
        self.threshold = low_card_threshold

    def _classify_column(self, col: str, dtype: Any, n_unique: int, total_rows: int) -> str | None:
        """
        Classify a column according to its type and cardinality.

        Args:
            col: Column name.
            dtype: Column data type.
            n_unique: Number of unique values.
            total_rows: Total number of rows in the DataFrame.

        Returns:
            str | None: The detected category of the column.
        """
        if col == self.target or col in self.id_cols:
            return None

        # 1. DATETIME (Explicitly user-defined OR native Pandas datetime type)
        if col in self.datetime_cols or pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"

        # 2. CATEGORICAL
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            if n_unique <= self.threshold:
                return "categorical_low"
            return "categorical_high"

        # 3. NUMERICAL
        if pd.api.types.is_numeric_dtype(dtype):
            # High cardinality in integers might be a masked ID
            if pd.api.types.is_integer_dtype(dtype) and n_unique > (total_rows * 0.05):
                return "id_like"
            # Low cardinality (e.g., 0 and 1, flags)
            if n_unique <= self.threshold:
                return "categorical_low"
            # Real continuous numerical features
            return "numerical"

        return None

    def detect(self, df: pd.DataFrame) -> dict[str, list[str]]:
        """
        Detect and group columns by their statistical types.

        Args:
            df: Input DataFrame.

        Returns:
            dict[str, list[str]]: Dictionary mapping type names to lists of column names.
        """
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
