import pandas as pd

from model_track.base import BaseTransformer
from model_track.stats.metrics import compute_cramers_v, compute_iv


class StatisticalSelector(BaseTransformer):
    """
    Filtra features combinando Information Value (IV) e Correlação de Cramer's V.
    Retém a feature de maior IV quando há alta correlação.
    """

    def __init__(
        self,
        iv_threshold: float = 0.10,
        cramers_threshold: float = 0.85,
        sample_size: int | None = 50000,
    ):
        self.iv_threshold = iv_threshold
        self.cramers_threshold = cramers_threshold
        self.sample_size = sample_size
        self.iv_results_: dict[str, float] = {}
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []

    def fit(self, df: pd.DataFrame, target: str, features: list[str]) -> "StatisticalSelector":  # type: ignore[override]
        """Avalia as features e define quais sobreviverão."""
        df_sample = df

        # Amostragem estratificada para acelerar Cramer's V
        if self.sample_size and len(df) > self.sample_size:
            frac = self.sample_size / len(df)
            df_sample = df.groupby(target, group_keys=False, observed=True).apply(
                lambda x: x.sample(frac=frac, random_state=42)
            )

        valid_features = [f for f in features if f in df_sample.columns]

        # 1. Filtro de IV
        for col in valid_features:
            self.iv_results_[col] = compute_iv(df_sample, col, target)

        strong_features = [f for f, iv in self.iv_results_.items() if iv >= self.iv_threshold]

        # 2. Ordena as features fortes pelo IV (A mais forte 'vence' no Cramer's V)
        strong_features.sort(key=lambda x: self.iv_results_[x], reverse=True)

        # 3. Filtro de Cramer's V
        to_drop_corr = set()
        df_sample_cat = df_sample[strong_features]

        for i, f1 in enumerate(strong_features):
            if f1 in to_drop_corr:
                continue

            for f2 in strong_features[i + 1 :]:
                if f2 in to_drop_corr:
                    continue

                v = compute_cramers_v(df_sample_cat, f1, f2)
                if v > self.cramers_threshold:
                    to_drop_corr.add(f2)  # Descarta a feature de menor IV

        self.selected_features_ = [f for f in strong_features if f not in to_drop_corr]
        self.dropped_features_ = [f for f in valid_features if f not in self.selected_features_]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove as features descartadas do DataFrame."""
        return df.drop(
            columns=[f for f in self.dropped_features_ if f in df.columns], errors="ignore"
        )
