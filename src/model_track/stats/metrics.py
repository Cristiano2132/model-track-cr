import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_iv(df: pd.DataFrame, feature: str, target: str) -> float:
    """Calcula o Information Value usando Laplace Smoothing para evitar Log(0)."""
    counts = pd.crosstab(df[feature], df[target])

    # Se só houver uma classe do target (ex: só tem fraude na base toda)
    if counts.shape[1] < 2:
        return 0.0

    # Assume que a coluna 0 é "Good" e 1 é "Bad" (padrão de Fraude/Crédito)
    dist_good = (counts.iloc[:, 0] + 0.5) / (counts.iloc[:, 0].sum() + 0.5)
    dist_bad = (counts.iloc[:, 1] + 0.5) / (counts.iloc[:, 1].sum() + 0.5)

    woe = np.log(dist_good / dist_bad)
    iv = (dist_good - dist_bad) * woe

    return float(iv.sum())


def compute_cramers_v(df: pd.DataFrame, f1: str, f2: str) -> float:
    """Calcula a correlação categórica de Cramer's V com correção de viés."""
    obs = pd.crosstab(df[f1], df[f2]).values
    if obs.size == 0 or obs.sum() == 0:
        return 0.0

    try:
        chi2 = chi2_contingency(obs)[0]
        n = obs.sum()
        phi2 = chi2 / n
        r, k = obs.shape

        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)

        denom = min((kcorr - 1), (rcorr - 1))
        return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0
    except Exception:
        return 0.0
