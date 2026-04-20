import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_iv(df: pd.DataFrame, feature: str, target: str) -> float:
    """
    Calculate the Information Value (IV) using Laplace Smoothing to avoid log(0).

    Args:
        df: Input DataFrame.
        feature: Feature column name.
        target: Target column name (expected to be binary).

    Returns:
        float: Calculated Information Value.
    """
    counts = pd.crosstab(df[feature], df[target])

    # If there's only one target class (e.g., only fraud in the entire dataset)
    if counts.shape[1] < 2:
        return 0.0

    # Assume column 0 is "Good" and 1 is "Bad" (Fraud/Credit standard)
    dist_good = (counts.iloc[:, 0] + 0.5) / (counts.iloc[:, 0].sum() + 0.5)
    dist_bad = (counts.iloc[:, 1] + 0.5) / (counts.iloc[:, 1].sum() + 0.5)

    woe = np.log(dist_good / dist_bad)
    iv = (dist_good - dist_bad) * woe

    return float(iv.sum())


def compute_cramers_v(df: pd.DataFrame, f1: str, f2: str) -> float:
    """
    Calculate Cramer's V categorical correlation with bias correction.

    Args:
        df: Input DataFrame.
        f1: First feature column name.
        f2: Second feature column name.

    Returns:
        float: Calculated Cramer's V correlation.
    """
    obs = pd.crosstab(df[f1], df[f2]).to_numpy()
    if obs.size == 0 or obs.sum() == 0:
        return 0.0

    chi2 = chi2_contingency(obs)[0]
    n = obs.sum()
    phi2 = chi2 / n
    r, k = obs.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0
