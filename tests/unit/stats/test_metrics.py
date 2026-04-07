import numpy as np
import pandas as pd

from model_track.stats.metrics import compute_cramers_v, compute_iv


def test_compute_iv():
    """Valida o cálculo do Information Value, incluindo tratamento de zeros."""
    df = pd.DataFrame(
        {
            "perfeito": [0, 0, 1, 1, 0, 0],  # Separa perfeitamente o target
            "aleatorio": [0, 1, 0, 1, 0, 1],  # Não tem relação
            "target": [0, 0, 1, 1, 0, 0],
        }
    )

    iv_perf = compute_iv(df, "perfeito", "target")
    iv_rand = compute_iv(df, "aleatorio", "target")

    assert iv_perf > 0.5  # Feature forte
    assert iv_rand < 0.1  # Feature fraca


def test_compute_cramers_v():
    """Valida se features idênticas retornam ~1.0 e independentes ~0.0."""
    df = pd.DataFrame(
        {
            "f1": [1, 1, 2, 2, 3, 3],
            "f2": [1, 1, 2, 2, 3, 3],  # Cópia de f1
            "f3": [1, 2, 1, 2, 1, 2],  # Independente
        }
    )

    v_identico = compute_cramers_v(df, "f1", "f2")
    v_indep = compute_cramers_v(df, "f1", "f3")

    assert np.isclose(v_identico, 1.0, atol=0.05)
    assert v_indep < 0.5


def test_compute_iv_with_zero_counts():
    """
    Garante que categorias com 0 bons ou 0 maus não gerem IV infinito (Laplace Smoothing).
    """
    df = pd.DataFrame(
        {
            "feature_rara": ["A", "A", "B", "B", "C", "C"],
            "target": [0, 0, 1, 1, 0, 0],
            # A categoria 'A' tem 0 maus (target=1)
            # A categoria 'B' tem 0 bons (target=0)
        }
    )

    iv = compute_iv(df, "feature_rara", "target")

    # O IV deve ser um número real válido, não infinito nem NaN
    assert not np.isinf(iv)
    assert not np.isnan(iv)
    assert iv > 0.0  # Como é perfeitamente separada, o IV será alto, mas finito.


def test_compute_iv_single_class():
    """Garante que IV retorne 0.0 se o target tiver apenas 1 classe."""
    df = pd.DataFrame({"f": [1, 1, 2], "target": [1, 1, 1]})
    assert compute_iv(df, "f", "target") == 0.0


def test_compute_cramers_v_edge_cases():
    """Cobre retornos de correlação vazia ou colunas constantes."""
    # DataFrame vazio
    df_empty = pd.DataFrame({"f1": [], "f2": []})
    assert compute_cramers_v(df_empty, "f1", "f2") == 0.0

    # DataFrame constante (divisão por zero segura no Cramer's V)
    df_const = pd.DataFrame({"f1": [1, 1, 1], "f2": [2, 2, 2]})
    assert compute_cramers_v(df_const, "f1", "f2") == 0.0
