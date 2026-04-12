import numpy as np
import pandas as pd

from model_track.stats.metrics import compute_cramers_v, compute_iv


def test_compute_iv_logic():
    """Valida Information Value com massa de dados suficiente."""
    # Aumentamos o tamanho para 20 linhas para estabilizar o cálculo
    df = pd.DataFrame(
        {"feat": ["A"] * 10 + ["B"] * 10, "target": [0] * 8 + [1] * 2 + [1] * 8 + [0] * 2}
    )
    iv = compute_iv(df, "feat", "target")
    assert iv > 0.1
    assert not np.isinf(iv)


def test_compute_iv_single_class():
    """Cobre o early return se o target tiver apenas 1 classe."""
    df = pd.DataFrame({"f": [1, 2, 3], "target": [1, 1, 1]})
    assert compute_iv(df, "f", "target") == 0.0


def test_compute_cramers_v_logic():
    """Valida correlação perfeita e independente com amostra estável."""
    # 20 amostras garantem que o teste chi2 não falhe por graus de liberdade/vies
    size = 20
    df = pd.DataFrame(
        {
            "f1": [1, 2] * (size // 2),
            "f2": [1, 2] * (size // 2),  # Perfeita
            "f3": [1] * (size // 2) + [2] * (size // 2),  # Independente
        }
    )

    v_perfeito = compute_cramers_v(df, "f1", "f2")
    v_independente = compute_cramers_v(df, "f1", "f3")

    assert round(v_perfeito, 1) >= 0.9
    assert round(v_independente, 1) < 0.2


def test_compute_cramers_v_empty_and_zero():
    """Cobre o early return para tabelas vazias ou somas zeradas."""
    df = pd.DataFrame({"f1": [np.nan], "f2": [np.nan]})
    assert compute_cramers_v(df, "f1", "f2") == 0.0


def test_compute_cramers_v_exception_path():
    """Cobre as linhas 42-43 (bloco except)."""
    # Forçamos uma crosstab que quebra a lógica do chi2_contingency
    # passando uma estrutura de apenas uma célula ou dados constantes
    df = pd.DataFrame({"f1": ["A", "A"], "f2": [1, 1]})
    result = compute_cramers_v(df, "f1", "f2")
    assert result == 0.0


def test_compute_cramers_v_zero_denom():
    """Cobre o caso onde o denominador é zero (ex: tabela 1x1)."""
    df = pd.DataFrame({"f1": [np.nan] * 10, "f2": ["X"] * 10})
    result = compute_cramers_v(df, "f1", "f2")
    assert result == 0.0
