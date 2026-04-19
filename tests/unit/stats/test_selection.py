from unittest.mock import patch

import pandas as pd

from model_track.stats.selection import StatisticalSelector


def test_statistical_selector_fit_transform():
    """Garante que a classe descarta IVs baixos e features correlacionadas."""

    # Criamos 200 linhas (* 50) para garantir que a estatística funcione sem
    # sofrer com o viés de amostras muito pequenas (Bias Correction).
    df = pd.DataFrame(
        {
            "target": [0, 0, 1, 1] * 50,
            "boa": [0, 0, 1, 1] * 50,  # Replica o target (IV altíssimo)
            "copia": [0, 0, 1, 1] * 50,  # Idêntica à 'boa' (Cramer's V = 1.0)
            "ruim": [0, 1, 0, 1] * 50,  # Perfeitamente 50/50 independente (IV = 0.0)
        }
    )

    selector = StatisticalSelector(iv_threshold=0.10, cramers_threshold=0.80)

    # Fit & Transform
    selector.fit(df, target="target", features=["boa", "copia", "ruim"])
    df_transformed = selector.transform(df)

    # Assertions
    assert "boa" in df_transformed.columns
    assert "target" in df_transformed.columns
    assert "ruim" not in df_transformed.columns  # Caiu no filtro de IV baixo
    assert "copia" not in df_transformed.columns  # Caiu no filtro de Cramer's V

    # Verifica integridade da auditoria salva na classe
    assert selector.iv_results_["boa"] > 0.10
    assert "ruim" in selector.dropped_features_


def test_selector_sample_size_and_fallback():
    """Força o IF de amostragem quando o dataset é maior que o sample_size."""
    df = pd.DataFrame(
        {"f1": [1, 2, 1, 2, 1, 2], "f2": [1, 1, 1, 2, 2, 2], "target": [0, 1, 0, 1, 0, 1]}
    )
    # sample_size=2 força a entrar no IF `if len(df) > self.sample_size`
    selector = StatisticalSelector(sample_size=2, iv_threshold=0.0)
    selector.fit(df, target="target", features=["f1", "f2"])

    assert hasattr(selector, "selected_features_")


def test_statistical_selector_full_flow():
    """Cobre o fit (com amostragem), transform (Linha 58) e filtros."""
    # Criamos um DF maior para disparar a amostragem do fit
    size = 100
    df = pd.DataFrame(
        {
            "target": [0, 1] * (size // 2),
            "feature_ok": [0, 1] * (size // 2),
            "feature_ruim": [0, 0] * (size // 2),
        }
    )

    # iv_threshold baixo para aceitar a feature_ok
    # sample_size pequeno para forçar o código de amostragem
    selector = StatisticalSelector(iv_threshold=0.01, sample_size=50)

    selector.fit(df, target="target", features=["feature_ok", "feature_ruim"])

    # Executa transform para cobrir linha 58
    df_transformed = selector.transform(df)

    assert "feature_ok" in df_transformed.columns
    assert "target" in df_transformed.columns
    assert "feature_ruim" not in df_transformed.columns


def test_statistical_selector_transitive_drop():
    """Cobre a linha onde f2 já está em to_drop_corr (f2 já descartado por um f1 anterior)."""
    df = pd.DataFrame(
        {
            "target": [0, 1] * 20,
            "A": [0, 1] * 20,
            "B": [0, 1] * 20,
            "C": [0, 1] * 20,
        }
    )

    selector = StatisticalSelector(iv_threshold=0.01, cramers_threshold=0.80)

    # Vamos forçar: IV de A = 0.5, B = 0.4, C = 0.3
    # E Cramer's V(A, C) = 0.9, os demais 0.1.
    def mock_iv(df, col, target):
        return {"A": 0.5, "B": 0.4, "C": 0.3}[col]

    def mock_cramers(df, f1, f2):
        if {f1, f2} == {"A", "C"}:
            return 0.9
        return 0.1

    with patch("model_track.stats.selection.compute_iv", side_effect=mock_iv):
        with patch("model_track.stats.selection.compute_cramers_v", side_effect=mock_cramers):
            selector.fit(df, target="target", features=["A", "B", "C"])

    assert "C" in selector.dropped_features_
