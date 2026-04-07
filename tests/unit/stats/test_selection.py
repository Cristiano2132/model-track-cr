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
