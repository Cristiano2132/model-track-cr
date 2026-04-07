import numpy as np
import pandas as pd

from model_track.preprocessing.auditor import DataAuditor


def test_data_auditor_compare_with_delta():
    """Valida comparação de schema e valores com tolerância (delta)."""
    df_a = pd.DataFrame({"col_idx": [1, 2], "val": [1.000001, 2.0], "excl_a": [0, 0]})
    df_b = pd.DataFrame({"col_idx": [1, 2], "val": [1.000002, 2.0], "excl_b": [1, 1]})

    auditor = DataAuditor()
    # Teste com delta que aceita a diferença
    res = auditor.compare_schemas(df_a, df_b, tolerance=1e-5)
    assert len(res["diff_value_cols"]) == 0

    # Teste com delta rígido que aponta a diferença
    res_rigid = auditor.compare_schemas(df_a, df_b, tolerance=1e-7)
    assert "val" in res_rigid["diff_value_cols"]
    assert "excl_a" in res_rigid["only_in_a"]
    assert "excl_b" in res_rigid["only_in_b"]


def test_data_auditor_summary_extended():
    """Valida o sumário com min, max e lista de exemplos de forma robusta."""
    # Criamos um DataFrame onde ambas as colunas têm o mesmo comprimento (11)
    df = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # 11 únicos (> 10)
            "cats": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", None],  # 5 únicos + NaN
        }
    )

    auditor = DataAuditor()
    summary = auditor.get_summary(df)

    # 1. Verificação de Alta Cardinalidade
    assert summary.loc["numeric", "max"] == 11
    assert summary.loc["numeric", "unique_examples"] == "Too many values..."

    # 2. Verificação de Baixa Cardinalidade (Exemplos)
    examples_str = summary.loc["cats", "unique_examples"]
    # O sorted na implementação garantirá a ordem: "A, B, C, D, E"
    assert "A" in examples_str
    assert "E" in examples_str
    assert "None" not in examples_str  # dropna=True

    # 3. Verificação de Top Class
    assert str(summary.loc["cats", "top_class"]) in ["A", "B", "C", "D", "E"]
    assert summary.loc["cats", "top_class_pct"] > 0


def test_data_auditor_with_categorical_columns():
    """Garante que o auditor não quebre com colunas do tipo category."""
    df = pd.DataFrame(
        {"cat_col": pd.Series(["A", "B", "A"]).astype("category"), "num_col": [1, 2, 3]}
    )

    auditor = DataAuditor()
    # Isso deve rodar sem disparar o TypeError
    summary = auditor.get_summary(df)

    assert summary.loc["cat_col", "dtype"] == "category"
    assert np.isnan(summary.loc["cat_col", "min"])


def test_auditor_compare_nan_handling():
    """Valida o comparador de schemas lidando com NaNs (coluna corrompida)."""
    auditor = DataAuditor()
    df_a = pd.DataFrame({"val": [np.nan, np.nan]})
    df_b = pd.DataFrame({"val": [1.0, 2.0]})

    res = auditor.compare_schemas(df_a, df_b)
    assert "val" in res["diff_value_cols"]
