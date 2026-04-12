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


def test_compare_schemas_one_sided_nan():
    """Cobre a linha 43: quando apenas um lado é NaN."""
    auditor = DataAuditor()

    df_a = pd.DataFrame({"v1": [1, 2, 3]})
    df_b = pd.DataFrame({"v1": [np.nan, np.nan]})  # Média é NaN

    result = auditor.compare_schemas(df_a, df_b)
    assert "v1" in result["diff_value_cols"]


def test_compare_schemas_non_numeric_diff():
    """Cobre o 'else' final: colunas comuns mas com dtypes diferentes."""
    auditor = DataAuditor()

    df_a = pd.DataFrame({"feat": ["a", "b"]})
    df_b = pd.DataFrame({"feat": [1, 2]})

    result = auditor.compare_schemas(df_a, df_b)
    assert "feat" in result["diff_value_cols"]


def test_compare_schemas_full_coverage():
    """Cobre as linhas de comparação de médias e tratamento de NaNs (Linhas 42-45)."""
    auditor = DataAuditor()

    # Colunas com 3 linhas para todos
    df_a = pd.DataFrame(
        {
            "v1": [1.0, 2.0, 3.0],
            "v2": [np.nan, np.nan, np.nan],
            "v3": [10.0, 20.0, 30.0],
            "only_a": [1, 1, 1],
        }
    )

    df_b = pd.DataFrame(
        {
            "v1": [1.0, 2.0, 3.0],
            "v2": [np.nan, np.nan, np.nan],  # Ambas NaN -> cai no continue
            "v3": [100.0, 200.0, 300.0],  # Médias diferentes
            "only_b": [2, 2, 2],
        }
    )

    result = auditor.compare_schemas(df_a, df_b)

    assert "v3" in result["diff_value_cols"]
    assert "v2" not in result["diff_value_cols"]
    assert "only_a" in result["only_in_a"]
    assert "only_b" in result["only_in_b"]


def test_get_summary_fixed_assertions():
    """Cobre os caminhos do get_summary usando pd.isna() para validação de NaNs."""
    auditor = DataAuditor()

    size = 20
    df = pd.DataFrame(
        {
            "cat_low": ["A", "B", "C"] + ["A"] * (size - 3),
            "cat_high": [str(i) for i in range(size)],
            "num_val": [float(i) for i in range(size - 1)] + [np.nan],
            "bool_val": [True, False] * (size // 2),
            "empty_col": [np.nan] * size,
        }
    )

    summary = auditor.get_summary(df)

    # Assertions Corretas
    assert summary.loc["cat_low", "unique_examples"] == "A, B, C"
    assert summary.loc["cat_high", "unique_examples"] == "Too many values..."

    # Verificação de valores numéricos
    assert not pd.isna(summary.loc["num_val", "min"])

    # Verificação de NaNs (Onde o 'is np.nan' falhou)
    assert pd.isna(summary.loc["cat_low", "min"])
    assert pd.isna(summary.loc["empty_col", "top_class"])
    assert pd.isna(summary.loc["bool_val", "min"])  # Validando que bool não é considerado numérico
