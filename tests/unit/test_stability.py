import pandas as pd
import pytest
from matplotlib import pyplot as plt

from model_track.woe.stability import CategoryMapper, WoeStability


def test_woe_stability_matrix():
    """Valida se a matriz de estabilidade gera safras no índice e categorias nas colunas."""
    df = pd.DataFrame(
        {
            "safra": ["2023-01", "2023-01", "2023-02", "2023-02"],
            "cat": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )

    ws = WoeStability(date_col="safra")
    matrix = ws.calculate_stability_matrix(df, feature_col="cat", target_col="target")

    assert list(matrix.index) == ["2023-01", "2023-02"]
    assert "A" in matrix.columns
    assert "B" in matrix.columns


def test_category_mapper_exhaustive_search():
    """Garante que o algoritmo minimize cruzamentos resolvendo ambiguidades severas."""
    matrix = pd.DataFrame(
        {
            "C1": [-1.0, -1.0, -1.0],
            "C2": [0.0, 1.5, 0.0],
            "C3": [2.0, -0.5, 2.0],
            "C4": [3.0, 3.0, 3.0],
        },
        index=["safra1", "safra2", "safra3"],
    )

    mapper = CategoryMapper()
    suggested_map = mapper.auto_group(matrix, min_groups=2)

    assert suggested_map["C2"] == suggested_map["C3"]
    assert suggested_map["C1"] != suggested_map["C2"]


def test_category_mapper_smart_intervals():
    """Garante nomenclatura intervalar correta forçando estabilidade interna."""
    matrix = pd.DataFrame(
        {
            "2": [1.0, 1.2],
            "1": [1.2, 1.0],  # 1 e 2 se cruzam entre si
            "3": [5.0, 5.2],
            "4": [5.2, 5.0],  # 3 e 4 se cruzam entre si
            "8": [9.0, 9.2],
            "9": [9.1, 9.1],
            "10": [9.2, 9.0],  # 8, 9, 10 se misturam
        },
        index=["safra1", "safra2"],
    )

    mapper = CategoryMapper()
    suggested_map = mapper.auto_group(matrix, min_groups=3, is_ordered=True)

    assert suggested_map["1"] == "<=2"
    assert suggested_map["4"] == "3 a 4"
    assert suggested_map["10"] == ">=8"


def test_category_mapper_ordinal_constraint():
    """Garante que is_ordered=True impede agrupamento de riscos similares se não forem adjacentes (Risco em U)."""
    # 10 e 30 têm o mesmo risco exato (2.0), mas 20 tem risco diferente (-1.0).
    matrix = pd.DataFrame(
        {"10": [2.0, 2.0], "20": [-1.0, -1.0], "30": [2.0, 2.0]}, index=["s1", "s2"]
    )

    mapper = CategoryMapper()

    # 1. Sem restrição ordinal (Default): Vai juntar '10' com '30' porque o risco é idêntico
    map_unordered = mapper.auto_group(matrix, min_groups=2, is_ordered=False)
    assert map_unordered["10"] == map_unordered["30"]

    # 2. Com restrição ordinal: É PROIBIDO juntar '10' com '30'.
    # Ele será forçado a separar os 3, ou juntar vizinhos (ex: <=20 ou >=20)
    map_ordered = mapper.auto_group(matrix, min_groups=2, is_ordered=True)
    assert map_ordered["10"] != map_ordered["30"]


@pytest.fixture
def stability_data():
    """Gera um DataFrame íntegro (mesmo tamanho de listas)."""
    return pd.DataFrame(
        {
            "safra": ["2023-01"] * 10 + ["2023-02"] * 10,
            "target": [0, 1] * 10,
            "feat": ["10", "20", "30", "40", "50"] * 4,
        }
    )


@pytest.fixture
def mock_matrix():
    """Gera uma matriz de WoE para o teste do generate_view."""
    return pd.DataFrame({"A": [0.1, 0.2], "B": [0.5, 0.4]}, index=["2023-01", "2023-02"])


def test_generate_view_correct_return(mock_matrix):
    """
    Cobre as linhas 40-49:
    Garante que o retorno é o AX e valida o título.
    """
    ws = WoeStability(date_col="safra")

    # Cenário 1: ax=None (Linhas 40-43) - Cria novo ax internamente
    ax_new = ws.generate_view(mock_matrix, title="Novo Titulo")
    assert ax_new.get_title() == "Novo Titulo"
    plt.close(ax_new.get_figure())  # Fecha a figura via referência do ax

    # Cenário 2: ax fornecido (Linhas 45-49) - Usa o ax externo
    fig, ax_ext = plt.subplots()
    ax_returned = ws.generate_view(mock_matrix, ax=ax_ext, title="Titulo Externo")

    assert ax_returned is ax_ext
    assert ax_ext.get_title() == "Titulo Externo"
    plt.close(fig)


def test_calculate_stability_matrix_flow(stability_data):
    """Cobre a integração do calculate_stability_matrix."""
    ws = WoeStability(date_col="safra")
    matrix = ws.calculate_stability_matrix(stability_data, "feat", "target")
    assert not matrix.empty
    assert "10" in matrix.columns


def test__is_numeric_from_mapper():
    """Cobre a função auxiliar _is_numeric do CategoryMapper."""
    mapper = CategoryMapper()
    assert mapper._is_numeric("123") is True
    assert mapper._is_numeric("12.3") is True
    assert mapper._is_numeric("N/A") is False
    assert mapper._is_numeric("abc") is False
