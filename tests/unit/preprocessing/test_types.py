import pandas as pd

from model_track.preprocessing.types import TypeDetector


def test_type_detector_classification():
    """Garante que as colunas sejam separadas nas listas corretas."""
    size = 1000
    df = pd.DataFrame(
        {
            "ID": range(size),
            "TransactionDT": [1000 * i for i in range(size)],
            "target": [0, 1] * (size // 2),
            "cat_low": ["A", "B"] * (size // 2),
            "cat_high": [f"cat_{i}" for i in range(size)],
            "num_cont": [float(i) for i in range(size)],
        }
    )

    detector = TypeDetector(
        target="target",
        id_cols=["ID"],
        datetime_cols=["TransactionDT"],
    )
    feature_types = detector.detect(df)

    assert "cat_low" in feature_types["categorical_low"]
    assert "cat_high" in feature_types["categorical_high"]
    assert "num_cont" in feature_types["numerical"]
    assert "TransactionDT" in feature_types["datetime"]


def test_type_detector_numerical_branch_coverage():
    """Cobre as linhas 53 e 56 do TypeDetector com listas de tamanho igual."""
    size = 50
    df = pd.DataFrame(
        {
            # n_unique = 1 (Abaixo do threshold de 15) -> categorical_low (Linha 53)
            "num_low": [1.0] * size,
            # n_unique = 50 (Acima do threshold de 15) -> numerical (Linha 56)
            "num_high": [float(i) for i in range(size)],
        }
    )

    detector = TypeDetector()
    types = detector.detect(df)

    assert "num_low" in types["categorical_low"]
    assert "num_high" in types["numerical"]


def test_type_detector_id_like_and_datetime_native():
    """Cobre o caminho de ID mascarado (Linha 50) e datetime nativo."""
    size = 100
    df = pd.DataFrame(
        {
            # Inteiro com alta cardinalidade (> 5% do DF) -> id_like
            "id_masked": list(range(size)),
            # Tipo datetime nativo do pandas
            "dt_native": pd.to_datetime(["2023-01-01"] * size),
        }
    )

    detector = TypeDetector()
    types = detector.detect(df)

    assert "id_masked" in types["id_like"]
    assert "dt_native" in types["datetime"]


def test_type_detector_fallback():
    """Cobre o retorno None para tipos desconhecidos como timedelta."""
    size = 10
    df = pd.DataFrame({"time_diff": pd.to_timedelta([f"{i} days" for i in range(size)])})

    detector = TypeDetector()
    types = detector.detect(df)

    # A coluna não entra em nenhuma lista
    for cat_list in types.values():
        assert "time_diff" not in cat_list
