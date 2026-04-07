import pandas as pd

from model_track.preprocessing.types import TypeDetector


def test_type_detector_classification():
    """Garante que as colunas sejam separadas nas listas corretas sem adivinhação de nomes."""
    df = pd.DataFrame(
        {
            "ID": range(1000),
            "TransactionDT": [
                1000 * i for i in range(1000)
            ],  # É int, mas vamos forçar como datetime
            "target": [0, 1] * 500,
            "cat_low": ["A", "B"] * 500,
            "cat_high": [f"cat_{i}" for i in range(1000)],
            "num_cont": [float(i) for i in range(1000)],
        }
    )

    # Passamos as colunas de data e IDs de forma declarativa e segura
    detector = TypeDetector(
        target="target",
        id_cols=["ID"],
        datetime_cols=["TransactionDT"],  # <--- NOVO
    )
    feature_types = detector.detect(df)

    assert "cat_low" in feature_types["categorical_low"]
    assert "cat_high" in feature_types["categorical_high"]
    assert "num_cont" in feature_types["numerical"]
    assert "TransactionDT" in feature_types["datetime"]
    assert "ID" not in feature_types["id_like"]
