import pandas as pd

from model_track.preprocessing.encoders import OrdinalEncoder


def test_ordinal_encoder_fit_transform():
    """Valida o mapeamento e o tratamento de categorias não vistas (-1)."""
    df_train = pd.DataFrame({"cat_col": ["A", "B", "A", None]})
    df_test = pd.DataFrame({"cat_col": ["B", "C", None]})

    encoder = OrdinalEncoder()
    # Fit no treino
    encoder.fit(df_train, columns=["cat_col"])

    # Transform no Treino e Teste
    res_train = encoder.transform(df_train, columns=["cat_col"])
    res_test = encoder.transform(df_test, columns=["cat_col"])

    # Assertions
    assert res_train["cat_col"].dtype == int
    assert res_test["cat_col"].dtype == int
    # O valor 'C' não existia no treino, deve virar -1
    assert (res_test["cat_col"] == -1).any()
