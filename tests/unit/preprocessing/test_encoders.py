import pandas as pd
import pytest

from model_track.preprocessing.encoders import OrdinalEncoder


def test_ordinal_encoder_fit_transform():
    """Validate mapping and handling of unseen categories (-1)."""
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
    assert res_test.loc[1, "cat_col"] == -1
    # Garante que o nulo (N/A) foi mapeado para um valor positivo consistente
    assert res_train.loc[3, "cat_col"] == res_test.loc[2, "cat_col"]


def test_ordinal_encoder_exceptions():
    """Test error validations."""
    encoder = OrdinalEncoder()
    df = pd.DataFrame({"col1": ["A", "B"]})

    # Error if transforming before fit
    with pytest.raises(RuntimeError, match="must be fitted"):
        encoder.transform(df, columns=["col1"])

    # Error if columns is None in fit
    with pytest.raises(ValueError, match="list of columns must be provided"):
        encoder.fit(df, columns=None)

    # Fit to test the next exception
    encoder.fit(df, columns=["col1"])

    # Error if columns is None in transform
    with pytest.raises(ValueError, match="list of columns must be provided"):
        encoder.transform(df, columns=None)


def test_ordinal_encoder_column_not_in_mapping():
    """Test defensive logic when a requested column was not fitted."""
    df_fit = pd.DataFrame({"col1": ["A", "B"]})
    df_trans = pd.DataFrame({"col2": ["X", "Y"]})

    encoder = OrdinalEncoder()
    encoder.fit(df_fit, columns=["col1"])

    # Se pedirmos para transformar col2 (que não foi fitada),
    # a linha 39 (if col in self.mapping_) deve pular a coluna e retornar o DF original
    res = encoder.transform(df_trans, columns=["col2"])

    # Verifica se a col2 continua sendo objeto/string (não foi alterada)
    assert res["col2"].dtype == object
