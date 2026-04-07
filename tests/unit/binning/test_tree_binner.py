import numpy as np
import pandas as pd
import pytest

from model_track.binning.tree_binner import TreeBinner


def test_tree_binner_fit_transform():
    """Garante que o TreeBinner cria bins baseados no target e lida com NaNs."""
    # Criamos um sinal claro: valores baixos -> target 0, valores altos -> target 1
    df = pd.DataFrame({"feature": [1, 2, 3, 10, 11, 12, np.nan], "target": [0, 0, 0, 1, 1, 1, 0]})

    binner = TreeBinner(max_depth=2, min_samples_leaf=1)
    binner.fit(df, column="feature", target="target")

    # O transform deve retornar strings
    result = binner.transform(df, column="feature")

    assert result.nunique() > 1
    assert result.iloc[-1] == "N/A"  # O último valor era NaN
    assert isinstance(result.iloc[0], str)


def test_tree_binner_not_fitted():
    """Garante erro se tentar transformar antes de fitar."""
    binner = TreeBinner()
    with pytest.raises(RuntimeError):
        binner.transform(pd.DataFrame({"a": [1]}), column="a")
