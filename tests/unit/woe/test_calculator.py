import pandas as pd

from model_track.woe.calculator import WoeCalculator


def test_woe_calculator_fit_transform():
    """Ensure correct WoE calculation and fallback to 0.0 for new categories."""
    df_train = pd.DataFrame({"cat": ["A", "A", "B", "B", None], "target": [0, 0, 1, 1, 0]})

    df_test = pd.DataFrame(
        {
            "cat": ["A", "B", "C"]  # 'C' é uma categoria nova
        }
    )

    calc = WoeCalculator()
    calc.fit(df_train, target="target", columns=["cat"])

    res_train = calc.transform(df_train, columns=["cat"])
    res_test = calc.transform(df_test, columns=["cat"])

    # 'A' só tem bons (target=0), então o WoE deve ser um número positivo
    assert res_train["cat_woe"].iloc[0] > 0
    # 'B' só tem maus (target=1), então o WoE deve ser um número negativo
    assert res_train["cat_woe"].iloc[2] < 0
    # 'C' não existia no treino, o WoE mapeado deve ser neutro (0.0)
    assert res_test["cat_woe"].iloc[2] == 0.0


def test_woe_calculator_unfitted_column_skip():
    """Test skipping columns that were not mapped during fit."""
    df_trans = pd.DataFrame({"col_b": ["Z", "Z"]})

    calc = WoeCalculator()
    # executar transform sem fit deve levantar um erro
    try:
        calc.transform(df_trans, columns=["col_b"])
    except RuntimeError as e:
        assert str(e) == "WoeCalculator must be fitted before transforming."
