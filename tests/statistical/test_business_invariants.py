import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from model_track.stats.metrics import compute_cramers_v, compute_iv

# Configuração para evitar falsos positivos com DataFrames pequenos ou vazios
ST_TARGET = st.integers(min_value=0, max_value=1)
ST_FEATURE = st.text(min_size=1, alphabet="ABCDE")


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    data_frames(
        columns=[
            column("feature", elements=ST_FEATURE),
            column("target", elements=ST_TARGET),
        ],
        index=range_indexes(min_size=2),
    )
)
def test_iv_non_negativity(df):
    """
    Invariante: O Information Value (IV) deve ser sempre >= 0.
    Matematicamente: (p - q) * ln(p/q) >= 0 para qualquer p, q > 0.
    """
    iv = compute_iv(df, "feature", "target")
    assert iv >= 0, f"IV negativo detectado: {iv}"
    assert np.isfinite(iv)


@given(
    data_frames(
        columns=[
            column("f1", elements=ST_FEATURE),
            column("f2", elements=ST_FEATURE),
        ],
        index=range_indexes(min_size=2),
    )
)
def test_cramers_v_range(df):
    """
    Invariante: O V de Cramer deve estar no intervalo [0, 1].
    """
    v = compute_cramers_v(df, "f1", "f2")
    assert 0 <= v <= 1.0000000000001, f"V de Cramer fora do intervalo [0, 1]: {v}"


@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=100), min_size=3, max_size=5, unique=True))
def test_woe_monotonicity_logic(counts):
    """
    Invariante: Se a taxa de 'Bad' aumenta monotonicamente entre categorias,
    o WoE deve diminuir monotonicamente (ou vice-versa, dependendo da ordem).
    Isso valida que a lógica de cálculo não inverte sinais aleatoriamente.
    """
    # Ordenamos os counts para garantir monotonicidade na entrada
    counts = sorted(counts)

    data = []
    for i, c in enumerate(counts):
        # Categoria i tem 'c' bads e um número fixo de goods
        data.extend([{"feat": str(i), "target": 1}] * c)
        data.extend([{"feat": str(i), "target": 0}] * 50)

    df = pd.DataFrame(data)
    from model_track.woe.calculator import WoeCalculator

    calc = WoeCalculator()
    calc.fit(df, target="target", columns=["feat"])

    # Pegamos os WoEs na ordem das categorias 0, 1, 2...
    woes = [calc.mapping_["feat"][str(i)] for i in range(len(counts))]

    # Como o número de Bads aumenta, o WoE (ln(Good/Bad)) deve diminuir
    diffs = np.diff(woes)
    assert (diffs <= 0).all(), f"WoE não foi monotônico decrescente: {woes}"


@given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5))
def test_iv_extremes_empty_classes(goods, bads):
    """
    Garante que o IV não quebre se uma classe estiver vazia.
    O Laplace smoothing deve lidar com isso.
    """
    data = [{"feature": "A", "target": 0}] * goods + [{"feature": "B", "target": 1}] * bads
    df = pd.DataFrame(data)
    iv = compute_iv(df, "feature", "target")
    assert iv >= 0
    assert np.isfinite(iv)
