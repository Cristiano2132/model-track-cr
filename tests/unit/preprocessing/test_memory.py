import numpy as np
import pandas as pd

from model_track.preprocessing.memory import DataOptimizer


def test_reduce_mem_usage_stress():
    """Testa limites de tipos, precisão de float e economia real de memória."""
    df = pd.DataFrame(
        {
            "int_bin": [0, 1] * 500,  # Deve virar int8
            "int_small": [100, -100] * 500,  # Deve virar int8
            "int_med": [30000, -30000] * 500,  # Deve virar int16
            "float_simple": [1.5, 2.5] * 500,  # Deve virar float32
            "mixed_col": [1.0, "text"] * 500,  # Mistura -> deve ser mantido/viver objeto
        }
    )

    start_mem = df.memory_usage().sum()
    optimized_df = DataOptimizer.reduce_mem_usage(df.copy())
    end_mem = optimized_df.memory_usage().sum()

    # 1. Validação de Economia
    assert end_mem < start_mem

    # 2. Validação de Tipos Específicos
    assert optimized_df["int_bin"].dtype == np.int8
    assert optimized_df["int_small"].dtype == np.int8
    assert optimized_df["int_med"].dtype == np.int16
    assert optimized_df["float_simple"].dtype == np.float32

    # 3. Validação de Tipos Mistos (Não deve quebrar)
    assert optimized_df["mixed_col"].dtype == object or isinstance(
        optimized_df["mixed_col"].dtype, pd.CategoricalDtype
    )

    # 4. Integridade dos Dados
    assert (optimized_df["int_med"].astype(int) == df["int_med"]).all()


def test_reduce_mem_usage_no_mutation():
    """Garante que a função retorna uma cópia e não altera o DataFrame original."""
    # Setup: Criamos um DF com tipos pesados (int64 por padrão no pandas)
    df_original = pd.DataFrame({"a": [1, 2, 3]})
    original_dtype = df_original["a"].dtype  # int64

    # Action
    df_optimized = DataOptimizer.reduce_mem_usage(df_original)

    # Asserts
    # 1. O otimizado deve ter mudado para int8
    assert df_optimized["a"].dtype == np.int8
    # 2. O original DEVE continuar sendo o que era (Não houve mutação)
    assert df_original["a"].dtype == original_dtype
    # 3. Garantir que são objetos diferentes na memória
    assert df_original is not df_optimized


def test_reduce_mem_usage_output_verbose(capsys):
    """Garante que o optimizer printa o ganho de memória quando verbose=True."""
    # Criamos um DF grande o suficiente para a redução ser óbvia
    df = pd.DataFrame(
        {"int_col": np.random.randint(0, 100, size=10000), "float_col": np.random.rand(10000)}
    )

    # Action
    DataOptimizer.reduce_mem_usage(df, verbose=True)

    # Captura o print
    captured = capsys.readouterr()

    # Assertions
    assert "Memória Inicial:" in captured.out
    assert "Memória Final:" in captured.out
    assert "Redução de:" in captured.out
