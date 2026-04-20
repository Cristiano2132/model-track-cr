import numpy as np
import pandas as pd

from model_track.preprocessing.memory import DataOptimizer


def test_reduce_mem_usage_full_int_coverage():
    """Cover all integer branches (int8, int16, int32, int64)."""
    df = pd.DataFrame(
        {
            "i8": [10, 20],
            "i16": [1000, -1000],
            "i32": [1000000, -1000000],
            # Linhas 30-33: Valores que excedem int32 (int64)
            "i64": [3000000000, -3000000000],
        }
    )

    optimized = DataOptimizer.reduce_mem_usage(df)

    assert optimized["i8"].dtype == np.int8
    assert optimized["i16"].dtype == np.int16
    assert optimized["i32"].dtype == np.int32
    assert optimized["i64"].dtype == np.int64


def test_reduce_mem_usage_float_coverage():
    """Cover float branches (float32 and float64)."""
    df = pd.DataFrame(
        {
            "f32": [1.5, 2.5],
            # Linha 38: Valor que excede a precisão/limite de float32
            "f64": [1.79e308, -1.79e308],
        }
    )

    optimized = DataOptimizer.reduce_mem_usage(df)

    assert optimized["f32"].dtype == np.float32
    assert optimized["f64"].dtype == np.float64


def test_reduce_mem_usage_category_and_verbose(capsys):
    """Cover 'object' -> 'category' branch and verbose print."""
    df = pd.DataFrame({"cat_col": ["A", "B", "A", "C"], "obj_col": ["text"] * 4})

    # Testa o verbose=True (Linhas 47-51)
    optimized = DataOptimizer.reduce_mem_usage(df, verbose=True)
    captured = capsys.readouterr()

    assert "Initial Memory:" in captured.out
    assert optimized["cat_col"].dtype == "category"
    assert optimized["obj_col"].dtype == "category"


def test_reduce_mem_usage_no_mutation():
    """Ensure immutability."""
    df_original = pd.DataFrame({"a": [1, 2, 3]})
    _ = DataOptimizer.reduce_mem_usage(df_original)

    # Original deve permanecer int64 (padrão pandas)
    assert df_original["a"].dtype == np.int64
