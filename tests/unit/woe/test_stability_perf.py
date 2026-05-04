import pandas as pd
import pytest
from pandas.errors import PerformanceWarning

from model_track.woe.stability import CategoryMapper


def test_category_mapper_exhaustive_no_warning():
    """Ensure small number of categories does not trigger warning."""
    matrix = pd.DataFrame(
        {
            "A": [1.0, 1.1],
            "B": [2.0, 2.1],
            "C": [3.0, 3.1],
        },
        index=["s1", "s2"],
    )
    mapper = CategoryMapper()

    import warnings

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        mapper.auto_group(matrix, min_groups=2)
        # Check that PerformanceWarning was NOT raised
        assert not any(isinstance(r.message, PerformanceWarning) for r in record)


def test_category_mapper_greedy_trigger_warning():
    """Ensure large number of categories triggers PerformanceWarning and fallback."""
    # Create 16 categories to exceed default limit of 15
    cats = {f"C{i}": [float(i), float(i) + 0.1] for i in range(16)}
    matrix = pd.DataFrame(cats, index=["s1", "s2"])

    mapper = CategoryMapper()

    with pytest.warns(PerformanceWarning, match="exceeds the exhaustive search limit"):
        result = mapper.auto_group(matrix, min_groups=2)

    assert len(result) == 16


def test_category_mapper_greedy_quality():
    """Verify that greedy heuristic still produces a sensible grouping."""
    # Group 1: 1, 2, 3 (similar risk)
    # Group 2: 10, 11 (similar risk)
    matrix = pd.DataFrame(
        {
            "1": [1.0, 1.0],
            "2": [1.1, 0.9],
            "3": [0.9, 1.1],
            "10": [10.0, 10.0],
            "11": [10.1, 9.9],
        },
        index=["s1", "s2"],
    )

    mapper = CategoryMapper()
    # Force greedy by setting max_categories to 2
    with pytest.warns(PerformanceWarning):
        result = mapper.auto_group(matrix, min_groups=2, max_categories=2, is_ordered=True)

    assert result["1"] == result["3"]
    assert result["10"] == result["11"]
    assert result["1"] != result["10"]
    # Check interval names
    assert "<=3" in result["1"]
    assert ">=10" in result["10"]


def test_category_mapper_greedy_speed():
    """Verify that 30 categories complete quickly (exhaustive would take years)."""
    import time

    # 30 categories
    cats = {f"{i:02d}": [float(i), float(i)] for i in range(30)}
    matrix = pd.DataFrame(cats, index=["s1", "s2"])

    mapper = CategoryMapper()

    start_time = time.time()
    # Should use greedy fallback
    with pytest.warns(PerformanceWarning):
        mapper.auto_group(matrix, min_groups=5)
    end_time = time.time()

    duration = end_time - start_time
    # Increased limit to 15s to avoid CI flakiness.
    # The goal is to ensure it doesn't hang (O(2^n)), not to bench hardware.
    assert duration < 15.0
