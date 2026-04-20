import pandas as pd
import pytest
from matplotlib import pyplot as plt

from model_track.woe.stability import CategoryMapper, WoeStability


def test_woe_stability_matrix():
    """Validate if the stability matrix generates periods in the index and categories in the columns."""
    df = pd.DataFrame(
        {
            "safra": ["2023-01", "2023-01", "2023-02", "2023-02"],
            "cat": ["A", "B", "A", "B"],
            "target": [0, 1, 0, 1],
        }
    )

    ws = WoeStability(date_col="safra")
    matrix = ws.calculate_stability_matrix(df, feature_col="cat", target_col="target")

    assert list(matrix.index) == ["2023-01", "2023-02"]
    assert "A" in matrix.columns
    assert "B" in matrix.columns


def test_category_mapper_exhaustive_search():
    """Ensure the algorithm minimizes crossings while resolving severe ambiguities."""
    matrix = pd.DataFrame(
        {
            "C1": [-1.0, -1.0, -1.0],
            "C2": [0.0, 1.5, 0.0],
            "C3": [2.0, -0.5, 2.0],
            "C4": [3.0, 3.0, 3.0],
        },
        index=["safra1", "safra2", "safra3"],
    )

    mapper = CategoryMapper()
    suggested_map = mapper.auto_group(matrix, min_groups=2)

    assert suggested_map["C2"] == suggested_map["C3"]
    assert suggested_map["C1"] != suggested_map["C2"]


def test_category_mapper_smart_intervals():
    """Ensure correct interval naming while forcing internal stability."""
    matrix = pd.DataFrame(
        {
            "2": [1.0, 1.2],
            "1": [1.2, 1.0],  # 1 e 2 se cruzam entre si
            "3": [5.0, 5.2],
            "4": [5.2, 5.0],  # 3 e 4 se cruzam entre si
            "8": [9.0, 9.2],
            "9": [9.1, 9.1],
            "10": [9.2, 9.0],  # 8, 9, 10 se misturam
        },
        index=["safra1", "safra2"],
    )

    mapper = CategoryMapper()
    suggested_map = mapper.auto_group(matrix, min_groups=3, is_ordered=True)

    assert suggested_map["1"] == "<=2"
    assert suggested_map["4"] == "3 to 4"
    assert suggested_map["10"] == ">=8"


def test_category_mapper_ordinal_constraint():
    """Ensure that is_ordered=True prevents grouping of similar risks if they are not adjacent (U-shaped risk)."""
    # 10 e 30 têm o mesmo risco exato (2.0), mas 20 tem risco diferente (-1.0).
    matrix = pd.DataFrame(
        {"10": [2.0, 2.0], "20": [-1.0, -1.0], "30": [2.0, 2.0]}, index=["s1", "s2"]
    )

    mapper = CategoryMapper()

    # 1. No ordinal constraint (Default): Will join '10' with '30' because risk is identical
    map_unordered = mapper.auto_group(matrix, min_groups=2, is_ordered=False)
    assert map_unordered["10"] == map_unordered["30"]

    # 2. With ordinal constraint: Forbidden to join '10' with '30'.
    # It will be forced to separate the 3, or join neighbors (e.g., <=20 or >=20)
    map_ordered = mapper.auto_group(matrix, min_groups=2, is_ordered=True)
    assert map_ordered["10"] != map_ordered["30"]


@pytest.fixture
def stability_data():
    """Generate a valid DataFrame (same list lengths)."""
    return pd.DataFrame(
        {
            "safra": ["2023-01"] * 10 + ["2023-02"] * 10,
            "target": [0, 1] * 10,
            "feat": ["10", "20", "30", "40", "50"] * 4,
        }
    )


@pytest.fixture
def mock_matrix():
    """Generate a WoE matrix for testing generate_view."""
    return pd.DataFrame({"A": [0.1, 0.2], "B": [0.5, 0.4]}, index=["2023-01", "2023-02"])


def test_generate_view_correct_return(mock_matrix):
    """
    Ensure the return is the AX and validate the title.
    """
    ws = WoeStability(date_col="safra")

    # Scenario 1: ax=None - Create new ax internally
    ax_new = ws.generate_view(mock_matrix, title="New Title")
    assert ax_new.get_title() == "New Title"
    plt.close(ax_new.get_figure())  # Close figure via ax reference

    # Scenario 2: ax provided - Use external ax
    fig, ax_ext = plt.subplots()
    ax_returned = ws.generate_view(mock_matrix, ax=ax_ext, title="External Title")

    assert ax_returned is ax_ext
    assert ax_ext.get_title() == "External Title"
    plt.close(fig)


def test_calculate_stability_matrix_flow(stability_data):
    """Cover the integration of calculate_stability_matrix."""
    ws = WoeStability(date_col="safra")
    matrix = ws.calculate_stability_matrix(stability_data, "feat", "target")
    assert not matrix.empty
    assert "10" in matrix.columns


def test__is_numeric_from_mapper():
    """Cover the auxiliary function _is_numeric from CategoryMapper."""
    mapper = CategoryMapper()
    assert mapper._is_numeric("123") is True
    assert mapper._is_numeric("12.3") is True
    assert mapper._is_numeric("N/A") is False
    assert mapper._is_numeric("abc") is False


# n < min_group = 2 categories = 1 -> should return original category without grouping
def test_auto_group_min_groups_exceeds_categories():
    matrix = pd.DataFrame({"C1": [1.0, 1.0]}, index=["safra1", "safra2"])
    mapper = CategoryMapper()
    suggested_map = mapper.auto_group(matrix, min_groups=2)
    assert suggested_map["C1"] == "C1"


def test_format_num_integer_and_float():
    mapper = CategoryMapper()
    assert mapper._format_num("10.0") == "10"
    assert mapper._format_num("10.5") == "10.5"


def test_auto_group_with_non_convertible_numeric():
    matrix = pd.DataFrame(
        {
            "1": [1, 1],
            "x": [2, 2],  # cannot convert to float
        },
        index=["s1", "s2"],
    )

    mapper = CategoryMapper()
    result = mapper.auto_group(matrix, min_groups=1, is_ordered=True)

    assert "x" in result


def test_auto_group_with_na_category():
    matrix = pd.DataFrame(
        {
            "1": [1.0, 1.0],
            "N/A": [1.0, 1.0],
        },
        index=["s1", "s2"],
    )

    mapper = CategoryMapper()
    result = mapper.auto_group(matrix, min_groups=1, is_ordered=True)

    # Should append "or N/A"
    assert any("N/A" in v for v in result.values())


def test_auto_group_mixed_numeric_and_string():
    matrix = pd.DataFrame(
        {
            "1": [1, 1],
            "A": [2, 2],
        },
        index=["s1", "s2"],
    )

    mapper = CategoryMapper()
    result = mapper.auto_group(matrix, min_groups=1, is_ordered=True)

    assert "A" in result
