import itertools
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from model_track.woe.calculator import WoeCalculator


class WoeStability:
    """Temporal stability analysis of Weight of Evidence (WoE)."""

    def __init__(self, date_col: str) -> None:
        self.date_col = date_col
        self.calc = WoeCalculator()

    def calculate_stability_matrix(
        self, df: pd.DataFrame, feature_col: str, target_col: str
    ) -> pd.DataFrame:
        """
        Calculate the WoE matrix across different time periods.

        Args:
            df: Input DataFrame containing the date and feature columns.
            feature_col: Feature column name to analyze.
            target_col: Target column name.

        Returns:
            pd.DataFrame: A matrix where rows are time periods and columns are categories.
        """
        df_temp = df.copy()
        df_temp[feature_col] = df_temp[feature_col].astype(str).fillna("N/A")

        woe_history = []
        periods = sorted(df_temp[self.date_col].unique())

        for period in periods:
            period_df = df_temp[df_temp[self.date_col] == period].copy()
            self.calc.fit(period_df, target_col, [feature_col])
            woe_dict = self.calc.mapping_.get(feature_col, {})
            woe_dict[self.date_col] = period
            woe_history.append(woe_dict)

        matrix = pd.DataFrame(woe_history).set_index(self.date_col).fillna(0.0)
        return matrix

    def generate_view(
        self, matrix: pd.DataFrame, title: str = "WoE Stability", ax: Any | None = None
    ) -> Any:
        """
        Generate a line plot of WoE stability over time.

        Args:
            matrix: The stability matrix (output of calculate_stability_matrix).
            title: Title for the plot.
            ax: Matplotlib axes object (optional).

        Returns:
            Any: The axes object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        matrix.plot(ax=ax, marker="o")
        ax.set_title(title)
        ax.set_ylabel("WoE")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        return ax


class CategoryMapper:
    """
    Optimized Scorecard Grouper with Cost Function.
    Uses exhaustive search (brute force) to minimize temporal WoE crossings.
    """

    MAX_EXHAUSTIVE_CATEGORIES = 15

    def __init__(self) -> None:
        self.mapping_dict_: dict[str, str] = {}

    def _is_numeric(self, val: str) -> bool:
        if val in ["N/A", "nan", "None"]:
            return False
        try:
            float(val)
            return True
        except ValueError:
            return False

    def _format_num(self, val: str) -> str:
        v = float(val)
        return str(int(v)) if v.is_integer() else str(v)

    def _get_sorted_categories(
        self, categories: list[str], global_woe: pd.Series, is_ordered: bool
    ) -> list[str]:
        if not is_ordered:
            return global_woe.sort_values().index.tolist()  # type: ignore[no-any-return]

        nums, strs, nas = [], [], []
        for c in categories:
            if c in ["N/A", "nan", "None"]:
                nas.append(c)
            else:
                try:
                    nums.append((float(c), c))
                except ValueError:
                    strs.append(c)
        nums.sort(key=lambda x: x[0])
        strs.sort()
        return [x[1] for x in nums] + strs + nas

    def _check_inversion(
        self, a_global: float, b_global: float, a_row: float, b_row: float
    ) -> bool:
        if a_global < b_global - 1e-5:
            return bool(a_row >= b_row)
        if a_global > b_global + 1e-5:
            return bool(a_row <= b_row)
        return bool(abs(a_row - b_row) > 1e-5)

    def _count_row_inversions(self, row: Any, global_woes: list[float], k: int) -> int:
        inversions = 0
        for a in range(k):
            for b in range(a + 1, k):
                if self._check_inversion(global_woes[a], global_woes[b], row[a], row[b]):
                    inversions += 1
        return inversions

    def _score_partition(
        self,
        partition: list[list[str]],
        stability_matrix: pd.DataFrame,
        global_woe: pd.Series,
        k: int,
    ) -> tuple[int, float, int]:
        inversions = 0
        sse = 0
        grouped_global_woes = []
        grouped_safra_woes = []

        for group in partition:
            group_safra_woe = stability_matrix[group].mean(axis=1).to_numpy()
            grouped_safra_woes.append(group_safra_woe)

            group_global = global_woe[group].mean()
            grouped_global_woes.append(group_global)

            for cat in group:
                sse += (global_woe[cat] - group_global) ** 2

        safra_woes = np.array(grouped_safra_woes).T

        for row in safra_woes:
            inversions += self._count_row_inversions(row, grouped_global_woes, k)

        return (inversions, round(sse, 5), k)

    def _format_numeric_range_name(
        self, group_sorted_num: list[str], numeric_cats_orig: list[str], has_na: bool
    ) -> str | None:
        start_idx = numeric_cats_orig.index(group_sorted_num[0])
        expected_slice = numeric_cats_orig[start_idx : start_idx + len(group_sorted_num)]
        if expected_slice != group_sorted_num:
            return None

        min_val = self._format_num(group_sorted_num[0])
        max_val = self._format_num(group_sorted_num[-1])

        if start_idx == 0 and (start_idx + len(group_sorted_num)) == len(numeric_cats_orig):
            name = "All"
        elif start_idx == 0:
            name = f"<={max_val}"
        elif (start_idx + len(group_sorted_num)) == len(numeric_cats_orig):
            name = f">={min_val}"
        else:
            name = f"{min_val} to {max_val}" if min_val != max_val else min_val

        return name + " or N/A" if has_na else name

    def _name_group(
        self, group: list[str], is_all_numeric: bool, numeric_cats_orig: list[str]
    ) -> str:
        has_na = any(c in ["N/A", "nan"] for c in group)
        clean_cats = [c for c in group if c not in ["N/A", "nan"]]
        default_name = " or ".join([str(c) for c in group])

        if not is_all_numeric or not clean_cats:
            return default_name

        group_sorted_num = sorted(clean_cats, key=float)
        range_name = self._format_numeric_range_name(group_sorted_num, numeric_cats_orig, has_na)

        return range_name if range_name else default_name

    def _generate_intelligent_names(
        self, best_partition: list[list[str]] | None, categories: list[str]
    ) -> dict[str, str]:
        numeric_cats_orig = sorted([c for c in categories if self._is_numeric(c)], key=float)
        is_all_numeric = len(numeric_cats_orig) > 0 and len(numeric_cats_orig) == len(
            [c for c in categories if c not in ["N/A", "nan"]]
        )

        suggestion_map = {}
        for group in best_partition or []:
            name = self._name_group(group, is_all_numeric, numeric_cats_orig)
            for cat in group:
                suggestion_map[cat] = name

        self.mapping_dict_ = suggestion_map
        return suggestion_map

    def _greedy_group(
        self,
        stability_matrix: pd.DataFrame,
        min_groups: int = 2,
        is_ordered: bool = False,
    ) -> dict[str, str]:
        """
        Heuristic: Iteratively split groups to find a good partition in O(n^2).
        Useful when number of categories is too large for exhaustive search.
        """
        categories = stability_matrix.columns.tolist()
        n = len(categories)
        global_woe = stability_matrix.mean()
        sorted_cats = self._get_sorted_categories(categories, global_woe, is_ordered)

        # Initial state: one group containing all sorted categories
        best_overall_partition: list[list[str]] = [sorted_cats]
        best_overall_score: tuple[float, float, float] = self._score_partition(
            best_overall_partition, stability_matrix, global_woe, 1
        )

        current_partition = [sorted_cats]

        # Greedy expansion up to n groups
        for k in range(2, n + 1):
            best_k_score: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
            best_k_partition: list[list[str]] | None = None

            # Try splitting each existing group into two
            for i, group in enumerate(current_partition):
                if len(group) < 2:
                    continue

                # Try all possible split points within this group
                for split_idx in range(1, len(group)):
                    new_partition = (
                        current_partition[:i]
                        + [group[:split_idx], group[split_idx:]]
                        + current_partition[i + 1 :]
                    )

                    score = self._score_partition(new_partition, stability_matrix, global_woe, k)

                    if score < best_k_score:
                        best_k_score = score
                        best_k_partition = new_partition

            if best_k_partition is None:
                break

            current_partition = best_k_partition

            # If we reached min_groups, start tracking the overall best
            if k >= min_groups:
                if best_k_score < best_overall_score:
                    best_overall_score = best_k_score
                    best_overall_partition = current_partition

        return self._generate_intelligent_names(best_overall_partition, categories)

    def auto_group(
        self,
        stability_matrix: pd.DataFrame,
        min_groups: int = 2,
        is_ordered: bool = False,
        max_categories: int | None = None,
    ) -> dict[str, str]:
        """
        Find the best grouping of categories based on temporal stability.

        Args:
            stability_matrix: Matrix of WoE values over time.
            min_groups: Minimum number of groups to create.
            is_ordered: Whether the categories have a natural order.
            max_categories: Maximum categories to allow for exhaustive search.

        Returns:
            dict[str, str]: A mapping from original category names to group names.
        """
        categories = stability_matrix.columns.tolist()
        n = len(categories)

        if n <= min_groups:
            return {cat: str(cat) for cat in categories}

        limit = max_categories or self.MAX_EXHAUSTIVE_CATEGORIES
        if n > limit:
            warnings.warn(
                f"CategoryMapper.auto_group: {n} categories exceeds the exhaustive search "
                f"limit ({limit}). Falling back to greedy heuristic. "
                f"Set max_categories to override.",
                PerformanceWarning,
                stacklevel=2,
            )
            return self._greedy_group(stability_matrix, min_groups, is_ordered)

        global_woe = stability_matrix.mean()

        sorted_cats = self._get_sorted_categories(categories, global_woe, is_ordered)

        best_score: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))
        best_partition: list[list[str]] | None = None

        for k in range(min_groups, n + 1):
            for splits in itertools.combinations(range(1, n), k - 1):
                partition = []
                prev = 0
                for split in splits:
                    partition.append(sorted_cats[prev:split])
                    prev = split
                partition.append(sorted_cats[prev:])

                score = self._score_partition(partition, stability_matrix, global_woe, k)

                if score < best_score:
                    best_score = score
                    best_partition = partition

        return self._generate_intelligent_names(best_partition, categories)
