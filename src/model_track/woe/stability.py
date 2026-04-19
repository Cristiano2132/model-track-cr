import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_track.woe.calculator import WoeCalculator


class WoeStability:
    """Análise de estabilidade temporal do WoE."""

    def __init__(self, date_col: str) -> None:
        self.date_col = date_col
        self.calc = WoeCalculator()

    def calculate_stability_matrix(
        self, df: pd.DataFrame, feature_col: str, target_col: str
    ) -> pd.DataFrame:
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
    Agrupador Otimizado de Scorecard com Função de Custo.
    Usa busca exaustiva (força bruta) para minimizar cruzamentos de WoE temporal.
    """

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
            group_safra_woe = stability_matrix[group].mean(axis=1).values
            grouped_safra_woes.append(group_safra_woe)

            group_global = global_woe[group].mean()
            grouped_global_woes.append(group_global)

            for cat in group:
                sse += (global_woe[cat] - group_global) ** 2

        safra_woes = np.array(grouped_safra_woes).T

        for row in safra_woes:
            for a in range(k):
                for b in range(a + 1, k):
                    if grouped_global_woes[a] < grouped_global_woes[b] - 1e-5:
                        if row[a] >= row[b]:
                            inversions += 1
                    elif grouped_global_woes[a] > grouped_global_woes[b] + 1e-5:
                        if row[a] <= row[b]:
                            inversions += 1
                    else:
                        if abs(row[a] - row[b]) > 1e-5:
                            inversions += 1

        return (inversions, round(sse, 5), k)

    def _generate_intelligent_names(
        self, best_partition: list[list[str]] | None, categories: list[str]
    ) -> dict[str, str]:
        numeric_cats_orig = sorted([c for c in categories if self._is_numeric(c)], key=float)
        is_all_numeric = len(numeric_cats_orig) > 0 and len(numeric_cats_orig) == len(
            [c for c in categories if c not in ["N/A", "nan"]]
        )

        suggestion_map = {}
        for group in best_partition or []:
            has_na = any(c in ["N/A", "nan"] for c in group)
            clean_cats = [c for c in group if c not in ["N/A", "nan"]]

            name = " ou ".join([str(c) for c in group])

            if is_all_numeric and len(clean_cats) > 0:
                group_sorted_num = sorted(clean_cats, key=float)
                try:
                    start_idx = numeric_cats_orig.index(group_sorted_num[0])
                    if (
                        numeric_cats_orig[start_idx : start_idx + len(group_sorted_num)]
                        == group_sorted_num
                    ):
                        min_val = self._format_num(group_sorted_num[0])
                        max_val = self._format_num(group_sorted_num[-1])

                        if start_idx == 0 and (start_idx + len(group_sorted_num)) == len(
                            numeric_cats_orig
                        ):
                            name = "Todos"
                        elif start_idx == 0:
                            name = f"<={max_val}"
                        elif (start_idx + len(group_sorted_num)) == len(numeric_cats_orig):
                            name = f">={min_val}"
                        else:
                            name = f"{min_val} a {max_val}" if min_val != max_val else min_val

                        if has_na:
                            name += " ou N/A"
                except ValueError:
                    pass

            for cat in group:
                suggestion_map[cat] = name

        self.mapping_dict_ = suggestion_map
        return suggestion_map

    def auto_group(
        self, stability_matrix: pd.DataFrame, min_groups: int = 2, is_ordered: bool = False
    ) -> dict[str, str]:
        categories = stability_matrix.columns.tolist()
        n = len(categories)

        if n <= min_groups:
            return {cat: str(cat) for cat in categories}

        global_woe = stability_matrix.mean()

        sorted_cats = self._get_sorted_categories(categories, global_woe, is_ordered)

        best_score = (float("inf"), float("inf"), float("inf"))
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
