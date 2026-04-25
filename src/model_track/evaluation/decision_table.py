"""Decision table for binary classification — fraud/risk capture analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_DEFAULT_CAPTURE_LEVELS: list[float] = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]


class DecisionTable:
    """
    Generate a business-oriented capture table for binary classification models.

    For each desired capture level (e.g. "catch 50 % of fraud"), the table
    shows the corresponding score cutoff, decline rate, hit rate (precision),
    and false-negative rate.

    Args:
        capture_levels: Desired capture fractions (between 0 and 1, exclusive).
            Defaults to ``[0.05, 0.10, …, 0.90]``.

    Example:
        >>> import pandas as pd
        >>> from model_track.evaluation import DecisionTable
        >>> df = pd.DataFrame({
        ...     "target": [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        ...     "proba":  [0.95, 0.85, 0.70, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05],
        ... })
        >>> dt = DecisionTable(capture_levels=[0.50, 1.00])
        >>> table = dt.generate(df, target="target", proba="proba")
        >>> list(table.columns)
        ['target_capture_pct', 'orders_declined_pct', 'actual_tpr_pct', 'fnr_pct', 'hit_rate_pct', 'cutoff']
    """

    def __init__(
        self,
        capture_levels: list[float] | None = None,
    ) -> None:
        levels = capture_levels if capture_levels is not None else _DEFAULT_CAPTURE_LEVELS
        self._validate_levels(levels)
        self.capture_levels = levels
        self._table: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        target: str,
        proba: str,
    ) -> pd.DataFrame:
        """
        Compute the decision/capture table.

        Args:
            df: DataFrame containing binary labels and predicted probabilities.
            target: Column name for the binary target (0/1).
            proba: Column name for predicted probabilities.

        Returns:
            pd.DataFrame with columns:
                - ``target_capture_pct``: desired capture level (%).
                - ``orders_declined_pct``: % of total records declined.
                - ``actual_tpr_pct``: actual True Positive Rate achieved (%).
                - ``fnr_pct``: False Negative Rate (%).
                - ``hit_rate_pct``: precision at this threshold (%).
                - ``cutoff``: score threshold.

        Raises:
            ValueError: If target column has fewer than 2 unique values or
                contains no positive cases.

        Example:
            >>> import pandas as pd
            >>> from model_track.evaluation import DecisionTable
            >>> df = pd.DataFrame({
            ...     "target": [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            ...     "proba":  [0.95, 0.85, 0.70, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05],
            ... })
            >>> dt = DecisionTable(capture_levels=[0.50])
            >>> table = dt.generate(df, target="target", proba="proba")
            >>> int(table.iloc[0]["target_capture_pct"])
            50
        """
        self._validate_input(df, target)

        df_sorted = (
            df[[target, proba]].sort_values(by=proba, ascending=False).reset_index(drop=True)
        )

        total_positives = int(df_sorted[target].sum())
        total_records = len(df_sorted)

        cumulative_positives = df_sorted[target].cumsum()
        tpr_cumulative = cumulative_positives / total_positives

        rows: list[dict[str, float]] = []
        for level in self.capture_levels:
            mask = tpr_cumulative >= level
            if not mask.any():  # pragma: no cover
                continue

            idx = int(mask.idxmax())
            declined_count = idx + 1
            orders_declined_pct = (declined_count / total_records) * 100

            actual_tpr = float(tpr_cumulative.iloc[idx]) * 100
            fnr = 100.0 - actual_tpr
            cutoff = float(df_sorted[proba].iloc[idx])

            cumul_pos_at_idx = float(cumulative_positives.iloc[idx])
            hit_rate = (cumul_pos_at_idx / declined_count) * 100

            rows.append(
                {
                    "target_capture_pct": round(level * 100),
                    "orders_declined_pct": round(orders_declined_pct, 2),
                    "actual_tpr_pct": round(actual_tpr, 2),
                    "fnr_pct": round(fnr, 2),
                    "hit_rate_pct": round(hit_rate, 2),
                    "cutoff": round(cutoff, 4),
                }
            )

        self._table = pd.DataFrame(rows)
        return self._table

    def cutoff_for_capture(self, target_capture: float) -> float:
        """
        Interpolate the score cutoff for a given capture level.

        Args:
            target_capture: Desired capture fraction (e.g. 0.50 for 50 %).

        Returns:
            float: Interpolated cutoff score.

        Raises:
            RuntimeError: If ``generate()`` has not been called yet.

        Example:
            >>> import pandas as pd
            >>> from model_track.evaluation import DecisionTable
            >>> df = pd.DataFrame({
            ...     "target": [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            ...     "proba":  [0.95, 0.85, 0.70, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05],
            ... })
            >>> dt = DecisionTable(capture_levels=[0.30, 0.50, 0.70, 1.00])
            >>> _ = dt.generate(df, target="target", proba="proba")
            >>> cutoff = dt.cutoff_for_capture(0.50)
            >>> isinstance(cutoff, float)
            True
        """
        table = self._require_table()
        return float(
            np.interp(
                target_capture * 100,
                table["actual_tpr_pct"].to_numpy(),
                table["cutoff"].to_numpy(),
            )
        )

    def decline_rate_for_capture(self, target_capture: float) -> float:
        """
        Interpolate the decline rate for a given capture level.

        Args:
            target_capture: Desired capture fraction (e.g. 0.50 for 50 %).

        Returns:
            float: Interpolated decline rate (%).

        Raises:
            RuntimeError: If ``generate()`` has not been called yet.

        Example:
            >>> import pandas as pd
            >>> from model_track.evaluation import DecisionTable
            >>> df = pd.DataFrame({
            ...     "target": [1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            ...     "proba":  [0.95, 0.85, 0.70, 0.60, 0.55, 0.40, 0.30, 0.20, 0.10, 0.05],
            ... })
            >>> dt = DecisionTable(capture_levels=[0.30, 0.50, 0.70, 1.00])
            >>> _ = dt.generate(df, target="target", proba="proba")
            >>> rate = dt.decline_rate_for_capture(0.50)
            >>> isinstance(rate, float)
            True
        """
        table = self._require_table()
        return float(
            np.interp(
                target_capture * 100,
                table["actual_tpr_pct"].to_numpy(),
                table["orders_declined_pct"].to_numpy(),
            )
        )

    def plot(self, ax: Any = None) -> Any:
        """
        Plot Capture Rate vs Decline Rate.

        Args:
            ax: Optional matplotlib Axes to draw on. If ``None``, creates a
                new figure.

        Returns:
            matplotlib.axes.Axes: The axes with the plot.

        Raises:
            RuntimeError: If ``generate()`` has not been called yet.
        """
        table = self._require_table()

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            ) from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        ax.plot(
            table["orders_declined_pct"],
            table["actual_tpr_pct"],
            marker="o",
            linewidth=2,
            color="#2563EB",
            label="Capture vs Decline",
        )
        ax.set_xlabel("Orders Declined (%)")
        ax.set_ylabel("Fraud Captured (%)")
        ax.set_title("Decision Table — Capture vs Decline Trade-off")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_table(self) -> pd.DataFrame:
        if self._table is None:
            raise RuntimeError("No table generated yet. Call generate() first.")
        return self._table

    @staticmethod
    def _validate_levels(levels: list[float]) -> None:
        for lvl in levels:
            if not 0.0 < lvl <= 1.0:
                raise ValueError(f"Each capture level must be in (0, 1], got {lvl}.")

    @staticmethod
    def _validate_input(df: pd.DataFrame, target: str) -> None:
        if df[target].nunique() < 2:
            raise ValueError(
                "Target column must contain at least 2 unique values for a decision table."
            )
        if (df[target] == 1).sum() == 0:
            raise ValueError("Target column contains no positive cases (value=1).")
