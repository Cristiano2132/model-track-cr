"""Regression evaluator — RMSE, MAE, R², MAPE, Median AE."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from model_track.base import TaskType
from model_track.evaluation.base import BaseEvaluator


class RegressionEvaluator(BaseEvaluator):
    """
    Evaluator for regression models.

    Computes RMSE, MAE, R², MAPE, and Median Absolute Error.
    Supports temporal monitoring and residual analysis.

    Attributes:
        task_type: Always ``TaskType.REGRESSION``.

    Example:
        >>> import pandas as pd
        >>> from model_track.evaluation import RegressionEvaluator
        >>> evaluator = RegressionEvaluator()
        >>> y_true = pd.Series([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = pd.Series([2.5, 0.0, 2.1, 7.8])
        >>> metrics = evaluator.evaluate(y_true, y_pred)
        >>> metrics["rmse"] < 1.0
        True
    """

    task_type = TaskType.REGRESSION

    def evaluate(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
    ) -> dict[str, float]:
        """
        Compute all regression metrics.

        Args:
            y_true: True numeric values.
            y_pred: Predicted numeric values.

        Returns:
            dict[str, float]: Keys are ``rmse``, ``mae``, ``r2``,
            ``mape``, ``median_ae``.
        """
        y_true_arr = y_true.to_numpy()
        y_pred_arr = y_pred.to_numpy()

        # Handle zeros in MAPE
        if (y_true_arr == 0).any():
            warnings.warn(
                "y_true contains zeros. MAPE may be undefined or misleading.",
                UserWarning,
                stacklevel=2,
            )

        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
        r2 = float(r2_score(y_true_arr, y_pred_arr))
        mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))
        med_ae = float(median_absolute_error(y_true_arr, y_pred_arr))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "median_ae": med_ae,
        }

    def report(
        self,
        df: pd.DataFrame,
        target: str,
        pred_col: str,
        date_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute metrics per time period (or globally if ``date_col`` is None).

        Args:
            df: DataFrame containing labels, scores, and optionally dates.
            target: Column name of the numeric target.
            pred_col: Column name of predicted values.
            date_col: Optional column to group by (e.g., month/year).

        Returns:
            pd.DataFrame: One row per period with metric columns.
        """
        metric_keys = ["rmse", "mae", "r2", "mape", "median_ae"]

        if date_col is None:
            metrics = self.evaluate(df[target], df[pred_col])
            res: dict[str, Any] = dict(metrics)
            res["period"] = "overall"
            return pd.DataFrame([res])[["period"] + metric_keys]

        rows: list[dict[str, Any]] = []
        for period, group in df.groupby(date_col, sort=True, observed=False):
            if len(group) == 0:
                continue
            metrics = self.evaluate(group[target], group[pred_col])
            res_period: dict[str, Any] = dict(metrics)
            res_period["period"] = period
            rows.append(res_period)

        if not rows:
            return pd.DataFrame(columns=["period"] + metric_keys)

        return pd.DataFrame(rows)[["period"] + metric_keys]

    def residual_plot(self, y_true: pd.Series, y_pred: pd.Series, ax: Any = None) -> Any:
        """
        Generate a residual vs fitted plot.

        Note: Requires matplotlib and seaborn.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        residuals = y_true - y_pred
        sns.scatterplot(x=y_pred, y=residuals, ax=ax)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Residuals vs Fitted")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")

        return ax

    def prediction_interval_coverage(
        self,
        y_true: pd.Series,
        y_lower: pd.Series,
        y_upper: pd.Series,
    ) -> float:
        """
        Compute the proportion of true values within the prediction interval.

        Args:
            y_true: True values.
            y_lower: Lower bound of the interval.
            y_upper: Upper bound of the interval.

        Returns:
            float: Coverage proportion [0, 1].
        """
        within_interval = (y_true >= y_lower) & (y_true <= y_upper)
        return float(within_interval.mean())
