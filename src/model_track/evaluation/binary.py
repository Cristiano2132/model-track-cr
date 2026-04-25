"""Binary classification evaluator — KS, AUC, Gini, Brier, log-loss."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from model_track.base import TaskType
from model_track.evaluation.base import BaseEvaluator


class BinaryEvaluator(BaseEvaluator):
    """
    Evaluator for binary classification models.

    Computes KS, AUC, Gini, Brier score, and log-loss. Supports temporal
    monitoring via ``report()``, which breaks metrics down by time period.

    Attributes:
        task_type: Always ``TaskType.BINARY``.
        positive_class: The value that represents the positive class in
            the target series (default: ``1``).

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from model_track.evaluation import BinaryEvaluator
        >>> evaluator = BinaryEvaluator()
        >>> y_true = pd.Series([0, 1, 0, 1, 1])
        >>> y_proba = pd.Series([0.1, 0.9, 0.2, 0.8, 0.7])
        >>> metrics = evaluator.evaluate(y_true, y_proba)
        >>> 0.0 <= metrics["auc"] <= 1.0
        True
        >>> 0.0 <= metrics["ks"] <= 1.0
        True
    """

    task_type = TaskType.BINARY

    def __init__(self, positive_class: Any = 1) -> None:
        self.positive_class = positive_class

    def evaluate(
        self,
        y_true: pd.Series,
        y_proba: pd.Series,
    ) -> dict[str, float]:
        """
        Compute all binary classification metrics.

        Args:
            y_true: True binary labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            dict[str, float]: Keys are ``auc``, ``ks``, ``gini``,
            ``brier_score``, ``log_loss``.

        Raises:
            ValueError: If ``y_true`` contains more than 2 unique classes.

        Example:
            >>> import pandas as pd
            >>> from model_track.evaluation import BinaryEvaluator
            >>> ev = BinaryEvaluator()
            >>> metrics = ev.evaluate(pd.Series([0,1,0,1]), pd.Series([0.1,0.9,0.2,0.8]))
            >>> round(metrics["gini"], 4)
            1.0
        """
        if y_true.nunique() > 2:
            raise ValueError(
                f"BinaryEvaluator expects <= 2 unique classes, got {y_true.nunique()}."
            )

        y_true_arr = y_true.to_numpy()
        y_proba_arr = y_proba.to_numpy()

        auc = float(roc_auc_score(y_true_arr, y_proba_arr))
        ks = self._compute_ks(y_true_arr, y_proba_arr)
        gini = 2.0 * auc - 1.0
        brier = float(brier_score_loss(y_true_arr, y_proba_arr))
        ll = float(log_loss(y_true_arr, y_proba_arr))

        return {
            "auc": auc,
            "ks": ks,
            "gini": gini,
            "brier_score": brier,
            "log_loss": ll,
        }

    def report(
        self,
        df: pd.DataFrame,
        target: str,
        score_col: str,
        date_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute metrics per time period (or globally if ``date_col`` is None).

        Args:
            df: DataFrame containing labels, scores, and optionally dates.
            target: Column name of the binary target.
            score_col: Column name of predicted probabilities.
            date_col: Optional column to group by (e.g., month/year).

        Returns:
            pd.DataFrame: One row per period (or a single row if no date_col),
            with columns for each metric.

        Example:
            >>> import pandas as pd
            >>> from model_track.evaluation import BinaryEvaluator
            >>> df = pd.DataFrame({
            ...     "target": [0, 1, 0, 1, 1, 0],
            ...     "score":  [0.1, 0.9, 0.2, 0.8, 0.7, 0.3],
            ...     "month":  ["jan", "jan", "jan", "feb", "feb", "feb"],
            ... })
            >>> ev = BinaryEvaluator()
            >>> report = ev.report(df, "target", "score", date_col="month")
            >>> list(report.columns)
            ['period', 'auc', 'ks', 'gini', 'brier_score', 'log_loss']
        """
        if date_col is None:
            metrics: dict[str, Any] = dict(self.evaluate(df[target], df[score_col]))
            metrics["period"] = "overall"
            return pd.DataFrame([metrics])[
                ["period", "auc", "ks", "gini", "brier_score", "log_loss"]
            ]

        rows: list[dict[str, Any]] = []
        for period, group in df.groupby(date_col, sort=True):
            if group[target].nunique() < 2:
                continue
            metrics = dict(self.evaluate(group[target], group[score_col]))
            metrics["period"] = period
            rows.append(metrics)

        if not rows:
            return pd.DataFrame(columns=["period", "auc", "ks", "gini", "brier_score", "log_loss"])

        return pd.DataFrame(rows)[["period", "auc", "ks", "gini", "brier_score", "log_loss"]]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_ks(self, y_true: np.ndarray[Any, Any], y_proba: np.ndarray[Any, Any]) -> float:
        """Compute the Kolmogorov-Smirnov statistic between positive/negative score distributions."""
        scores_pos = y_proba[y_true == self.positive_class]
        scores_neg = y_proba[y_true != self.positive_class]
        if len(scores_pos) == 0 or len(scores_neg) == 0:
            return 0.0
        ks_stat, _ = ks_2samp(scores_pos, scores_neg)
        return float(ks_stat)
