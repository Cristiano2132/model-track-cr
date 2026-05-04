"""Multiclass classification evaluator — Accuracy, F1, Precision, Recall, Log-loss."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

from model_track.base import TaskType
from model_track.evaluation.base import BaseEvaluator


class MulticlassEvaluator(BaseEvaluator):
    """
    Evaluator for multiclass classification models.

    Computes Accuracy, F1-score, Precision, Recall, and Log-loss.
    Supports macro, micro, and weighted averaging.

    Attributes:
        task_type: Always ``TaskType.MULTICLASS``.
        average: The averaging method to use for precision, recall, and F1
            (default: ``'macro'``). Options: 'macro', 'micro', 'weighted'.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from model_track.evaluation import MulticlassEvaluator
        >>> evaluator = MulticlassEvaluator(average='macro')
        >>> y_true = pd.Series([0, 1, 2, 0, 1, 2])
        >>> y_pred = pd.Series([0, 2, 2, 0, 0, 1])
        >>> metrics = evaluator.evaluate(y_true, y_pred=y_pred)
        >>> metrics["accuracy"]
        0.5
    """

    task_type = TaskType.MULTICLASS

    def __init__(self, average: str = "macro") -> None:
        if average not in ["macro", "micro", "weighted"]:
            raise ValueError(f"Unsupported average method: {average}")
        self.average = average

    def evaluate(
        self,
        y_true: pd.Series,
        y_proba: pd.DataFrame | np.ndarray[Any, Any] | None = None,
        y_pred: pd.Series | np.ndarray[Any, Any] | None = None,
    ) -> dict[str, float]:
        """
        Compute multiclass classification metrics.

        Args:
            y_true: True class labels.
            y_proba: Predicted probabilities (matrix of shape N x K).
                If provided, log_loss will be computed.
            y_pred: Predicted class labels. If not provided, it will be
                inferred from y_proba (argmax).

        Returns:
            dict[str, float]: Keys are ``accuracy``, ``precision``, ``recall``,
            ``f1``, and optionally ``log_loss``.

        Raises:
            ValueError: If ``y_true`` contains 2 or fewer classes (should use BinaryEvaluator).
            ValueError: If neither ``y_proba`` nor ``y_pred`` is provided.
        """
        if y_true.nunique() <= 2:
            raise ValueError(
                f"MulticlassEvaluator expects > 2 unique classes, got {y_true.nunique()}. "
                "Use BinaryEvaluator for binary tasks."
            )

        if y_proba is None and y_pred is None:
            raise ValueError("Either y_proba or y_pred must be provided.")

        y_true_arr = y_true.to_numpy()

        if y_pred is None and y_proba is not None:
            # Infer y_pred from y_proba
            if isinstance(y_proba, pd.DataFrame):
                y_pred_arr = y_proba.idxmax(axis=1).to_numpy()
                y_proba_arr = y_proba.to_numpy()
            else:
                y_pred_arr = np.argmax(y_proba, axis=1)
                y_proba_arr = y_proba
        else:
            y_pred_arr = y_pred.to_numpy() if isinstance(y_pred, pd.Series) else y_pred
            y_proba_arr = y_proba.to_numpy() if isinstance(y_proba, pd.DataFrame) else y_proba

        metrics = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "precision": float(
                precision_score(y_true_arr, y_pred_arr, average=self.average, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true_arr, y_pred_arr, average=self.average, zero_division=0)
            ),
            "f1": float(f1_score(y_true_arr, y_pred_arr, average=self.average, zero_division=0)),
        }

        if y_proba_arr is not None:
            metrics["log_loss"] = float(log_loss(y_true_arr, y_proba_arr))

        return metrics

    def report(
        self,
        df: pd.DataFrame,
        target: str,
        score_cols: list[str] | None = None,
        pred_col: str | None = None,
        date_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute metrics per time period (or globally if ``date_col`` is None).

        Args:
            df: DataFrame containing labels, scores/predictions, and optionally dates.
            target: Column name of the multiclass target.
            score_cols: List of column names for predicted probabilities (one per class).
            pred_col: Column name of predicted class labels.
            date_col: Optional column to group by (e.g., month/year).

        Returns:
            pd.DataFrame: One row per period with metric columns.
        """
        metric_keys = ["accuracy", "precision", "recall", "f1"]
        if score_cols is not None:
            metric_keys.append("log_loss")

        def _get_eval(data: pd.DataFrame) -> dict[str, Any]:
            y_true = data[target]
            y_proba = data[score_cols] if score_cols is not None else None
            y_pred = data[pred_col] if pred_col is not None else None
            return self.evaluate(y_true, y_proba=y_proba, y_pred=y_pred)

        if date_col is None:
            res = _get_eval(df)
            res["period"] = "overall"
            return pd.DataFrame([res])[["period"] + metric_keys]

        rows: list[dict[str, Any]] = []
        for period, group in df.groupby(date_col, sort=True):
            if group[target].nunique() <= 2:
                continue
            res = _get_eval(group)
            res["period"] = period
            rows.append(res)

        if not rows:
            return pd.DataFrame(columns=["period"] + metric_keys)

        return pd.DataFrame(rows)[["period"] + metric_keys]

    def confusion_matrix(self, y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
        """
        Compute confusion matrix.

        Returns:
            pd.DataFrame: Rows are True classes, Columns are Predicted classes.
        """
        classes = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        return pd.DataFrame(cm, index=classes, columns=classes)

    def classification_report(self, y_true: pd.Series, y_pred: pd.Series) -> str:
        """Return a text report of the main classification metrics."""
        return str(classification_report(y_true, y_pred))
