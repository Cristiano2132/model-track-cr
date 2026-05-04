"""
Evaluation module for model-track.

Provides evaluators for binary, multiclass, and regression classification tasks.
Each evaluator follows a unified interface defined by ``BaseEvaluator``.

Classes:
    BaseEvaluator: Abstract base class for all evaluators.
    BinaryEvaluator: Evaluator for binary classification (KS, AUC, Gini, Brier, log-loss).
    DecisionTable: Capture/decline table for binary risk models.

Example:
    >>> from model_track.evaluation import BinaryEvaluator
    >>> import pandas as pd
    >>> evaluator = BinaryEvaluator()
    >>> y_true = pd.Series([0, 1, 0, 1])
    >>> y_proba = pd.Series([0.1, 0.9, 0.2, 0.8])
    >>> metrics = evaluator.evaluate(y_true, y_proba)
    >>> sorted(metrics.keys())
    ['auc', 'brier_score', 'gini', 'ks', 'log_loss']
"""

from model_track.evaluation.base import BaseEvaluator
from model_track.evaluation.binary import BinaryEvaluator
from model_track.evaluation.decision_table import DecisionTable
from model_track.evaluation.multiclass import MulticlassEvaluator

__all__ = ["BaseEvaluator", "BinaryEvaluator", "DecisionTable", "MulticlassEvaluator"]
