"""
Preprocessing module for model-track.
Includes tools for data auditing, memory optimization, type detection, and categorical encoding.

Example:
    >>> from model_track.preprocessing import DataOptimizer, TypeDetector
    >>> # Optimize memory
    >>> df = DataOptimizer.reduce_mem_usage(df)
    >>> # Detect feature types
    >>> detector = TypeDetector(target="target")
    >>> feature_types = detector.detect(df)
"""

from model_track.preprocessing.auditor import DataAuditor
from model_track.preprocessing.encoders import OrdinalEncoder
from model_track.preprocessing.memory import DataOptimizer
from model_track.preprocessing.types import TypeDetector

__all__ = ["DataAuditor", "OrdinalEncoder", "DataOptimizer", "TypeDetector"]
