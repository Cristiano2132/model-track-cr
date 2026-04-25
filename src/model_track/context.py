import warnings
from datetime import datetime
from typing import Any

import joblib
import pandas as pd

from model_track.base import TaskType


class VersionMismatchError(Exception):
    """Exception raised when loading an incompatible ProjectContext schema."""


class ProjectContext:
    """
    DTO (Data Transfer Object) that centralizes the modeling state.
    Allows saving and loading project progress.
    """

    def __init__(self) -> None:
        self.schema_version: str = "2.0"
        self.task_type: TaskType = TaskType.BINARY
        self.target: str | None = None
        self.metadata: dict[str, Any] = {}
        self.bins_map: dict[str, list[float]] = {}
        self.woe_maps: dict[str, dict[str, float]] = {}
        self.selected_features: list[str] = []
        self.model_hash: str | None = None
        self.training_date: datetime | None = None
        self.reference_stats: dict[str, Any] | None = None

    def validate(self) -> None:
        """
        Validate the schema version and structural integrity of the context.
        Raises VersionMismatchError if the schema is severely outdated or incompatible.
        """
        if getattr(self, "schema_version", None) != "2.0":
            raise VersionMismatchError(
                f"Incompatible schema version: {getattr(self, 'schema_version', 'legacy')}."
            )

    def summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing the current context state.
        """
        data = {
            "Field": [
                "Schema Version",
                "Task Type",
                "Target",
                "Model Hash",
                "Training Date",
                "Features Selected",
                "Metadata Keys",
                "Bins Maps",
                "WOE Maps",
            ],
            "Value": [
                self.schema_version,
                self.task_type.value,
                self.target,
                self.model_hash,
                self.training_date.isoformat() if self.training_date else None,
                len(self.selected_features),
                len(self.metadata),
                len(self.bins_map),
                len(self.woe_maps),
            ],
        }
        return pd.DataFrame(data)

    def save(self, path: str) -> None:
        """
        Persist the context to a binary file.

        Args:
            path: Path to save the context file.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ProjectContext":
        """
        Load a context from disk. Automatically handles schema migrations.

        Args:
            path: Path to the saved context file.

        Returns:
            ProjectContext: The loaded context instance.
        """
        ctx = joblib.load(path)

        if not hasattr(ctx, "schema_version"):
            warnings.warn(
                "Loading legacy context without schema_version. Attributes will be backfilled.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Backfill missing attributes to gracefully upgrade
            ctx.schema_version = "2.0"
            ctx.task_type = TaskType.BINARY
            ctx.model_hash = None
            ctx.training_date = None
            if not hasattr(ctx, "reference_stats"):
                ctx.reference_stats = None

        ctx.validate()
        return ctx  # type: ignore[no-any-return]
