from typing import Any

import joblib


class ProjectContext:
    """
    DTO (Data Transfer Object) that centralizes the modeling state.
    Allows saving and loading project progress.
    """

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.bins_map: dict[str, list[float]] = {}
        self.woe_maps: dict[str, dict[str, float]] = {}
        self.selected_features: list[str] = []
        self.target: str | None = None
        self.reference_stats: dict[str, Any] = {}

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
        Load a context from disk.

        Args:
            path: Path to the saved context file.

        Returns:
            ProjectContext: The loaded context instance.
        """
        return joblib.load(path)  # type: ignore[no-any-return]
