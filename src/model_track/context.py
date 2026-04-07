from typing import Any

import joblib


class ProjectContext:
    """
    DTO (Data Transfer Object) que centraliza o estado da modelagem.
    Permite salvar e carregar o progresso do projeto.
    """

    def __init__(self) -> None:
        self.metadata: dict[str, Any] = {}
        self.bins_map: dict[str, list[float]] = {}
        self.woe_maps: dict[str, dict[str, float]] = {}
        self.selected_features: list[str] = []
        self.target: str | None = None

    def save(self, path: str) -> None:
        """Persiste o contexto em um arquivo binário."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ProjectContext":
        """Carrega um contexto do disco."""
        return joblib.load(path)  # type: ignore[no-any-return]
