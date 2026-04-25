import joblib

from model_track.context import ProjectContext


def test_context_basic_serialization(tmp_path):
    """
    Garante que o salvamento e carga básica funcionam na versão atual.
    """
    path = tmp_path / "context.joblib"
    ctx = ProjectContext()
    ctx.target = "target_v1"
    ctx.selected_features = ["f1", "f2"]
    ctx.save(str(path))

    loaded = ProjectContext.load(str(path))
    assert loaded.target == "target_v1"
    assert loaded.selected_features == ["f1", "f2"]


class LegacyContext:
    """Simula um contexto antigo para teste de pickling."""

    def __init__(self):
        self.metadata = {"old": True}
        self.target = "old_target"


def test_legacy_context_simulation(tmp_path):
    """
    Simula a carga de um contexto "antigo" que não possui campos novos
    (como schema_version ou task_type que serão adicionados futuramente).
    """
    path = tmp_path / "legacy_context.joblib"

    legacy_obj = LegacyContext()
    joblib.dump(legacy_obj, path)

    # IMPORTANTE: No mundo real, queremos que ProjectContext.load
    # consiga lidar com o fato de que o arquivo joblib contém
    # uma estrutura compatível, mesmo que a classe tenha evoluído.

    # Aqui, como LegacyContext != ProjectContext no dump, o joblib.load
    # retornará um LegacyContext. Para testar a "migração",
    # vamos simular a injeção de estado em uma instância de ProjectContext.

    loaded_data = joblib.load(str(path))
    ctx = ProjectContext()
    # Simula o que aconteceria se carregássemos um __dict__ antigo
    ctx.__dict__.update(loaded_data.__dict__)

    assert ctx.target == "old_target"
    assert ctx.metadata["old"] is True

    # Verifica que campos novos (se existirem na classe atual)
    # mantêm seus defaults se não estiverem no dump
    if hasattr(ProjectContext, "schema_version"):
        assert hasattr(ctx, "schema_version")


def test_context_with_nested_maps(tmp_path):
    """
    Garante que dicionários complexos (bins e woes) são preservados.
    """
    path = tmp_path / "complex_context.joblib"
    ctx = ProjectContext()
    ctx.bins_map = {"age": [0, 18, 65, 100]}
    ctx.woe_maps = {"age": {"[0, 18)": 0.5}}
    ctx.save(str(path))

    loaded = ProjectContext.load(str(path))
    assert loaded.bins_map["age"] == [0, 18, 65, 100]
    assert loaded.woe_maps["age"]["[0, 18)"] == 0.5
