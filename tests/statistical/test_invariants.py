# from hypothesis import given, strategies as st
# from model_track.context import ProjectContext
# import os

# @given(st.lists(st.text(min_size=1)), st.text(min_size=1))
# def test_context_persistence_invariant(tmp_path, features, target_name):
#     """
#     Teste de Propriedade: Garante que a persistência do contexto
#     é invariante aos dados contidos nele.
#     """
#     ctx = ProjectContext()
#     ctx.selected_features = features
#     ctx.target = target_name

#     file_path = os.path.join(tmp_path, "pbt_context.pkl")

#     ctx.save(file_path)
#     loaded_ctx = ProjectContext.load(file_path)

#     assert loaded_ctx.selected_features == features
#     assert loaded_ctx.target == target_name
