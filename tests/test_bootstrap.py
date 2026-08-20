from enterprise_rag.config import load_settings


def test_settings_use_project_root(tmp_path):
    settings = load_settings(tmp_path)
    assert settings.root_dir == tmp_path.resolve()
    assert settings.embedding_dimension > 0
    assert settings.final_top_k > 0

