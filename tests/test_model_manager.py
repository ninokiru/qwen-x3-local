import pytest
from app.model_manager import ModelManager


def test_builds_server_args_for_local_model(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "base.gguf").write_bytes(b"x")
    args = ModelManager(tmp_path).server_args("models/base.gguf", 4096, 2)
    assert "--host" in args and "127.0.0.1" in args and "-c" in args


def test_rejects_model_outside_project(tmp_path):
    manager = ModelManager(tmp_path)
    with pytest.raises(ValueError):
        manager.resolve_model("../secret.gguf")
