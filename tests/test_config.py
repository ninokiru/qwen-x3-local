import json
from app.config import load_config


def test_loads_v03_config(tmp_path):
    p = tmp_path / "app.json"
    p.write_text(json.dumps({"server": {"base_url": "http://127.0.0.1:8080"}, "max_steps": 6}), encoding="utf-8")
    cfg = load_config(p)
    assert cfg.max_steps == 6
    assert cfg.runtime.context_size == 4096
    assert cfg.memory.semantic_search is False


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "app.json"
    p.write_text('{"not_a_setting": true}', encoding="utf-8")
    assert load_config(p).workspace == "workspace"
