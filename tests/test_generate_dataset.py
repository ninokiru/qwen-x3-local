import json

from data.generate_dataset import TOOL_TEMPLATES, build, generate, raw_question_keys, verify_code
from data.prepare_dataset import has_tool_call, is_vietnamese, validate


def _small(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "tool_call.jsonl").write_text("", encoding="utf-8")
    return build(seed=7, scale=0.1, raw_dir=raw)


def test_generated_records_are_valid(tmp_path):
    data = _small(tmp_path)
    assert set(data) == {"chat_vi", "code_vi", "tool_call"}
    for rows in data.values():
        assert rows
        for rec in rows:
            assert validate(rec) == []
            assert is_vietnamese(rec)


def test_every_generated_code_answer_runs(tmp_path):
    for rec in _small(tmp_path)["code_vi"]:
        assert verify_code(rec) == []


def test_every_generated_tool_answer_calls_a_tool():
    # Khuôn "no_call" cố ý dạy model KHÔNG gọi công cụ, các khuôn còn lại phải có tool_call.
    calling = [t for t in TOOL_TEMPLATES if t.name != "no_call"]
    for rec in generate(calling, seed=11, max_attempts_per_sample=5):
        assert has_tool_call(rec)


def test_same_seed_is_reproducible(tmp_path):
    a = _small(tmp_path)
    b = _small(tmp_path)
    assert {k: [json.dumps(r, sort_keys=True) for r in v] for k, v in a.items()} == \
           {k: [json.dumps(r, sort_keys=True) for r in v] for k, v in b.items()}


def test_generator_skips_questions_already_in_raw(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "chat_vi.jsonl").write_text(json.dumps({"messages": [
        {"role": "user", "content": "Docker là gì?"},
        {"role": "assistant", "content": "Docker đóng gói ứng dụng."},
    ]}, ensure_ascii=False) + "\n", encoding="utf-8")

    assert "docker là gì?" in raw_question_keys(tmp_path / "raw")
    data = build(seed=7, scale=0.3, raw_dir=tmp_path / "raw")
    questions = {next(m["content"] for m in r["messages"] if m["role"] == "user").strip().lower()
                 for r in data["chat_vi"]}
    assert "docker là gì?" not in questions


def test_verify_code_flags_wrong_answer():
    bad = {"messages": [
        {"role": "user", "content": "tính tổng"},
        {"role": "assistant", "content": "```python\nprint(sum(range(5)))\n```\nKết quả là **99**."},
    ]}
    assert verify_code(bad)

    broken = {"messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "```python\nprint(undefined_name)\n```\nLỗi."},
    ]}
    assert any("lỗi khi chạy" in e for e in verify_code(broken))
