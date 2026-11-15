import json

from data.prepare_dataset import load_records, split, validate


def _rec(*msgs):
    return {"messages": [{"role": r, "content": c} for r, c in msgs]}


def test_valid_record_passes():
    rec = _rec(("user", "Đọc file a.txt"),
               ("assistant", '<tool_call>{"name": "read_file", "arguments": {"path": "a.txt"}}</tool_call>'))
    assert validate(rec) == []


def test_rejects_bad_tool_call_and_missing_final_assistant():
    bad = _rec(("user", "hi"), ("assistant", "<tool_call>{oops}</tool_call>"))
    errs = validate(bad)
    assert any("JSON" in e for e in errs)
    assert validate(_rec(("user", "hi"), ("assistant", "ok"), ("user", "tiếp"))) == [
        "message cuối phải là assistant"]


def test_load_records_separates_good_and_bad(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join([
        json.dumps(_rec(("user", "chào"), ("assistant", "xin chào"))),
        "{not json",
        json.dumps({"messages": []}),
    ]), encoding="utf-8")
    good, bad = load_records([p])
    assert len(good) == 1 and len(bad) == 2


def test_split_is_deterministic_and_disjoint():
    recs = [_rec(("user", str(i)), ("assistant", str(i))) for i in range(100)]
    train1, eval1 = split(recs, 0.1, seed=7)
    train2, eval2 = split(recs, 0.1, seed=7)
    assert eval1 == eval2 and len(eval1) == 10 and len(train1) == 90
    assert not {json.dumps(r) for r in train1} & {json.dumps(r) for r in eval1}
