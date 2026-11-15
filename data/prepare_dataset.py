"""Kiểm tra và chia dữ liệu chat/tool-call (.jsonl) trước khi upload lên Drive.

Mỗi dòng: {"messages": [{"role": ..., "content": ...}, ...]}
Chạy từ thư mục gốc dự án:
    python -m data.prepare_dataset --inputs data/raw/*.jsonl --out-dir data/processed
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from app.tool_calls import parse_reply

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
VI_CHARS = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")
VI_MIN_HITS = 3


def validate(record: object) -> list[str]:
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return ["Thiếu trường 'messages' (danh sách)"]
    msgs = record["messages"]
    if not msgs:
        return ["'messages' rỗng"]
    errors: list[str] = []
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or m.get("role") not in ALLOWED_ROLES:
            errors.append(f"message {i}: role không hợp lệ")
            continue
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            errors.append(f"message {i}: content rỗng hoặc không phải chuỗi")
            continue
        if m["role"] == "assistant":
            parsed = parse_reply(m["content"])
            if parsed.errors:
                errors.append(f"message {i}: {'; '.join(parsed.errors)}")
    if msgs[-1].get("role") != "assistant":
        errors.append("message cuối phải là assistant")
    return errors


def is_vietnamese(record: dict) -> bool:
    text = " ".join(m["content"] for m in record["messages"]).lower()
    return sum(1 for ch in text if ch in VI_CHARS) >= VI_MIN_HITS


def has_tool_call(record: dict) -> bool:
    return any(m["role"] == "assistant" and parse_reply(m["content"]).calls for m in record["messages"])


def load_records(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    good, bad = [], []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                bad.append({"source": f"{path}:{lineno}", "errors": [f"JSON lỗi: {exc.msg}"]})
                continue
            errs = validate(rec)
            if errs:
                bad.append({"source": f"{path}:{lineno}", "errors": errs})
            else:
                good.append(rec)
    return good, bad


def split(records: list[dict], eval_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_eval = max(1, int(len(records) * eval_ratio)) if records else 0
    eval_idx = set(idx[:n_eval])
    train = [records[i] for i in range(len(records)) if i not in eval_idx]
    evals = [records[i] for i in range(len(records)) if i in eval_idx]
    return train, evals


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    good, bad = load_records([Path(p) for p in args.inputs])
    train, evals = split(good, args.eval_ratio, args.seed)
    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", evals)
    write_jsonl(out / "rejects.jsonl", bad)

    vi = sum(is_vietnamese(r) for r in good)
    tool = sum(has_tool_call(r) for r in good)
    print(f"Hợp lệ: {len(good)} | lỗi: {len(bad)}")
    print(f"Train: {len(train)} | eval: {len(evals)}")
    if good:
        print(f"Có tiếng Việt: {vi / len(good):.0%} | có tool_call: {tool / len(good):.0%}")
    if bad:
        print(f"Xem chi tiết lỗi tại {out / 'rejects.jsonl'}")


if __name__ == "__main__":
    main()
