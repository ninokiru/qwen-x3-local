"""Chạy toàn bộ eval suite, lưu báo cáo JSON vào evals/reports/.

    python -m evals.run_all --label baseline

Tên báo cáo: <timestamp>_<label>.json. Bỏ trống --label thì chỉ dùng timestamp.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from evals import chat_eval, code_eval, tool_call_eval

DEFAULT_SUITES = {
    "tool_call": ("evals/cases/tool_call.jsonl", tool_call_eval.run),
    "code": ("evals/cases/code.jsonl", code_eval.run),
    "chat": ("evals/cases/chat.jsonl", chat_eval.run),
}


def run_all(base_url: str, suites: dict | None = None, label: str = "", out_dir: str | Path = "evals/reports") -> Path:
    suites = suites or DEFAULT_SUITES
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"), "base_url": base_url, "suites": {}}
    for name, (cases_path, runner) in suites.items():
        report["suites"][name] = runner(Path(cases_path), base_url)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out / (f"{stamp}_{label}.json" if label else f"{stamp}.json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--label", default="", help="nhãn gắn vào tên báo cáo, ví dụ baseline")
    parser.add_argument("--out-dir", default="evals/reports")
    args = parser.parse_args()

    path = run_all(args.base_url, label=args.label, out_dir=args.out_dir)
    report = json.loads(path.read_text(encoding="utf-8"))
    for name, scores in report["suites"].items():
        items = ", ".join(f"{k}={v:.0%}" if isinstance(v, float) else f"{k}={v}" for k, v in scores.items())
        print(f"{name:<10} {items}")
    print(f"Báo cáo: {path}")


if __name__ == "__main__":
    main()
