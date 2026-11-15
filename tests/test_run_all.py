import json

from evals.run_all import run_all


def test_run_all_writes_report_for_every_suite(tmp_path):
    calls = []

    def fake_runner(cases_path, base_url):
        calls.append((cases_path.name, base_url))
        return {"metric": 1.0, "cases": 1}

    suites = {"a": ("evals/cases/a.jsonl", fake_runner), "b": ("evals/cases/b.jsonl", fake_runner)}
    path = run_all("http://x", suites=suites, label="baseline", out_dir=tmp_path)

    assert path.parent == tmp_path and path.name.endswith("_baseline.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert set(report["suites"]) == {"a", "b"} and report["base_url"] == "http://x"
    assert [c[0] for c in calls] == ["a.jsonl", "b.jsonl"]


def test_default_suite_covers_all_eval_cases():
    from evals.run_all import DEFAULT_SUITES

    assert set(DEFAULT_SUITES) == {"tool_call", "code", "chat"}
    for _, (cases_path, _) in DEFAULT_SUITES.items():
        assert len(open(cases_path, encoding="utf-8").read().splitlines()) > 0
