import importlib.util
import json
from pathlib import Path

from jevgo.core import Usage

PATH = Path(__file__).resolve().parents[2] / "analysis/p14_remediation_summary.py"
SPEC = importlib.util.spec_from_file_location("p14_remediation_summary", PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def test_independent_reconciliation_detects_dropped_failed_cost(tmp_path):
    usage = Usage(generation_calls=1, model_requests=1, input_tokens=7).model_dump(mode="json")
    entry = {"usage": usage, "metadata": {"usage_complete": False, "unknown_usage": ["output_tokens"]}}
    ledger = {"used": usage, "entries": [entry]}
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    (tmp_path / "result.json").write_text(json.dumps({"usage": usage}))
    request = {"event": "request", "call": 1, "request": {"prompt": "public only"}}
    finish = {**request, "event": "finished", "ledger_entries": [entry], "error": "failure"}
    (tmp_path / "logical.jsonl").write_text(json.dumps(request) + "\n" + json.dumps(finish) + "\n")
    report = tool.summarize_condition(tmp_path)
    assert all(report["checks"].values())
    assert not report["model_cost_complete"]
    assert report["cost_by_role"]["generation:unspecified"]["input_tokens"] == 7
    ledger["used"] = {**usage, "input_tokens": 0}
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    report = tool.summarize_condition(tmp_path)
    assert not report["checks"]["ledger_sum_matches"]
    assert not report["checks"]["logical_model_cost_matches_ledger"]
