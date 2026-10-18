"""Failure accounting and immutable fixed-state provenance for the P14 harness."""
import importlib.util
import json
from pathlib import Path

import pytest

from jevgo.core import BudgetLedger, GenerationRequest, ModelFailure, ModelIdentity, Usage

PATH = Path(__file__).resolve().parents[2] / "tools/p14_remediation.py"
SPEC = importlib.util.spec_from_file_location("p14_remediation", PATH)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def test_failed_call_keeps_request_consumption_and_unknown_usage(tmp_path):
    class Backend:
        identity = ModelIdentity(backend="fixture", model="fixture")
        def generate(self, request, *, ledger, phase):
            ledger.record(Usage(generation_calls=1, model_requests=1, input_tokens=17),
                          source="fixture", phase=phase, failed=True,
                          metadata={"usage_complete": False, "unknown_usage": ["output_tokens"]})
            raise ModelFailure("transport failed")
    ledger = BudgetLedger()
    trace = tool.TraceBackend(Backend(), tmp_path)
    with pytest.raises(ModelFailure):
        trace.generate(GenerationRequest(prompt="unchanged request"), ledger=ledger)
    rows = [json.loads(line) for line in (tmp_path / "logical.jsonl").read_text().splitlines()]
    assert rows[0]["request"]["prompt"] == "unchanged request"
    assert rows[1]["error"]["type"] == "ModelFailure"
    assert rows[1]["ledger_entries"][0]["metadata"]["unknown_usage"] == ["output_tokens"]
    assert ledger.used.model_requests == 1


def test_fixed_state_reuses_exact_ids_and_checks_hash(tmp_path):
    state = tool.fixed_state({"id": "fixed", "setup_actions": ["forward"]}, BudgetLedger())
    path = tmp_path / "state.json"
    tool.write(path, state.model_dump(mode="json"))
    condition = {"state_file": str(path), "state_sha256": tool.sha(path)}
    ledger = BudgetLedger()
    assert tool.fixed_state(condition, ledger) == state
    assert ledger.used.env_steps == 0  # source setup cost is recorded separately
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash changed"):
        tool.fixed_state(condition, ledger)


def test_manifest_rejects_unbounded_or_training_conditions():
    manifest = {"max_attempts": 1, "model": {"backend": "semif"}, "conditions": [
        {"id": "bounded", "model_key": "semif", "budget": {"model_requests": 4}, "run": {}}]}
    with pytest.raises(ValueError, match="finite"):
        tool.validate_manifest(manifest)
    manifest["conditions"][0]["budget"].update(input_tokens=1000, output_tokens=1000)
    manifest["conditions"][0]["run"]["learning_mode"] = "joint"
    with pytest.raises(ValueError, match="frozen"):
        tool.validate_manifest(manifest)
