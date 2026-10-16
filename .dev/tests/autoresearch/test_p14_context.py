"""P14a CPU checks: payload coverage, no truncation, accounting and admission."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from jevgo.core import BudgetLedger, GenerationRequest, ModelFailure
from test_autotrust import assets  # Shared small CPU-only head/config fixture.

PATH = Path(__file__).resolve().parents[2] / "tools/p14_context.py"
spec = importlib.util.spec_from_file_location("p14_context", PATH)
p14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p14)


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return "<chat>" + "".join(m["content"] for m in messages) + "<assistant>"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return list(text.encode())


@pytest.fixture
def lengths():
    result = object.__new__(p14.Lengths)
    result.tokenizer = CharacterTokenizer()
    return result


def generation_event(n=1):
    return {"event_id": "test", "event_type": "generation_request", "payload": {"attempt": 0,
            "request": {"prompt": "task", "system": "system", "n": n, "max_new_tokens": 256}}}


def test_actual_chat_template_system_and_per_sequence_reserve(assets, lengths):
    payload = p14.CaptureBackend(assets).expand(generation_event(4))[0]
    measured = lengths.measure(payload["endpoint"], payload["payload"], 283)
    assert measured["offline_input_tokens"] == 27  # chat framing, system and user
    assert measured["requested_output_tokens_all_sequences"] == 1024
    assert measured["total_window_tokens"] == 283
    assert measured["fits"] is True
    assert lengths.measure(payload["endpoint"], payload["payload"], 282)["fits"] is False


def test_judge_all_items_and_bare_template(assets, lengths):
    state = {"actions": [{}] * 9, "memory": [{"content": "record"}] * 19}
    event = {"event_type": "judge_request", "payload": {"request": {
        "role": "evidence", "context": "Public state:\n" + json.dumps(state),
        "items": ["first evidence", "second counterevidence"]}}}
    payloads = p14.CaptureBackend(assets).expand(event)
    assert len(payloads) == 2
    for index, capture in enumerate(payloads):
        value = capture["payload"]
        assert f"Item [item-{index}]" in value["prompt"]
        assert event["payload"]["request"]["items"][index] in value["prompt"]
        assert value["add_special_tokens"] is False and value["max_tokens"] == 1
        measured = lengths.measure(capture["endpoint"], value, 8192)
        assert measured["branch_actions"] == 9 and measured["memory_entries"] == 19


def test_counterfactual_and_repair_are_labelled_and_original_untouched():
    state = {"actions": [{"name": "left"}] * 9, "memory": [{"id": i} for i in range(19)]}
    original = generation_event()
    original["payload"]["request"]["prompt"] = "Public state:\n" + json.dumps(state)
    original["payload"]["request"]["seed"] = 18
    versions = dict(p14.variants({"event": original}))
    assert len(versions) == 4
    assert p14.frontier(original["payload"]["request"]["prompt"])["memory_entries"] == 19
    reduced = versions["counterfactual_suffix_memory4"]["payload"]["request"]["prompt"]
    assert p14.frontier(reduced) == {"branch_actions": 9, "memory_entries": 4}
    repair = versions["historical_repair_probe"]["payload"]["request"]
    assert repair["prompt"].endswith(p14.REPAIR_SUFFIX) and repair["seed"] == 19


def test_transient_memory_and_individual_evidence_pool_preserved():
    state = {"actions": [{}] * 9, "memory": [{"id": i} for i in range(19)]}
    event = {"event_type": "judge_request", "payload": {"request": {
        "context": "Public state:\n" + json.dumps(state), "role": "evidence",
        "items": [str(i) for i in range(19)], "options": {}}}}
    reduced = dict(p14.variants({"event": event}))["counterfactual_suffix_memory4"]["payload"]["request"]
    assert p14.frontier(reduced["context"])["memory_entries"] == 6
    assert reduced["items"] == ["15", "16", "17", "18"]


def test_pre_send_rejection_has_no_http_charge_or_truncation(assets, lengths, tmp_path):
    seen = []
    client = httpx.Client(transport=httpx.MockTransport(lambda r: seen.append(r)))
    backend = p14.GuardedBackend("jev_9b", assets_root=assets, lengths=lengths,
                                journal=tmp_path / "http.jsonl", client=client, max_model_len=260)
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure, match="local_context_rejection"):
        backend.generate(GenerationRequest(prompt="too long", max_new_tokens=256), ledger=ledger)
    assert not seen and ledger.used.model_requests == 0
    rows = [json.loads(line) for line in (tmp_path / "http.jsonl").read_text().splitlines()]
    assert rows[-1]["network_sent"] is False and rows[-1]["actual_usage"] is None
    assert rows[-1]["request"]["messages"][-1]["content"] == "too long"


@pytest.mark.parametrize("missing_usage", [False, True])
def test_n4_usage_prompt_once_and_unknown_remains_unknown(assets, lengths, tmp_path, missing_usage):
    def respond(request):
        body = json.loads(request.content)
        count = lengths.measure(request.url.path, body, 4096)["offline_input_tokens"]
        result = {"model": body["model"], "choices": [{"index": i, "message": {"content": "ok"}}
                                                            for i in range(body["n"])]}
        if not missing_usage:
            result["usage"] = {"prompt_tokens": count, "completion_tokens": 8}
        return httpx.Response(200, json=result)
    backend = p14.GuardedBackend("jev_9b", assets_root=assets, lengths=lengths,
                                journal=tmp_path / "http.jsonl", max_model_len=4096,
                                client=httpx.Client(transport=httpx.MockTransport(respond)))
    ledger = BudgetLedger()
    backend.generate(GenerationRequest(prompt="task", n=4), ledger=ledger)
    row = json.loads((tmp_path / "http.jsonl").read_text().splitlines()[-1])
    assert ledger.used.model_requests == 1 and row["network_sent"] is True
    assert row["actual_usage_complete"] is not missing_usage
    if missing_usage:
        assert row["actual_usage"] is None and "offline_service_prompt_match" not in row
        assert ledger.entries[-1].metadata["unknown_usage"] == ("input_tokens", "output_tokens")
    else:
        assert row["offline_service_prompt_match"] is True
        assert row["actual_usage"]["prompt_tokens"] == row["offline_input_tokens"]


def test_http400_body_and_unknown_cost_preserved(assets, lengths, tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(400, json={"error": "window failure"})))
    backend = p14.GuardedBackend("jev_9b", assets_root=assets, lengths=lengths,
                                journal=tmp_path / "http.jsonl", client=client, max_model_len=4096)
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure, match="window failure"):
        backend.generate(GenerationRequest(prompt="task"), ledger=ledger)
    row = json.loads((tmp_path / "http.jsonl").read_text().splitlines()[-1])
    assert row["status_code"] == 400 and "window failure" in row["raw_response"]
    assert row["actual_usage"] is None and not row["actual_usage_complete"]
    assert ledger.used.model_requests == 1


@pytest.mark.parametrize("server_window", [4096, None])
def test_service_mismatch_or_unknown_cannot_admit(monkeypatch, server_window):
    def get(url, **kwargs):
        return httpx.Response(200, request=httpx.Request("GET", url), json={"data": [
            {"id": "autotrust/JEV-9B", "max_model_len": server_window}]})
    monkeypatch.setattr(p14.httpx, "get", get)
    with pytest.raises(ValueError, match="mismatch or unknown"):
        p14.service_admission("http://localhost:8019/v1", 8192)


def test_config_keeps_old_cumulative_budgets():
    original = {"episode": {"seed": 0, "task": "MiniGrid-Empty-5x5-v0"}, "run": {
        "method": "B3", "seed": 0, "max_steps": 20, "options": {"max_new_tokens": 256},
        "budget": {"env_steps": 2000, "model_requests": 3000},
        "commit_reserve": {"env_steps": 20}}}
    result = p14.make_config(original, 8192, 32, "B0", "http://localhost:8019")
    assert result["run"]["budget"] == original["run"]["budget"]
    assert result["run"]["commit_reserve"] == original["run"]["commit_reserve"]
    assert "memory_records" not in original["run"]["options"]
    result["run"]["budget"]["input_tokens"] = 2000000
    with pytest.raises(ValueError, match="declared"):
        p14.validate_config(result)


def test_journal_missing_rows_wrong_raw_usage_and_wrong_entries_fail():
    before = {"event": "request_preflight", "request_number": 1, "request": {"prompt": "same"}}
    after = {**before, "event": "request_finished", "network_sent": True,
             "raw_response": '{"usage":{"prompt_tokens":12,"completion_tokens":1}}',
             "actual_usage": {"prompt_tokens": 12, "completion_tokens": 1},
             "ledger_entries": [{"source": "autotrust.request", "sequence": 1},
                                {"source": "autotrust.response", "sequence": 2}]}
    ledger = {"used": {"model_requests": 1, "input_tokens": 12, "output_tokens": 1},
              "entries": after["ledger_entries"]}
    assert p14.journal_accounting([before, after], ledger)["ok"]
    assert not p14.journal_accounting([before], ledger)["ok"]
    assert not p14.journal_accounting([after], ledger)["ok"]
    assert not p14.journal_accounting([before, after, after], ledger)["ok"]
    assert not p14.journal_accounting([before, {**after, "raw_response": "{}"}], ledger)["ok"]
    assert not p14.journal_accounting([before, {**after, "ledger_entries": []}], ledger)["ok"]


def test_frontier_requires_original_role_successful_request():
    generation = {"role": "generation", "network_sent": True, "status_code": 200,
                  "branch_actions": 9, "memory_entries": 19}
    assert p14.frontier_met([generation], "B0")
    assert not p14.frontier_met([generation], "B3")
    evidence = {**generation, "role": "evidence"}
    assert p14.frontier_met([evidence], "B3")
    assert not p14.frontier_met([{**evidence, "error": "HTTP400"}], "B3")
    assert not p14.frontier_met([{**evidence, "memory_entries": 18}], "B3")
    assert not p14.frontier_met([{**evidence, "network_sent": False}], "B3")


@pytest.mark.parametrize("field", ["within_budget", "declared_over_budget_consistent", "cost_complete"])
def test_overbudget_inconsistent_or_unknown_cost_cannot_pass(field):
    audit = {"within_budget": True, "declared_over_budget_consistent": True, "cost_complete": True}
    assert p14.budget_acceptance(audit)
    audit[field] = False
    assert not p14.budget_acceptance(audit)


def test_real_cpu_loop_journal_and_audit_roundtrip(assets, lengths, tmp_path, monkeypatch):
    """Exercise the unchanged runner/MiniGrid and final serialization with fake HTTP."""
    import yaml
    from jevgo import factory
    config = yaml.safe_load((p14.ROOT / "configs/autoresearch-smoke.yaml").read_text())
    config["model"] = {"backend": "jev_9b", "base_url": "http://fixture", "max_model_len": 4096}
    config["run"].update(method="B0")
    config["run"]["options"]["memory_records"] = 4
    lengths.tokenizer.encode = lambda text, **kw: list(range(len(text) // 10))
    journal_dir = tmp_path / "http"
    journal_dir.mkdir()
    def respond(request):
        body = json.loads(request.content)
        tokens = lengths.measure(request.url.path, body, 4096)["offline_input_tokens"]
        return httpx.Response(200, json={"model": body["model"], "choices": [{"index": 0,
            "message": {"content": '{"action":{"name":"left","arguments":{}},"hypothesis":"","rationale":"fixture"}'}}],
            "usage": {"prompt_tokens": tokens, "completion_tokens": 25}})
    backend = p14.GuardedBackend("jev_9b", assets_root=assets, lengths=lengths,
                                journal=journal_dir / "http.jsonl", max_model_len=4096,
                                client=httpx.Client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(factory, "build_backend", lambda *a, **kw: backend)
    run = tmp_path / "run"
    result = factory.run_from_dict(config, assets_root=assets, output_dir=run)
    assert result.termination_reason.value == "environment_truncated"
    args = SimpleNamespace(run=run, audit=journal_dir, output=tmp_path / "audit.json")
    p14.audit(args)
    audit = json.loads(args.output.read_text())
    assert audit["technical_closed_loop_pass"] is True
    assert audit["event_ledger_result_reconcile"] and audit["journal_accounting"]["ok"]
    assert audit["committed_steps"] == 20 and not audit["objective_success"]
