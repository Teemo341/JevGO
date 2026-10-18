"""CPU evidence tests: no pretrained model loading and no CUDA access."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pretrial = load("p14_semif_pretrial", ROOT / ".dev/tools/p14_semif_pretrial.py")
costs = load("p14_semif_costs", ROOT / ".dev/analysis/p14_semif_costs.py")

ACTION = {"name": "forward", "arguments": {}}
VALID = json.dumps({"action": ACTION, "hypothesis": "unknown", "rationale": "try"})


@pytest.mark.parametrize("text,stage", [
    (VALID[:-1], "json"),
    ('```json\n' + VALID, "json"),
    ('{"action":{"name":"forward","arguments":{}}}', "schema"),
    (json.dumps({"action": {"name": "forward"}, "hypothesis": "x", "rationale": "x"}), "schema"),
    (VALID.replace('"forward"', '"right"'), "public_action"),
    ('{"action":{},"action":{},"hypothesis":"x","rationale":"x"}', "json"),
    (VALID[:-1] + ', "extra":NaN}', "json"),
])
def test_reject_incomplete_or_nonconforming(text, stage):
    result = pretrial.validate_proposal(text, [ACTION])
    assert result["passed"] is False
    assert result["error_stage"] == stage


def test_complete_fence_and_unknown_public_legality():
    assert pretrial.validate_proposal('```json\n' + VALID + '\n```', [ACTION])["passed"]
    result = pretrial.validate_proposal(VALID, None)
    assert result["schema_valid"] and result["public_action_legal"] is None and result["passed"]
    assert result["legality_note"].startswith("Unknown:")


@pytest.mark.parametrize("text", [
    VALID, '```json\n' + VALID + '\n```', VALID[:-1],
    '{"action":{"name":"forward","arguments":{"x":[{"a":1,"a":2}]}},"hypothesis":"h","rationale":"r"}',
    VALID.replace('"arguments": {}', '"arguments": {"value":1e999}'),
    VALID.replace('"arguments": {}', '"arguments": []'),
    VALID.replace('"forward"', '"right"'),
])
@pytest.mark.parametrize("allowed", [[ACTION], [], None])
def test_development_validation_uses_production_contract(text, allowed):
    from jevgo.research import ProposalValidationError, parse_generated_proposal
    actions = tuple(pretrial.Action.model_validate(action) for action in allowed or ())
    result = pretrial.validate_proposal(text, allowed)
    try:
        parse_generated_proposal(text, admissible_actions=actions)
    except ProposalValidationError as exc:
        assert not result["passed"]
        assert result["error_stage"] == exc.stage and result["error"] == exc.reason
    else:
        assert result["passed"] and result["error_stage"] is None


def test_capture_preserves_generate_and_observes_eos():
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[9], to_dict=lambda: {"eos_token_id": [9]})

        def generate(self, *, input_ids, max_new_tokens, **kwargs):
            assert max_new_tokens == 3
            return SimpleNamespace(sequences=torch.cat((input_ids, torch.tensor([[4, 9]])), dim=1))

    model = Model()
    backend = SimpleNamespace(model=model)
    assert "generate" not in model.__dict__
    with pretrial.capture_generation(backend) as captures:
        response = model.generate(input_ids=torch.tensor([[1, 2]]), max_new_tokens=3, temperature=0.7)
    assert "generate" not in model.__dict__
    assert response.sequences.tolist() == [[1, 2, 4, 9]]
    assert captures[0]["generated_token_ids"] == [4, 9]
    assert captures[0]["terminal_eos"] and captures[0]["finish_reason"] is None
    assert captures[0]["invocation_options"]["temperature"] == 0.7


def test_length_does_not_imply_eos_or_finish_reason():
    row = pretrial.stop_evidence([1, 2], 2, [9])
    assert row["at_output_limit"] and not row["terminal_eos"]
    assert row["finish_reason"] is None
    assert pretrial.stop_evidence([1, 2], 8, [9])["stop_evidence"].endswith("unknown")
    assert pretrial.stop_evidence([1, 9], 2, [9])["stop_evidence"] == "terminal_eos_at_limit"


def event(kind, payload):
    return {"event_type": kind, "payload": payload}


def usage(sequence, source, phase, **values):
    return event("usage", {"sequence": sequence, "source": source, "phase": phase,
                          "budget_phase": "search" if phase == "replay" else phase,
                          "usage": pretrial.Usage(**values).model_dump(mode="json")})


def test_cost_partition_preserves_initialization_real_commits_and_tail():
    rows = [usage(1, "minigrid.step", "commit", env_steps=1),  # reset replay verification
            event("initial_state", {}),
            event("generation_request", {"phase": "search", "request": {}}),
            usage(2, "semif.generate", "search", generation_calls=1),
            usage(3, "semif.model.generate", "search", model_requests=4, input_tokens=400),
            event("judge_request", {"phase": "search", "request": {"role": "evidence"}}),
            usage(4, "semif.score", "search", judge_calls=1),
            usage(5, "semif.direct", "search", model_requests=6, input_tokens=900),
            usage(6, "minigrid.step", "replay", env_steps=3, replay_steps=3),
            usage(7, "minigrid.step", "search", env_steps=1),
            usage(8, "minigrid.step", "commit", env_steps=1),
            event("transition", {"phase": "commit", "state": {"state_id": "one"},
                                 "candidate": {"action": ACTION}, "step": {"success": False}}),
            event("judge_request", {"phase": "commit", "request": {"role": "stop"}}),
            usage(9, "semif.score", "commit", judge_calls=1),
            usage(10, "semif.direct", "commit", model_requests=1, input_tokens=50),
            event("generation_request", {"phase": "search", "request": {}}),
            usage(11, "semif.generate", "search", generation_calls=1),
            event("error", {"error": "BudgetExceeded: input_tokens"})]
    segments = costs.partition_events(rows)
    assert [row["kind"] for row in segments] == ["initialization", "committed_step", "uncommitted_tail"]
    assert segments[0]["usage"]["env_steps"] == 1
    assert segments[1]["usage"]["env_steps"] == 5
    assert segments[1]["categories"]["restore_replay"]["replay_steps"] == 3
    assert segments[1]["categories"]["judge_evidence"]["judge_calls"] == 1
    assert segments[1]["categories"]["judge_evidence"]["model_requests"] == 6
    assert segments[1]["categories"]["judge_stop"]["model_requests"] == 1
    assert segments[2]["commit_transition_count"] == 0
    assert [sequence for segment in segments for sequence in segment["ledger_sequences"]] == list(range(1, 12))


def test_cost_rejects_unattributable_direct_score():
    with pytest.raises(ValueError, match="preceding judge_request"):
        costs.partition_events([usage(1, "semif.direct", "search", model_requests=1)])


def test_call_audit_distinguishes_rejected_request_and_real_generation():
    detail = {"source": "semif.model.generate", "input_tokens": 9, "output_tokens": 3,
              "unknown_usage": [], "usage_complete": True, "failed": False}
    returned = usage(3, "semif.model.generate.result", "search", output_tokens=3)
    returned["payload"]["metadata"] = detail
    rows = [event("generation_request", {"phase": "search", "request": {"n": 1}}),
            usage(1, "semif.generate", "search", generation_calls=1),
            usage(2, "semif.model.generate", "search", model_requests=1, input_tokens=9), returned,
            event("proposal_samples", {"texts": [VALID]}),
            event("generation_request", {"phase": "search", "request": {"n": 1}}),
            event("error", {"error": "BudgetExceeded"})]
    calls = costs.audit_semif_calls(rows)
    assert calls[0]["internal_request_count"] == 1 and calls[0]["status"] == "complete_response"
    assert calls[1]["internal_request_count"] == 0 and calls[1]["status"] == "request_logged_no_semif_charge"
    rows[0]["payload"]["request"]["n"] = 2
    with pytest.raises(ValueError, match="sample count"):
        costs.audit_semif_calls(rows)


def test_judge_audit_checks_itemwise_request_metadata():
    request = {"role": "evidence", "items": ["first", "second"]}
    rows = [event("judge_request", {"phase": "search", "request": request}),
            usage(1, "semif.score", "search", judge_calls=1)]
    details = []
    for index in range(2):
        detail = {"source": "semif.direct", "input_tokens": 10 + index, "output_tokens": 0,
                  "unknown_usage": [], "usage_complete": True, "failed": False}
        rows.append(usage(index * 2 + 2, "semif.direct", "search", input_tokens=10 + index, model_requests=1))
        returned = usage(index * 2 + 3, "semif.direct.result", "search")
        returned["payload"]["metadata"] = detail
        rows.append(returned)
        details.append(dict(detail))
    rows.append(event("judge", {"scores": [0.1, 0.2], "metadata": {"role": "evidence", "num_model_requests": 2, "requests": details}}))
    assert costs.audit_semif_calls(rows)[0]["internal_request_count"] == 2
    details[1]["input_tokens"] = 99
    with pytest.raises(ValueError, match="metadata does not match"):
        costs.audit_semif_calls(rows)


def test_extract_preserves_request_seed_and_repair_suffix(tmp_path):
    request = pretrial.GenerationRequest(prompt="prefix\nPublic state:\n" + json.dumps({"observation": {"admissible_actions": [ACTION]}}) + "\nThe previous proposals were invalid. Return only the required JSON object.", seed=8).model_dump(mode="json")
    rows = [{"event_id": "request", "event_type": "generation_request", "payload": {"attempt": 1, "request": request}},
            {"event_id": "usage", **usage(1, "semif.model.generate", "commit", input_tokens=15, output_tokens=74, model_requests=1)},
            {"event_id": "proposal", "episode_id": "seed=1", "event_type": "proposal_samples", "payload": {"attempt": 1, "texts": [VALID[:-1]], "identity": {"revision": "frozen"}}}]
    path = tmp_path / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    case = pretrial.extract_case(path, "proposal")
    assert case["request"] == request and case["request"]["seed"] == 8
    assert case["original_usage"]["output_tokens"] == 74
    assert case["original_validation"]["error_stage"] == "json"


def test_actual_cpu_loop_segments_and_precommit_budget_stop(tmp_path):
    from jevgo.factory import run_from_dict
    payload = {"episode": {"environment": "minigrid", "task": "MiniGrid-Empty-5x5-v0", "seed": 0},
               "run": {"method": "B3", "max_steps": 2, "simulations": 2,
                       "options": {"memory_records": 4}},
               "model": {"backend": "deterministic"}}
    run_from_dict(payload, assets_root=ROOT, output_dir=tmp_path / "loop")
    report = costs.analyze(tmp_path / "loop")
    assert report["reconciliation_passed"] and report["committed_steps"] == 2
    assert [row["kind"] for row in report["segments"]] == ["initialization", "committed_step", "committed_step"]
    payload["run"]["budget"] = {"model_requests": 0}
    run_from_dict(payload, assets_root=ROOT, output_dir=tmp_path / "stopped")
    stopped = costs.analyze(tmp_path / "stopped")
    assert stopped["termination_reason"] == "budget_exhausted" and stopped["committed_steps"] == 0
    assert not stopped["completed_requested_commits_or_objective_success"]


def test_fresh_prepare_separates_actual_environment_horizon_and_commit_limit(tmp_path, monkeypatch):
    """The saved config must actually reach factory/reset with env20 and run2."""
    from jevgo.core import BudgetLedger, EpisodeSpec, RunConfig
    from jevgo.factory import build_environment, run_from_dict

    sources = (("seed0", "a0"), ("seed0", "a1"), ("seed1", "b0"), ("seed1", "b1"))
    evidence = tmp_path / "evidence"
    for seed in (0, 1):
        directory = evidence / f"seed{seed}"
        directory.mkdir(parents=True)
        source = {"episode": EpisodeSpec(environment="minigrid", task="MiniGrid-Empty-5x5-v0",
                                        seed=seed, options={"verify_replay": True}).model_dump(mode="json"),
                  "run": RunConfig(max_steps=20, seed=seed).model_dump(mode="json")}
        (directory / "config.json").write_text(json.dumps(source))
    monkeypatch.setattr(pretrial, "SOURCES", sources)
    # Request extraction has separate tests; this fixture isolates config assembly.
    monkeypatch.setattr(pretrial, "extract_case", lambda path, identifier: {
        "case_id": identifier, "request": pretrial.GenerationRequest(prompt="fixture").model_dump(mode="json"),
        "original_text": "{"})
    output = tmp_path / "prepared"
    pretrial.prepare(evidence, output)
    payload = json.loads((output / "c/conditions/output256-seed0-sim8.json").read_text())
    episode = EpisodeSpec.model_validate(payload["episode"])
    config = RunConfig.model_validate(payload["run"])
    environment = build_environment(episode, config, assets_root=ROOT, output_dir=tmp_path / "adapter")
    try:
        environment.reset(episode, ledger=BudgetLedger())
        assert environment._env.unwrapped.max_steps == 20
        assert config.max_steps == 2
    finally:
        environment.close()
    # Also exercise the actual runner limit. The fixture's first2 actions do not
    # finish the task; env20 must not report environment truncation at commit2.
    payload["model"] = {"backend": "deterministic"}
    result = run_from_dict(payload, assets_root=ROOT, output_dir=tmp_path / "loop")
    assert len(result.final_state.actions) == 2
    assert result.termination_reason.value == "max_steps"
    assert not result.final_state.truncated
