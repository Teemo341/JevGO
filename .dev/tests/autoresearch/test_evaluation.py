import json
from pathlib import Path

import pytest

from jevgo.core import Action, BudgetLedger, EpisodeSpec, EventRecord, JSONLEventStore, Observation, ResearchState, RunConfig, RunResult, TerminationReason, Usage
from jevgo.evaluation import audit_run, build_matrix, run_matrix, summarize, wilson


def spec():
    return {"episodes": [{"environment": "minigrid", "task": "MiniGrid-Empty-5x5-v0"}],
            "seeds": [0, 1], "models": [{"backend": "deterministic"}],
            "methods": ["B0", "B1", "B2", "B3"], "ablations": ["full", "no_stop_judge"]}


def artifact(path, payload, success=True):
    path.mkdir()
    events = JSONLEventStore(path / "events.jsonl")
    ledger = BudgetLedger(event_store=events)
    ledger.charge(Usage(env_steps=2, replay_steps=1), source="env", phase="replay")
    episode = EpisodeSpec.model_validate(payload["episode"])
    config = RunConfig.model_validate(payload["run"])
    state = ResearchState(episode=episode, observation=Observation(text="result"), success=success)
    result = RunResult(episode=episode, config=config, final_state=state,
        termination_reason=TerminationReason.SUCCESS if success else TerminationReason.BUDGET_EXHAUSTED,
        usage=ledger.used, metadata={"validation_level": "cpu_fixture"})
    (path / "ledger.json").write_text(json.dumps(ledger.summary()))
    (path / "result.json").write_text(result.model_dump_json())


def test_matrix_conditions_and_role_ablations():
    rows = build_matrix(spec())
    assert len(rows) == 10
    assert len({r["run_id"] for r in rows}) == 10
    assert all(r["payload"]["run"]["learning_mode"] == "none" for r in rows)
    assert sum(not r["payload"]["run"]["judge_stop"] for r in rows) == 2
    for seed in (0, 1):
        conditions = [r["payload"]["episode"] for r in rows if r["payload"]["episode"]["seed"] == seed]
        assert all(c == conditions[0] for c in conditions)


def test_duplicate_matrix_rejected():
    cfg = spec()
    cfg["models"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        build_matrix(cfg)


def test_dry_run_does_not_load_backends(tmp_path):
    cfg = spec()
    cfg["models"] = [{"backend": "jev_27b_vl"}]
    manifest = run_matrix(cfg, output_dir=tmp_path / "plan", assets_root=tmp_path)
    report = summarize(manifest)
    assert all(g["attempted"] == 0 and g["within_budget_successes"] == 0 for g in report["groups"])
    assert list((tmp_path / "plan").glob("*/result.json")) == []


def test_costs_recomputed_and_corruption_rejected(tmp_path):
    payload = build_matrix(spec())[0]["payload"]
    path = tmp_path / "run"
    artifact(path, payload)
    audited = audit_run(path)
    assert audited["usage"]["env_steps"] == 2
    assert audited["phase_usage"]["replay/search"]["replay_steps"] == 1
    ledger = json.loads((path / "ledger.json").read_text())
    ledger["used"]["env_steps"] = 1
    (path / "ledger.json").write_text(json.dumps(ledger))
    with pytest.raises(ValueError, match="usage"):
        audit_run(path)


def test_failures_and_skips_never_become_success(tmp_path):
    manifest_path = run_matrix(spec(), output_dir=tmp_path / "matrix", assets_root=tmp_path)
    manifest = json.loads(manifest_path.read_text())
    for i, entry in enumerate(manifest["runs"]):
        if i % 3 == 0:
            artifact(Path(entry["directory"]), entry["payload"], success=i == 0)
            entry["status"] = "finished"
        else:
            entry["status"] = "skipped" if i % 3 == 1 else "error"
    manifest_path.write_text(json.dumps(manifest))
    summary = summarize(manifest_path)
    assert sum(g["within_budget_successes"] for g in summary["groups"]) == 1
    assert all(g["validation_levels"] != ["pretrained_effect"] for g in summary["groups"])
    assert not summary["comparisons"][0]["all_runs_finished_with_known_cost"]
    assert summary["comparisons"][0]["same_task_conditions"]


def test_model_and_modality_groups_stay_separate(tmp_path):
    cfg = spec()
    cfg["models"] += [{"backend": "jev_9b"}]
    cfg["episodes"] += [{"environment": "minigrid", "task": "MiniGrid-Empty-5x5-v0", "options": {"modality": "vl"}}]
    manifest = run_matrix(cfg, output_dir=tmp_path / "matrix", assets_root=tmp_path)
    assert len(summarize(manifest)["comparisons"]) == 4


def test_wilson_is_defined_for_failures_and_empty():
    assert wilson(0, 0) is None
    assert wilson(0, 10)[0] == 0
    assert 0 < wilson(5, 10)[0] < .5 < wilson(5, 10)[1] < 1
