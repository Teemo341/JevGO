import json
from pathlib import Path

import pytest

from jevgo.core import Action, BudgetLimits, BudgetLedger, EpisodeSpec, EventRecord, JSONLEventStore, Observation, ResearchState, RunConfig, RunResult, TerminationReason, Usage
from jevgo.evaluation import audit_run, build_matrix, run_matrix, summarize, wilson


def spec():
    return {"episodes": [{"environment": "minigrid", "task": "MiniGrid-Empty-5x5-v0"}],
            "seeds": [0, 1], "models": [{"backend": "deterministic"}],
            "methods": ["B0", "B1", "B2", "B3"], "ablations": ["full", "no_stop_judge"]}


def artifact(path, payload, success=True, *, judge_identities=(), judge_calls=None, validation_level="cpu_fixture"):
    path.mkdir()
    events = JSONLEventStore(path / "events.jsonl")
    ledger = BudgetLedger(BudgetLimits.model_validate(payload["run"]["budget"]), event_store=events)
    ledger.record(Usage(env_steps=2, replay_steps=1), source="env", phase="replay")
    calls = len(judge_identities) if judge_calls is None else judge_calls
    if calls:
        ledger.record(Usage(judge_calls=calls, model_requests=calls), source="fixture.judge")
    for identity in judge_identities:
        payload_identity = {} if identity is None else {"identity": identity}
        events.append(EventRecord(event_type="judge", payload={"role": "candidate", **payload_identity}))
    episode = EpisodeSpec.model_validate(payload["episode"])
    config = RunConfig.model_validate(payload["run"])
    state = ResearchState(episode=episode, observation=Observation(text="result"), success=success)
    result = RunResult(episode=episode, config=config, final_state=state,
        termination_reason=TerminationReason.SUCCESS if success else TerminationReason.BUDGET_EXHAUSTED,
        usage=ledger.used, metadata={"validation_level": validation_level, "generator": {"backend": "fixture", "model": "fixture", "revision": "v1", "adapter": None}})
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


def test_overage_is_recomputed_and_unknown_cost_cannot_count_within_budget(tmp_path):
    payload = build_matrix(spec())[0]["payload"]
    payload["run"]["budget"]["env_steps"] = 1
    path = tmp_path / "overage"
    artifact(path, payload)
    ledger = json.loads((path / "ledger.json").read_text())
    ledger["limits"] = payload["run"]["budget"]
    ledger["over_budget"] = []  # Stale bookkeeping cannot defeat the audit.
    (path / "ledger.json").write_text(json.dumps(ledger))
    audited = audit_run(path)
    assert audited["result"].success
    assert not audited["within_budget"]
    assert audited["recomputed_over_budget"] == ["env_steps"]
    assert not audited["declared_over_budget_consistent"]

    payload = build_matrix(spec())[0]["payload"]
    path = tmp_path / "unknown"
    artifact(path, payload)
    ledger = json.loads((path / "ledger.json").read_text())
    ledger["entries"][0]["metadata"] = {"unknown_usage": ["input_tokens"], "usage_complete": False}
    (path / "ledger.json").write_text(json.dumps(ledger))
    events = [json.loads(x) for x in (path / "events.jsonl").read_text().splitlines()]
    events[0]["payload"]["metadata"] = ledger["entries"][0]["metadata"]
    (path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    audited = audit_run(path)
    assert audited["result"].success and not audited["cost_complete"]
    assert not audited["within_budget"]


def test_actual_backbone_revision_mismatch_splits_comparisons(tmp_path):
    cfg = spec()
    cfg.update(methods=["B0", "B1"], seeds=[0], ablations=["full"])
    manifest_path = run_matrix(cfg, output_dir=tmp_path / "identities", assets_root=tmp_path)
    manifest = json.loads(manifest_path.read_text())
    for index, entry in enumerate(manifest["runs"]):
        path = Path(entry["directory"])
        artifact(path, entry["payload"])
        entry["status"] = "finished"
        result = json.loads((path / "result.json").read_text())
        result["metadata"]["generator"]["revision"] = "revision-" + str(index)
        (path / "result.json").write_text(json.dumps(result))
    manifest_path.write_text(json.dumps(manifest))
    report = summarize(manifest_path)
    assert len(report["comparisons"]) == 2
    assert not any(c["fair_comparison_eligible"] for c in report["comparisons"])


@pytest.mark.parametrize("wall_seconds,eligible", [(None, True), (0.0, False), (60.0, False)])
def test_finite_instrumented_wall_budget_excludes_fair_comparison(tmp_path, wall_seconds, eligible):
    cfg = spec()
    cfg.update(methods=["B0", "B3"], seeds=[0], ablations=["full"],
               run={"budget": {"wall_seconds": wall_seconds}})
    manifest_path = run_matrix(cfg, output_dir=tmp_path / "wall-budget", assets_root=tmp_path)
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["runs"]:
        artifact(Path(entry["directory"]), entry["payload"])
        entry["status"] = "finished"
    manifest_path.write_text(json.dumps(manifest))

    report = summarize(manifest_path)
    assert len(report["comparisons"]) == 1
    comparison = report["comparisons"][0]
    assert comparison["methods"] == ["B0/full", "B3/full"]
    assert comparison["condition"]["budget"]["wall_seconds"] == wall_seconds
    assert comparison["same_task_conditions"]
    assert comparison["consistent_observed_generator"]
    assert comparison["consistent_active_judge"]
    assert comparison["all_runs_finished_with_known_cost"]
    assert comparison["finite_wall_seconds_budget"] is (wall_seconds is not None)
    assert comparison["fair_comparison_eligible"] is eligible
    assert "not end-to-end elapsed time" in report["interpretation"]
    assert "coverage differs between B0/B1 and MCTS" in report["interpretation"]


AUTOTRUST_JUDGE = {"backend": "autotrust", "model": "autotrust/JEV-9B", "revision": "checkpoint-v1", "adapter": "jev-decision"}


def judged_matrix(tmp_path, *, identities, calls=None, validation_level="real_model_attempt"):
    cfg = spec()
    cfg.update(methods=["B0", "B3"], seeds=[0], ablations=["full"])
    path = run_matrix(cfg, output_dir=tmp_path / "judged", assets_root=tmp_path)
    manifest = json.loads(path.read_text())
    for entry in manifest["runs"]:
        active = entry["payload"]["run"]["method"] == "B3"
        directory = Path(entry["directory"])
        artifact(directory, entry["payload"], judge_identities=identities if active else (),
                 judge_calls=calls if active else 0, validation_level=validation_level)
        # Shared-backend metadata is intentionally stale: scoring events must win.
        result = json.loads((directory / "result.json").read_text())
        result["metadata"]["judge"] = {**AUTOTRUST_JUDGE, "adapter": None}
        (directory / "result.json").write_text(json.dumps(result))
        entry["status"] = "finished"
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize("identity,validation_level", [
    (AUTOTRUST_JUDGE, "real_model_attempt"),
    ({"backend": "semif", "model": "Qwen/Qwen3.5-4B", "revision": "model-v2", "adapter": None}, "real_model_attempt"),
    ({"backend": "deterministic_fixture", "model": "script-v1", "revision": None, "adapter": None}, "cpu_fixture"),
])
def test_scoring_response_identity_overrides_shared_backend_metadata(tmp_path, identity, validation_level):
    path = judged_matrix(tmp_path, identities=[identity], validation_level=validation_level)
    report = summarize(path)
    groups = {group["method"]: group for group in report["groups"]}
    assert groups["B0"]["observed_judge_identities"] == []
    assert groups["B0"]["rows"][0]["judge_identity"] is None
    assert groups["B3"]["observed_judge_identities"] == [identity]
    assert groups["B3"]["rows"][0]["judge_identity"] == identity
    assert all(group["judge_identity_complete_runs"] == 1 for group in groups.values())
    comparison = report["comparisons"][0]
    assert comparison["all_active_judge_identities_known"]
    assert comparison["consistent_active_judge"]
    assert comparison["fair_comparison_eligible"]


@pytest.mark.parametrize("identities,calls", [
    ([], 1),  # A failed call has no scoring response identity.
    ([None], 1),
    ([{key: value for key, value in AUTOTRUST_JUDGE.items() if key != "adapter"}], 1),
    ([{key: value for key, value in AUTOTRUST_JUDGE.items() if key != "revision"}], 1),
    ([{**AUTOTRUST_JUDGE, "revision": None}], 1),
    ([{**AUTOTRUST_JUDGE, "model": ""}], 1),
    ([AUTOTRUST_JUDGE], 2),
    ([AUTOTRUST_JUDGE], 0),
])
def test_missing_or_unaccounted_judge_identity_excludes_fair_comparison(tmp_path, identities, calls):
    report = summarize(judged_matrix(tmp_path, identities=identities, calls=calls))
    group = next(group for group in report["groups"] if group["method"] == "B3")
    assert group["judge_identity_complete_runs"] == 0
    assert group["rows"][0]["judge_identity_issues"]
    comparison = report["comparisons"][0]
    assert comparison["all_runs_finished_with_known_cost"]
    assert not comparison["all_active_judge_identities_known"]
    assert not comparison["consistent_active_judge"]
    assert not comparison["fair_comparison_eligible"]


def test_mixed_scoring_adapters_remain_visible_and_ineligible(tmp_path):
    identities = [AUTOTRUST_JUDGE, {**AUTOTRUST_JUDGE, "adapter": "other-decision"}]
    report = summarize(judged_matrix(tmp_path, identities=identities))
    group = next(group for group in report["groups"] if group["method"] == "B3")
    assert group["observed_judge_identities"] == sorted(identities, key=lambda item: json.dumps(item, sort_keys=True))
    assert group["rows"][0]["judge_identity"] is None
    assert report["comparisons"][0]["all_active_judge_identities_known"]
    assert not report["comparisons"][0]["consistent_active_judge"]
    assert not report["comparisons"][0]["fair_comparison_eligible"]
