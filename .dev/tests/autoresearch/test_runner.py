import json
import os
from pathlib import Path

import pytest

from jevgo.core import (
    BudgetLimits, EpisodeSpec, GenerationResponse, JudgeResponse, ModelIdentity,
    ResearchState, RunConfig, TerminationReason, Usage,
)
from jevgo.envs.minigrid import MiniGridAdapter
from jevgo.runner import run_episode


class Fixture:
    identity = ModelIdentity(backend="deterministic_fixture", model="script-v1")

    def __init__(self, *, malformed=False, fail_after_action=False):
        self.closed = 0
        self.malformed, self.fail_after_action = malformed, fail_after_action

    @staticmethod
    def task(request):
        return json.JSONDecoder().raw_decode((request.prompt if hasattr(request, "prompt") else request.context).split("Public state:\n", 1)[1])[0]

    def generate(self, request, *, ledger, phase):
        state = self.task(request)
        target = ("forward", "forward", "right", "forward", "forward")[min(len(state["actions"]), 4)]
        names = (target, "left", "right", "forward")
        texts = tuple("bad" if self.malformed else json.dumps({
            "action": {"name": names[i % 4], "arguments": {}},
            "hypothesis": "unverified fixture observation", "rationale": "fixture action",
        }) for i in range(request.n))
        usage = Usage(generation_calls=1, model_requests=1, input_tokens=1, output_tokens=len(texts))
        ledger.charge(usage, source="fixture", phase=phase)
        return GenerationResponse(texts=texts, identity=self.identity, usage=usage)

    def score(self, request, *, ledger, phase):
        from jevgo.core import ModelFailure
        state = self.task(request)
        usage = Usage(judge_calls=1, model_requests=1, input_tokens=1)
        ledger.charge(usage, source="fixture", phase=phase)
        if self.fail_after_action and state["actions"]:
            raise ModelFailure("fixture scoring failure after actual commit")
        target = ("forward", "forward", "right", "forward", "forward")[min(len(state["actions"]), 4)]
        scores = tuple(float(json.loads(item)["action"]["name"] == target) if request.role == "candidate" else 1.
                       for item in request.items)
        return JudgeResponse(scores=scores, identity=self.identity, usage=usage)

    def close(self):
        self.closed += 1


def episode(**options):
    return EpisodeSpec(episode_id="smoke-0", environment="minigrid", task="MiniGrid-Empty-5x5-v0", options=options)


def run(tmp_path, method, **changes):
    model = Fixture()
    result = run_episode(episode(), RunConfig(method=method, max_steps=8, **changes),
        MiniGridAdapter(), model, model, output_dir=tmp_path / method,
        metadata={"validation_level": "cpu_fixture"})
    assert model.closed == 1
    return result


@pytest.mark.parametrize("method", ["B0", "B1", "B2", "B3"])
def test_four_real_minigrid_methods(tmp_path, method):
    result = run(tmp_path, method)
    assert result.success and result.termination_reason == TerminationReason.SUCCESS
    assert len(result.final_state.actions) == 5
    assert (tmp_path / method / "ledger.json").is_file()
    events = [json.loads(line) for line in (tmp_path / method / "events.jsonl").read_text().splitlines()]
    commits = [e for e in events if e["event_type"] == "transition" and e["payload"]["phase"] == "commit"]
    assert len(commits) == 5
    assert all(e["branch_id"] == "root" for e in commits)
    if method in ("B0", "B2"):
        assert result.usage.judge_calls == 0
    if method in ("B2", "B3"):
        assert result.usage.replay_steps > 1
        assert any(e["event_type"] == "search" for e in events)
    assert all(e.branch_id == "root" for e in result.final_state.evidence)


def test_reproducible_actions_and_cost_counts(tmp_path):
    results = [run(tmp_path / str(i), "B3") for i in range(2)]
    assert results[0].final_state.actions == results[1].final_state.actions
    costs = [r.usage.model_dump(exclude={"wall_seconds"}) for r in results]
    assert costs[0] == costs[1]


def test_search_cannot_spend_last_commit_allowance(tmp_path):
    result = run(tmp_path, "B2", budget=BudgetLimits(env_steps=5))
    assert result.termination_reason == TerminationReason.BUDGET_EXHAUSTED
    assert result.final_state.actions  # A real action was submitted after interrupted search.
    assert result.usage.env_steps <= 5
    assert not result.success


def test_online_only_rejects_mcts_explicitly(tmp_path):
    model = Fixture()
    result = run_episode(episode(verify_replay=False), RunConfig(method="B2"), MiniGridAdapter(),
                         model, output_dir=tmp_path / "online")
    assert result.termination_reason == TerminationReason.ENVIRONMENT_FAILURE
    assert "MCTS requires" in result.error
    assert not result.final_state.actions and model.closed == 1


def test_invalid_proposals_do_not_execute_task_actions(tmp_path):
    model = Fixture(malformed=True)
    result = run_episode(episode(), RunConfig(method="B0"), MiniGridAdapter(), model,
                         output_dir=tmp_path / "invalid")
    assert result.termination_reason == TerminationReason.INVALID_CANDIDATES
    assert not result.final_state.actions
    assert result.usage.env_steps == 2  # Capability verification only.


def test_post_commit_model_error_keeps_observation(tmp_path):
    model = Fixture(fail_after_action=True)
    result = run_episode(episode(), RunConfig(method="B1"), MiniGridAdapter(), model, model,
                         output_dir=tmp_path / "error")
    assert result.termination_reason == TerminationReason.MODEL_FAILURE
    assert len(result.final_state.actions) == 1
    assert result.final_state.evidence[-1].origin == "environment"
    assert model.closed == 1


def test_existing_output_never_overwritten_and_resources_closed(tmp_path):
    model = Fixture()
    (tmp_path / "old").mkdir()
    with pytest.raises(FileExistsError):
        run_episode(episode(), RunConfig(), MiniGridAdapter(), model, model, output_dir=tmp_path / "old")
    assert model.closed == 1 and not list((tmp_path / "old").iterdir())


@pytest.mark.parametrize("method", ["B0", "B1"])
def test_sequential_methods_need_no_successor_snapshots(tmp_path, method):
    result = run(tmp_path, method, budget=BudgetLimits(env_steps=1000, env_snapshots=2))
    assert result.success and result.usage.env_snapshots == 2


def test_known_reward_is_backed_up_when_postprocessing_exhausts_budget(tmp_path):
    class DenseMiniGrid(MiniGridAdapter):
        def step(self, action, *, ledger, phase):
            result = super().step(action, ledger=ledger, phase=phase)
            return result.model_copy(update={"reward": 2.0})
    model = Fixture()
    result = run_episode(episode(), RunConfig(method="B3", simulations=1, rollout_depth=0,
        judge_candidates=False, judge_stop=False, budget=BudgetLimits(judge_calls=0),
        options={"memory_records": 1, "memory_pool_policy": "history_tail", "memory_pool_records": 4}),
        DenseMiniGrid(), model, model, output_dir=tmp_path / "dense")
    events = [json.loads(line) for line in (tmp_path / "dense/events.jsonl").read_text().splitlines()]
    search = next(e["payload"]["result"] for e in events if e["event_type"] == "search")
    assert search["partial_simulations"] == 1
    visited = [edge for edge in search["rootstats"] if edge["N"]]
    assert len(visited) == 1
    assert visited[0]["N"] == visited[0]["partial_visits"] == 1
    assert visited[0]["W"] == visited[0]["Q"] == 2
    assert len(result.final_state.actions) == 1


@pytest.mark.parametrize("raw", [
    '{"action":{"name":"forward","arguments":{}}}',
    '{"action":{"name":"forward","arguments":{}},"hypothesis":"x","hypothesis":"y","rationale":"r"}',
])
def test_strict_invalid_generated_samples_do_not_execute(tmp_path, raw):
    class InvalidFixture(Fixture):
        def generate(self, request, *, ledger, phase):
            usage = Usage(generation_calls=1, model_requests=1, output_tokens=7)
            ledger.charge(usage, source="invalid_fixture", phase=phase)
            return GenerationResponse(texts=(raw,) * request.n, identity=self.identity, usage=usage)

    result = run_episode(episode(), RunConfig(method="B0"), MiniGridAdapter(), InvalidFixture(),
                         output_dir=tmp_path / "strict-invalid")
    assert result.termination_reason == TerminationReason.INVALID_CANDIDATES
    assert result.final_state.actions == ()
    assert result.usage.env_steps == 2  # Existing capability check, no proposal execution.
    assert result.usage.generation_calls == result.usage.model_requests == 2
    assert result.usage.output_tokens == 14


def test_only_independent_valid_repair_executes_and_retains_source(tmp_path):
    class RepairedFixture(Fixture):
        def __init__(self):
            super().__init__()
            self.requests = []

        def generate(self, request, *, ledger, phase):
            self.requests.append(request)
            usage = Usage(generation_calls=1, model_requests=1, output_tokens=7)
            ledger.charge(usage, source="repaired_fixture", phase=phase)
            raw = ('{"action":{"name":"left","arguments":{}},"hypothesis":"bad",'
                   '"hypothesis":"duplicate","rationale":"bad"}') if len(self.requests) == 1 else json.dumps({
                       "action": {"name": "right", "arguments": {}},
                       "hypothesis": "unverified right turn", "rationale": "valid new sample"})
            return GenerationResponse(texts=(raw,), identity=self.identity, usage=usage)

    model = RepairedFixture()
    result = run_episode(episode(), RunConfig(method="B0", max_steps=1, options={"proposal_repair": "bounded"}),
                         MiniGridAdapter(), model, output_dir=tmp_path / "repaired")
    assert result.termination_reason == TerminationReason.MAX_STEPS
    assert [a.name for a in result.final_state.actions] == ["right"]
    assert result.usage.generation_calls == result.usage.model_requests == 2
    assert result.usage.env_steps == 3  # Two verification steps, one repaired action.
    events = [json.loads(line) for line in (tmp_path / "repaired/events.jsonl").read_text().splitlines()]
    candidate = next(e["payload"] for e in events if e["event_type"] == "candidates")
    assert candidate["candidate_sources"][0]["sources"] == [{"attempt": 1, "source": "repair", "sample_index": 0}]
    sample_events = [e["payload"] for e in events if e["event_type"] == "proposal_samples"]
    assert sample_events[0]["samples"][0]["error_stage"] == "json"
    assert sample_events[1]["samples"][0]["accepted"]


def test_repair_budget_interruption_is_not_an_environment_action(tmp_path):
    result = run_episode(episode(), RunConfig(method="B0", budget=BudgetLimits(generation_calls=1)),
                         MiniGridAdapter(), Fixture(malformed=True), output_dir=tmp_path / "repair-budget")
    assert result.termination_reason == TerminationReason.BUDGET_EXHAUSTED
    assert result.final_state.actions == () and not result.success
    assert result.usage.generation_calls == result.usage.model_requests == 1
    assert result.usage.env_steps == 2


@pytest.mark.parametrize("method", ["B0", "B1", "B2", "B3"])
@pytest.mark.parametrize("mode", ["e1", "e1_e2"])
def test_selected_observation_mode_follows_real_branch_traces_without_memory(tmp_path, method, mode):
    from jevgo.research import public_state
    model = Fixture()
    result = run_episode(episode(), RunConfig(method=method, max_steps=2, simulations=1, rollout_depth=0,
        use_memory=False, options={"public_observation": mode}), MiniGridAdapter(), model, model,
        output_dir=tmp_path / "observation-mode")
    assert result.termination_reason == TerminationReason.MAX_STEPS and len(result.final_state.actions) == 2
    rows = [json.loads(line) for line in (tmp_path / "observation-mode/events.jsonl").read_text().splitlines()]
    branches = {}
    seen_roles = set()
    available_feedback = 0
    for row in rows:
        if row["event_type"] in ("initial_state", "transition"):
            branches[row["branch_id"]] = ResearchState.model_validate(row["payload"]["state"])
        elif row["event_type"] in ("generation_request", "judge_request"):
            request = row["payload"]["request"]
            field = "prompt" if row["event_type"] == "generation_request" else "context"
            context = json.JSONDecoder().raw_decode(request[field].split("Public state:\n")[1])[0]
            expected = public_state(branches[row["branch_id"]], use_memory=False, public_observation=mode)
            assert context == expected and context["memory"] == []
            assert row["payload"]["public_observation"] == mode
            assert "data.image[x][y][channel]" in context["observation"]["text"]
            assert "action_success" not in json.dumps(context) and "agent_pos" not in json.dumps(context)
            if field == "context":
                seen_roles.add(request["role"])
            if mode == "e1_e2" and context["actions"]:
                feedback = context["observation_feedback"]
                assert feedback["status"] == "available"
                assert feedback["action"] == context["actions"][-1]
                available_feedback += 1
    if mode == "e1_e2":
        assert available_feedback
    if method in ("B1", "B3"):
        assert seen_roles == {"candidate", "stop"}  # no_memory disables persistent evidence selection.
    else:
        assert seen_roles == set()


@pytest.mark.parametrize("method", ["B0", "B2"])
def test_runner_records_actual_proposal_scopes_and_root_visited_edges(tmp_path, method):
    result = run(tmp_path, method)
    events = [json.loads(line) for line in Path(result.events_path).read_text().splitlines()]
    requests = [e["payload"] for e in events if e["event_type"] == "generation_request"]
    assert requests and all(r["source"] in ("initial", "repair", "coverage") for r in requests)
    if method == "B0":
        assert {r["proposal_scope"] for r in requests} == {"root"}
    else:
        assert {r["proposal_scope"] for r in requests} == {"root", "tree", "rollout"}
        for search in (e["payload"] for e in events if e["event_type"] == "search"):
            assert search["root_candidate_edges"] == len(search["result"]["rootstats"])
            assert search["root_visited_edges"] == sum(edge["N"] > 0 for edge in search["result"]["rootstats"])


@pytest.mark.parametrize("judge_stop", [False, True])
def test_stop_off_advisory_metadata_and_high_scores_do_not_create_success(tmp_path, judge_stop):
    model = Fixture()
    result = run_episode(episode(), RunConfig(method="B1", max_steps=3, judge_candidates=False,
        judge_evidence=False, judge_stop=judge_stop), MiniGridAdapter(), model, model,
        output_dir=tmp_path / "stop")
    assert result.termination_reason == TerminationReason.MAX_STEPS and not result.success
    assert len(result.final_state.actions) == 3
    semantics = result.metadata["policy_mechanisms"]["stop"]
    assert semantics["mode"] == ("advisory" if judge_stop else "off")
    assert semantics["calls_attempted"] == semantics["advice_records"] == (3 if judge_stop else 0)
    assert not semantics["controls_success"] and not semantics["controls_action_or_search"]
    assert result.usage.judge_calls == (3 if judge_stop else 0)
    assert result.metadata["policy_mechanisms"]["memory"]["scored_selections"] == 0


@pytest.mark.parametrize("judge_stop,steps,reason", [
    (False, 3, TerminationReason.MAX_STEPS),
    (True, 1, TerminationReason.BUDGET_EXHAUSTED),
])
def test_stop_budget_effect_is_indirect_and_keeps_already_committed_action(tmp_path, judge_stop, steps, reason):
    model = Fixture()
    result = run_episode(episode(), RunConfig(method="B1", max_steps=3, judge_candidates=False,
        judge_evidence=False, judge_stop=judge_stop, budget=BudgetLimits(judge_calls=0)),
        MiniGridAdapter(), model, model, output_dir=tmp_path / "stop-budget")
    assert result.termination_reason == reason and not result.success
    assert len(result.final_state.actions) == steps
    assert result.final_state.evidence[-1].origin == "environment"
    assert result.usage.judge_calls == 0  # Rejection before charge, not a free successful stop score.
    semantics = result.metadata["policy_mechanisms"]["stop"]
    assert semantics["cost_can_reduce_remaining_budget"] is judge_stop
    assert semantics["calls_attempted"] == int(judge_stop) and semantics["advice_records"] == 0


def test_stop_backend_failure_keeps_commit_observation_memory_and_charges(tmp_path):
    model = Fixture(fail_after_action=True)
    result = run_episode(episode(), RunConfig(method="B1", max_steps=3, judge_candidates=False,
        judge_evidence=False, judge_stop=True), MiniGridAdapter(), model, model,
        output_dir=tmp_path / "stop-error")
    assert result.termination_reason == TerminationReason.MODEL_FAILURE
    assert len(result.final_state.actions) == 1 and result.usage.judge_calls == 1
    assert result.final_state.evidence[-1].origin == "environment"
    assert result.final_state.evidence[-1].evidence_id in result.final_state.working_memory
    semantics = result.metadata["policy_mechanisms"]["stop"]
    assert semantics["backend_failure_can_terminate"]
    assert semantics["calls_attempted"] == 1 and semantics["advice_records"] == 0
    assert not result.success
