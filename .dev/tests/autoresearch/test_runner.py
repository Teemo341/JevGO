import json
import os
from pathlib import Path

import pytest

from jevgo.core import (
    BudgetLimits, EpisodeSpec, GenerationResponse, JudgeResponse, ModelIdentity,
    RunConfig, TerminationReason, Usage,
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
        texts = tuple("bad" if self.malformed else json.dumps({"action": {"name": names[i % 4]}}) for i in range(request.n))
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
        judge_candidates=False, judge_stop=False, budget=BudgetLimits(judge_calls=0)),
        DenseMiniGrid(), model, model, output_dir=tmp_path / "dense")
    events = [json.loads(line) for line in (tmp_path / "dense/events.jsonl").read_text().splitlines()]
    search = next(e["payload"]["result"] for e in events if e["event_type"] == "search")
    assert search["partial_simulations"] == 1
    visited = [edge for edge in search["rootstats"] if edge["N"]]
    assert len(visited) == 1
    assert visited[0]["N"] == visited[0]["partial_visits"] == 1
    assert visited[0]["W"] == visited[0]["Q"] == 2
    assert len(result.final_state.actions) == 1
