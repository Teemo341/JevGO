"""Real MiniGrid adapter acceptance, visibility, restoration and accounting."""
from dataclasses import replace

import numpy as np
import pytest
from minigrid.core.world_object import Key

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, BudgetLimits, EnvironmentAdapter,
    EnvironmentFailure, EnvironmentSnapshot, EpisodeSpec, UnsupportedCapability,
    Usage, UsagePhase,
)
from jevgo.envs.minigrid import MiniGridAdapter


def spec(**options):
    return EpisodeSpec(environment="minigrid", task="MiniGrid-Empty-5x5-v0",
                       seed=0, options=options)


@pytest.fixture
def adapter():
    adapter = MiniGridAdapter()
    yield adapter
    adapter.close()


def act(adapter, ledger, name, phase=UsagePhase.COMMIT):
    return adapter.step(Action(name=name), ledger=ledger, phase=phase)


def counts(ledger):
    return {k: v for k, v in ledger.used.model_dump().items() if k != "wall_seconds"}


def test_real_five_step_success_four_steps_not_success(adapter):
    ledger = BudgetLedger()
    initial = adapter.reset(spec(), ledger=ledger)
    assert isinstance(adapter, EnvironmentAdapter)
    assert adapter.capabilities.can_branch
    assert adapter.capabilities.details["replay_verified"] is True
    assert not initial.done
    assert adapter.observe() == initial.observation
    assert len(adapter._history) == 0
    assert adapter._env.unwrapped.step_count == 0
    assert ledger.used.env_steps == 2
    assert ledger.used.replay_steps == 1
    assert ledger.used.env_resets == 3
    assert ledger.used.env_snapshots == ledger.used.env_restores == 2
    assert ledger.used.env_observations == 5
    assert all(e.metadata["purpose"] == "capability_verification" for e in ledger.entries)
    for name in ("forward", "forward", "right", "forward"):
        result = act(adapter, ledger, name)
        assert result.action_success
        assert not result.success
        assert not result.done
    result = act(adapter, ledger, "forward")
    assert result.success and result.terminated
    assert result.reward == pytest.approx(0.955)
    assert ledger.used.env_steps == 7  # Five committed actions plus two probe steps.
    assert ledger.used.env_observations == 10
    assert len(adapter._history) == 5
    terminal = adapter.snapshot(ledger=ledger)
    restored = adapter.restore(terminal, ledger=ledger)
    assert restored == result
    with pytest.raises(EnvironmentFailure, match="finished"):
        act(adapter, ledger, "left")


def test_sibling_branches_are_isolated_and_replay_is_billed(adapter):
    ledger = BudgetLedger()
    initial = adapter.reset(spec(), ledger=ledger)
    root = adapter.snapshot(ledger=ledger)
    left = act(adapter, ledger, "left", UsagePhase.SEARCH)
    left_snapshot = adapter.snapshot(ledger=ledger)
    assert adapter.restore(root, ledger=ledger) == initial
    right = act(adapter, ledger, "right", UsagePhase.SEARCH)
    right_snapshot = adapter.snapshot(ledger=ledger)
    before = ledger.used
    assert adapter.restore(left_snapshot, ledger=ledger) == left
    after = ledger.used
    assert after.env_steps - before.env_steps == 1
    assert after.replay_steps - before.replay_steps == 1
    assert after.env_resets - before.env_resets == 1
    assert after.env_restores - before.env_restores == 1
    assert after.env_observations - before.env_observations == 2
    assert left.observation.data["direction"] == 3
    assert adapter.restore(right_snapshot, ledger=ledger) == right
    assert right.observation.data["direction"] == 1
    assert adapter.restore(root, ledger=ledger) == initial
    replay = [e for e in ledger.entries if e.usage.replay_steps and not e.metadata]
    assert replay and all(e.phase == UsagePhase.REPLAY and e.budget_phase == "search"
                          for e in replay)


def test_observation_only_contains_public_local_fields_and_observe_is_cached(adapter, monkeypatch):
    ledger = BudgetLedger()
    adapter.reset(spec(), ledger=ledger)
    env = adapter._env.unwrapped
    original = env.gen_obs

    def contaminated():
        raw = original()
        raw.update(agent_pos=[999, 999], global_map="SECRET_MAP", success_label="SECRET_LABEL",
                   rng="SECRET_RNG", carrying="SECRET_CARRYING")
        return raw

    monkeypatch.setattr(env, "gen_obs", contaminated)
    result = act(adapter, ledger, "left")
    data = result.observation.data
    assert set(data) == {"image", "direction", "mission"}
    assert np.asarray(data["image"]).shape == (7, 7, 3)
    assert np.array_equal(np.asarray(data["image"]), original()["image"])
    assert result.observation.image_paths == ()
    assert {a.name for a in result.observation.admissible_actions} == {
        "left", "right", "forward", "pickup", "drop", "toggle", "done"}
    serialized = result.observation.model_dump_json()
    assert "SECRET" not in serialized
    assert "agent_pos" not in serialized and "snapshot" not in serialized
    assert "Local view" in result.observation.text and "Mission:" in result.observation.text
    before = ledger.used
    monkeypatch.setattr(env, "gen_obs", lambda: pytest.fail("observe must not refresh"))
    assert adapter.observe() is result.observation
    assert ledger.used == before


def test_invalid_actions_do_not_imply_task_success(adapter):
    ledger = BudgetLedger()
    adapter.reset(spec(verify_replay=False), ledger=ledger)
    for name in ("pickup", "drop", "toggle", "done"):
        result = act(adapter, ledger, name)
        assert result.action_success is False and not result.success
    act(adapter, ledger, "left")
    blocked = act(adapter, ledger, "forward")
    assert blocked.action_success is False and not blocked.done
    before = counts(ledger)
    for action in (Action(name="teleport"), Action(name="left", arguments={"n": 2})):
        result = adapter.step(action, ledger=ledger)
        assert result.action_success is False and not result.success
        assert result.info["reason"]
    assert counts(ledger) == before
    assert ledger.used.env_steps == 6


def test_online_only_and_truncation(adapter):
    ledger = BudgetLedger()
    adapter.reset(spec(verify_replay=False, max_steps=4), ledger=ledger)
    assert not adapter.capabilities.can_branch
    assert ledger.used.env_resets == ledger.used.env_observations == 1
    assert ledger.used.env_steps == ledger.used.env_restores == 0
    with pytest.raises(UnsupportedCapability):
        adapter.snapshot(ledger=ledger)
    for name in ("forward", "forward", "right", "forward"):
        result = act(adapter, ledger, name)
    assert result.truncated and not result.success
    assert ledger.used.env_steps == 4


def test_verification_works_with_single_step_episode(adapter):
    ledger = BudgetLedger()
    initial = adapter.reset(spec(max_steps=1), ledger=ledger)
    assert not initial.done and adapter.capabilities.can_branch
    assert act(adapter, ledger, "forward").truncated


def test_invalid_foreign_forged_and_expired_snapshots(adapter):
    ledger = BudgetLedger()
    adapter.reset(spec(), ledger=ledger)
    root = adapter.snapshot(ledger=ledger)
    invalid = [object(), EnvironmentSnapshot(owner_id="foreign", payload=root.payload),
               replace(root, payload=replace(root.payload, seed=42))]
    for handle in invalid:
        before = ledger.used
        with pytest.raises(EnvironmentFailure, match="Invalid, foreign, or expired"):
            adapter.restore(handle, ledger=ledger)
        assert ledger.used == before
        assert adapter.capabilities.can_branch
    adapter.reset(spec(), ledger=ledger)
    with pytest.raises(EnvironmentFailure, match="expired"):
        adapter.restore(root, ledger=ledger)


@pytest.mark.parametrize("resource,limit", [
    ("env_steps", 1), ("env_resets", 2), ("env_observations", 4),
    ("env_snapshots", 1), ("env_restores", 1), ("replay_steps", 0),
])
def test_verification_budget_preflight_does_no_work_and_closes(adapter, resource, limit):
    adapter.reset(spec(verify_replay=False), ledger=BudgetLedger())
    ledger = BudgetLedger(BudgetLimits(**{resource: limit}))
    with pytest.raises(BudgetExceeded):
        adapter.reset(spec(), ledger=ledger)
    assert ledger.used == Usage()
    assert not adapter.capabilities.can_branch and adapter._env is None
    with pytest.raises(EnvironmentFailure):
        adapter.observe()


def test_step_exhaustion_and_wall_overage_prevent_further_work(adapter):
    ledger = BudgetLedger(BudgetLimits(env_steps=1))
    adapter.reset(spec(verify_replay=False), ledger=ledger)
    result = act(adapter, ledger, "forward")
    before = ledger.used
    with pytest.raises(BudgetExceeded):
        act(adapter, ledger, "forward")
    assert ledger.used == before and adapter.observe() == result.observation
    ledger.record(Usage(wall_seconds=1), source="fixture")
    ledger.limits = BudgetLimits(wall_seconds=0.5)
    with pytest.raises(BudgetExceeded, match="wall_seconds"):
        act(adapter, ledger, "right")
    assert adapter.observe() == result.observation


def test_replay_respects_search_reserve_and_commit_can_use_it(adapter):
    ledger = BudgetLedger(BudgetLimits(env_steps=4), commit_reserve=Usage(env_steps=1))
    adapter.reset(spec(), ledger=ledger)  # Two probe steps.
    result = act(adapter, ledger, "forward")  # One committed step.
    saved = adapter.snapshot(ledger=ledger)
    before = ledger.used
    with pytest.raises(BudgetExceeded):
        adapter.restore(saved, ledger=ledger, phase=UsagePhase.SEARCH)
    assert ledger.used == before and adapter.observe() == result.observation
    assert adapter.restore(saved, ledger=ledger, phase=UsagePhase.COMMIT) == result
    replay = [e for e in ledger.entries if e.usage.replay_steps and not e.metadata]
    assert len(replay) == 1 and replay[0].budget_phase == "commit"
    assert ledger.used.env_steps == 4


@pytest.mark.parametrize("corruption", ["carrying", "step_count", "rng"])
def test_hidden_state_divergence_is_detected_without_public_changes(adapter, monkeypatch, corruption):
    ledger = BudgetLedger()
    initial = adapter.reset(spec(), ledger=ledger)
    root = adapter.snapshot(ledger=ledger)
    env = adapter._env.unwrapped
    original = env.reset

    def corrupt_reset(**kwargs):
        result = original(**kwargs)
        if corruption == "carrying":
            env.carrying = Key("red")
        elif corruption == "step_count":
            env.step_count += 1
        else:
            env.np_random.random()
        assert adapter._observation(result[0]) == initial.observation
        return result

    monkeypatch.setattr(env, "reset", corrupt_reset)
    before = ledger.used
    with pytest.raises(EnvironmentFailure, match="branching disabled"):
        adapter.restore(root, ledger=ledger)
    assert not adapter.capabilities.can_branch
    assert adapter._env is None
    assert ledger.used.env_restores == before.env_restores + 1
    assert ledger.used.env_resets == before.env_resets + 1
    assert ledger.used.env_observations == before.env_observations + 1
    assert any(e.failed for e in ledger.entries)
    with pytest.raises(EnvironmentFailure):
        adapter.observe()


def test_replayed_action_drift_disables_branching(adapter, monkeypatch):
    ledger = BudgetLedger()
    adapter.reset(spec(), ledger=ledger)
    act(adapter, ledger, "forward")
    saved = adapter.snapshot(ledger=ledger)
    env = adapter._env.unwrapped
    original = env.step

    def drift(action):
        output = original(action)
        env.np_random.random()
        return output

    monkeypatch.setattr(env, "step", drift)
    before = ledger.used
    with pytest.raises(EnvironmentFailure, match="branching disabled"):
        adapter.restore(saved, ledger=ledger)
    assert ledger.used.replay_steps == before.replay_steps + 1
    assert ledger.used.env_steps == before.env_steps + 1
    assert not adapter.capabilities.can_branch


def test_task_selection_and_seeded_random_replay(adapter):
    ledger = BudgetLedger()
    episode = EpisodeSpec(environment="minigrid", task="MiniGrid-Empty-Random-5x5-v0", seed=19)
    initial = adapter.reset(episode, ledger=ledger)
    root = adapter.snapshot(ledger=ledger)
    act(adapter, ledger, "right")
    assert adapter.restore(root, ledger=ledger) == initial
    override = episode.model_copy(update={"options": {"env_id": "MiniGrid-Empty-5x5-v0"}})
    adapter.reset(override, ledger=ledger)
    assert adapter._env.spec.id == "MiniGrid-Empty-5x5-v0"
    assert adapter._seed == 19


def test_initial_verification_failure_never_claims_branching(adapter, monkeypatch):
    from jevgo.envs import minigrid as module

    original_make = module.gym.make

    def make(*args, **kwargs):
        wrapper = original_make(*args, **kwargs)
        env = wrapper.unwrapped
        original_reset = env.reset
        calls = 0

        def nondeterministic_reset(**kwargs):
            nonlocal calls
            output = original_reset(**kwargs)
            calls += 1
            if calls > 1:
                env.np_random.random()
            return output

        monkeypatch.setattr(env, "reset", nondeterministic_reset)
        return wrapper

    monkeypatch.setattr(module.gym, "make", make)
    ledger = BudgetLedger()
    with pytest.raises(EnvironmentFailure, match="branching disabled"):
        adapter.reset(spec(), ledger=ledger)
    assert adapter._env is None and not adapter.capabilities.can_branch
    assert ledger.used.env_resets == 2
    assert ledger.used.env_steps == 1
    assert ledger.used.env_observations == 3
    assert ledger.used.env_snapshots == 2
    assert ledger.used.env_restores == 1
    assert any(entry.failed for entry in ledger.entries)
