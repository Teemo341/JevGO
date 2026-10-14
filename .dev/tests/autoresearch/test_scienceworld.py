import os
from pathlib import Path

import pytest

from jevgo.core import Action, BudgetLedger, EpisodeSpec, UsagePhase
from jevgo.envs.scienceworld import ScienceWorldAdapter

ROOT = Path(os.environ.get("JEVGO_ASSETS_ROOT", Path(__file__).resolve().parents[3]))
# Test-only, previously verified fixed trajectory; never supplied to a model.
PATH = ("open door to kitchen", "go to kitchen", "open door to outside", "go to outside",
        "look around", "focus on blue jay", "pick up blue jay", "open door to kitchen",
        "go to kitchen", "move egg blue jay egg in inventory to red box")


def spec(**options):
    return EpisodeSpec(environment="scienceworld", task="find-living-thing",
                       options={"variation": 0, "simplifications": "easy", **options})


@pytest.fixture
def world():
    env, ledger = ScienceWorldAdapter(assets_root=ROOT), BudgetLedger()
    try:
        initial = env.reset(spec(), ledger=ledger)
        yield env, ledger, initial
    finally:
        env.close()
        assert env.last_process.poll() is not None


def test_actual_success_and_cost(world):
    env, ledger, initial = world
    assert not initial.success and initial.objective_score == 8
    assert ledger.used.env_steps == 5 and ledger.used.env_observations == 40
    assert ledger.used.replay_steps == 3 and env.capabilities.can_branch
    assert set(initial.observation.data) == {"look", "inv", "taskDesc"}
    for command in PATH:
        result = env.step(Action(name=command), ledger=ledger)
    assert result.success and result.objective_score == 100 and result.terminated
    assert ledger.used.env_steps == 15


def test_replay_prefix_and_cached_observe(world):
    env, ledger, initial = world
    root = env.snapshot(ledger=ledger)
    first = env.step(Action(name="go to kitchen"), ledger=ledger)
    child = env.snapshot(ledger=ledger)
    env.restore(root, ledger=ledger)
    env.step(Action(name="focus on picture"), ledger=ledger)
    before = ledger.used
    assert env.restore(child, ledger=ledger) == first
    assert ledger.used.env_steps == before.env_steps + 2  # reset's look + one prefix
    assert ledger.used.replay_steps == before.replay_steps + 2
    before = ledger.used
    assert env.observe() == env.observe() == first.observation
    assert before == ledger.used
    assert env.restore(root, ledger=ledger) == initial


def test_failure_done_does_not_mean_success(world):
    env, ledger, _ = world
    result = env.step(Action(name="focus on picture"), ledger=ledger)
    assert result.terminated and not result.success and not result.truncated
    assert result.objective_score < 0 and result.info["failure"]


def test_step_limit_is_truncation_and_process_exits():
    env, ledger = ScienceWorldAdapter(assets_root=ROOT, max_steps=1), BudgetLedger()
    try:
        env.reset(spec(verify_replay=False), ledger=ledger)
        result = env.step(Action(name="wait1"), ledger=ledger)
        assert result.truncated and not result.success and not result.terminated
        assert not env.capabilities.can_branch
    finally:
        env.close()
    assert env.last_process.poll() is not None
