import os
from pathlib import Path

import pytest

from jevgo.core import (Action, BudgetExceeded, BudgetLedger, BudgetLimits, EnvironmentFailure,
                        EpisodeSpec, UnsupportedCapability, Usage, UsagePhase)
from jevgo.envs.alfworld import ALFWorldAdapter

ROOT = Path(os.environ.get("JEVGO_ASSETS_ROOT", Path(__file__).resolve().parents[3]))
GAME = ROOT / ".dev/fixtures/alfworld/game.tw-pddl"
PATH = ("go to dresser 2", "take keychain 1 from dresser 2", "go to armchair 1", "move keychain 1 to armchair 1")


def spec(**options):
    return EpisodeSpec(environment="alfworld", task=str(GAME), options=options)


@pytest.fixture
def world():
    env = ALFWorldAdapter(assets_root=ROOT)
    ledger = BudgetLedger()
    try:
        initial = env.reset(spec(), ledger=ledger)
        yield env, ledger, initial
    finally:
        env.close()


def test_real_task_four_steps_success_and_public_whitelist(world):
    env, ledger, initial = world
    assert not initial.success and env.capabilities.can_branch
    assert initial.observation.data == {}
    assert "_facts" not in initial.observation.model_dump_json()
    for i, action in enumerate(PATH):
        result = env.step(Action(name=action), ledger=ledger)
        assert result.success == (i == 3)
        assert result.action_success
    assert result.objective_score == 1 and result.terminated
    assert ledger.used.env_steps == 6  # 2 capability checks + 4 task actions.
    assert ledger.used.replay_steps == 1


def test_sibling_replay_and_monotonic_cost(world):
    env, ledger, initial = world
    root = env.snapshot(ledger=ledger)
    first = env.step(Action(name=PATH[0]), ledger=ledger)
    child = env.snapshot(ledger=ledger)
    env.restore(root, ledger=ledger)
    env.step(Action(name="go to armchair 1"), ledger=ledger)
    before = ledger.used
    restored = env.restore(child, ledger=ledger)
    assert restored == first
    assert ledger.used.env_steps == before.env_steps + 1
    assert ledger.used.replay_steps == before.replay_steps + 1
    assert env.restore(root, ledger=ledger) == initial


def test_invalid_command_still_costs_and_restores(world):
    env, ledger, _ = world
    before = ledger.used.env_steps
    result = env.step(Action(name="impossible command"), ledger=ledger)
    assert not result.action_success and not result.success
    assert ledger.used.env_steps == before + 1
    snapshot = env.snapshot(ledger=ledger)
    assert env.restore(snapshot, ledger=ledger) == result


def test_restore_preflight_preserves_world(world):
    env, ledger, _ = world
    env.step(Action(name=PATH[0]), ledger=ledger)
    child = env.snapshot(ledger=ledger)
    observation = env.observe()
    empty = BudgetLedger(BudgetLimits(env_steps=0))
    with pytest.raises(BudgetExceeded):
        env.restore(child, ledger=empty)
    assert env.observe() == observation and empty.used == Usage()


def test_unverified_is_online_and_missing_data_explicit():
    env = ALFWorldAdapter(assets_root=ROOT)
    try:
        env.reset(spec(verify_replay=False), ledger=BudgetLedger())
        assert not env.capabilities.can_branch
        with pytest.raises(UnsupportedCapability):
            env.snapshot(ledger=BudgetLedger())
        with pytest.raises(EnvironmentFailure, match="local ALFWorld"):
            env.reset(EpisodeSpec(environment="alfworld", task="missing.tw-pddl"), ledger=BudgetLedger())
    finally:
        env.close()


def test_max_steps_is_truncation_not_success():
    env = ALFWorldAdapter(assets_root=ROOT, max_steps=1)
    try:
        ledger = BudgetLedger()
        env.reset(spec(), ledger=ledger)
        result = env.step(Action(name=PATH[0]), ledger=ledger)
        assert result.truncated and not result.success and not result.terminated
    finally:
        env.close()
