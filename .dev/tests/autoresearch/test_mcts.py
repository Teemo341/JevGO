"""Deterministic ground-truth domains: PUCT math, rollouts and budget boundaries."""
import json

import pytest

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, BudgetLimits, Candidate,
    EnvironmentCapabilities, EnvironmentSnapshot, EpisodeSpec, Observation,
    ResearchState, RunConfig, SearchState, SearchTransition, UnsupportedCapability,
    Usage, UsagePhase, ValueEstimate,
)
from jevgo.search.mcts import MCTS, MCTSResult


class GraphDomain:
    """A recoverable test environment; transition alone owns all environment costs."""

    def __init__(self, graph, *, terminal=(), values=None, ledger=None, priors=None,
                 candidate_cost=None, evaluation_cost=None):
        self.graph = graph
        self.terminal = set(terminal)
        self.values = values or {}
        self.ledger = ledger or BudgetLedger()
        self.priors = priors or {}
        self.candidate_cost = candidate_cost or Usage()
        self.evaluation_cost = evaluation_cost or Usage()
        self.capabilities = EnvironmentCapabilities(snapshot_restore=True)
        self.episode = EpisodeSpec(environment="deterministic-test-graph", task="reach terminal")
        self.calls = []
        self.current = "root"

    def state(self, label="root", parent=None, action=None, reward=0):
        if parent is None:
            research = ResearchState(episode=self.episode, observation=Observation(text=label),
                                     terminated=label in self.terminal)
        else:
            research = parent.research.evolve(
                observation=Observation(text=label), actions=parent.research.actions + (action,),
                last_reward=reward, terminated=label in self.terminal)
        return SearchState(research, EnvironmentSnapshot(owner_id="test", payload=label))

    def candidates(self, state, *, phase=UsagePhase.SEARCH):
        label = state.research.observation.text
        self.calls.append(("candidates", label, phase))
        self.ledger.charge(self.candidate_cost, source="test.candidates", phase=phase)
        return tuple(Candidate(action=Action(name=action),
                               prior=self.priors.get((label, action), 1))
                     for action in self.graph.get(label, {}))

    def transition(self, state, candidate, *, phase=UsagePhase.SEARCH):
        label = state.research.observation.text
        self.calls.append(("transition", label, candidate.action.name, phase))
        assert state.snapshot.owner_id == "test" and state.snapshot.payload == label
        self.ledger.charge(Usage(env_restores=1, env_steps=1, env_observations=1,
                                 env_snapshots=1), source="test.transition", phase=phase)
        self.current = label  # Restore this exact parent before executing.
        next_label, reward = self.graph[label][candidate.action.name]
        self.current = next_label
        return SearchTransition(self.state(next_label, state, candidate.action, reward), reward)

    def evaluate(self, state, *, phase=UsagePhase.SEARCH):
        label = state.research.observation.text
        assert not state.terminal, "Terminal rewards must never be evaluated again"
        self.calls.append(("evaluate", label, phase))
        self.ledger.charge(self.evaluation_cost, source="test.evaluate", phase=phase)
        return ValueEstimate(value=self.values.get(label, 0), source="heuristic")


def run(domain, **config):
    return MCTS(RunConfig(**config)).search(domain, domain.state())


def edge_stats(result):
    return {item.candidate.action.name: item for item in result.rootstats}


def transitions(domain):
    return [call for call in domain.calls if call[0] == "transition"]


def test_hand_calculated_puct_counts_and_terminal_reward_once():
    domain = GraphDomain({"root": {"a": ("ta", 1), "b": ("tb", 0)}}, terminal=("ta", "tb"))
    result = run(domain, simulations=4, c_puct=2, rollout_depth=2)
    a, b = result.rootstats
    assert (a.P, a.N, a.W, a.Q) == (0.5, 3, 3, 1)
    assert (b.P, b.N, b.W, b.Q) == (0.5, 1, 0, 0)
    assert result.candidate.action.name == "a"
    assert result.simulations == result.completed_simulations == 4
    assert result.partial_simulations == 0 and result.stop_reason == "simulation_limit"
    assert len(transitions(domain)) == domain.ledger.used.env_steps == 2
    assert domain.ledger.used.env_restores == domain.ledger.used.env_snapshots == 2
    assert domain.ledger.used.rollouts == 0
    assert not any(call[0] == "evaluate" for call in domain.calls)
    assert all(call[-1] == UsagePhase.SEARCH for call in domain.calls)


@pytest.mark.parametrize("gamma,expected", [(0, 2), (0.5, 4.75), (1, 10)])
def test_real_rollout_discount_is_applied_once(gamma, expected):
    domain = GraphDomain({"root": {"a": ("a", 2)}, "a": {"b": ("b", 3)},
                          "b": {"finish": ("end", 5)}}, terminal=("end",))
    result = run(domain, simulations=1, max_depth=1, rollout_depth=2, gamma=gamma)
    assert result.rootstats[0].W == expected
    assert result.rootstats[0].N == 1
    assert domain.ledger.used.env_steps == 3
    assert domain.ledger.used.rollouts == 1
    assert [call[1] for call in transitions(domain)] == ["root", "a", "b"]
    assert not any(call[0] == "evaluate" for call in domain.calls)


def test_signed_rewards_never_alternate_player_sign():
    domain = GraphDomain({"root": {"a": ("a", -2)}, "a": {"b": ("b", -3)},
                          "b": {"finish": ("end", 5)}}, terminal=("end",))
    result = run(domain, simulations=1, max_depth=1, rollout_depth=2, gamma=0.5)
    assert result.rootstats[0].Q == -2.25


def test_leaf_value_is_extra_future_return_only():
    domain = GraphDomain({"root": {"a": ("a", 2)}}, values={"a": 4})
    result = run(domain, simulations=1, rollout_depth=0, gamma=0.5)
    assert result.rootstats[0].Q == 4
    assert ("evaluate", "a", UsagePhase.SEARCH) in domain.calls
    assert domain.ledger.used.env_steps == 1 and domain.ledger.used.rollouts == 0


def test_tree_depth_and_extra_rollout_depth_are_independent():
    domain = GraphDomain({"root": {"a": ("a", 1)}, "a": {"b": ("b", 2)},
                          "b": {"finish": ("end", 3)}},
                         terminal=("end",), values={"b": 10})
    result = run(domain, simulations=3, max_depth=2, rollout_depth=1, gamma=0.5)
    # Simulation 1: 1 + .5*(2 + .5*10) = 4.5.
    # Simulations 2/3: two tree edges + one rollout => 1 + .5*(2 + .5*3) = 2.75.
    assert result.rootstats[0].N == 3
    assert result.rootstats[0].W == 10
    assert result.rootstats[0].Q == pytest.approx(10 / 3)
    assert domain.ledger.used.env_steps == 5
    assert domain.ledger.used.rollouts == 3
    assert [call[1] for call in transitions(domain)] == ["root", "a", "a", "b", "b"]


def test_search_discovers_delayed_reward_beyond_greedy_first_action():
    domain = GraphDomain({"root": {"small": ("small_end", 1), "delayed": ("b", 0)},
                          "b": {"continue": ("c", 0)}, "c": {"finish": ("large_end", 6)}},
                         terminal=("small_end", "large_end"),
                         priors={("root", "small"): 0.6, ("root", "delayed"): 0.4})
    result = run(domain, simulations=12, c_puct=2, max_depth=3, rollout_depth=2)
    stats = edge_stats(result)
    assert result.candidate.action.name == "delayed"
    assert stats["delayed"].Q == 6 and stats["delayed"].N > stats["small"].N
    assert domain.ledger.used.rollouts > 0
    assert any(call[1] == "c" for call in transitions(domain))


def test_duplicate_actions_merge_prior_and_counts_in_stable_order():
    class DuplicateDomain(GraphDomain):
        def candidates(self, state, *, phase=UsagePhase.SEARCH):
            super().candidates(state, phase=phase)
            return (Candidate(candidate_id="first", action=Action(name="a"), prior=0.2),
                    Candidate(candidate_id="duplicate", action=Action(name="a"), prior=0.3,
                              sample_count=2),
                    Candidate(candidate_id="second", action=Action(name="b"), prior=0.5))

    domain = DuplicateDomain({})
    result = run(domain, simulations=0)
    assert [item.candidate.candidate_id for item in result.rootstats] == ["first", "second"]
    assert [item.P for item in result.rootstats] == [0.5, 0.5]
    assert result.rootstats[0].candidate.sample_count == 3
    assert result.candidate.candidate_id == "first"
    assert result.simulations == 0 and domain.ledger.used.env_steps == 0
    encoded = result.model_dump_json()
    assert "snapshot" not in encoded and "owner_id" not in encoded
    assert MCTSResult.model_validate_json(encoded) == result
    assert json.loads(encoded)["rootstats"][0]["P"] == 0.5


def test_zero_priors_are_uniform_without_dropping_actions():
    domain = GraphDomain({"root": {"a": ("a", 0), "b": ("b", 0)}},
                         priors={("root", "a"): 0, ("root", "b"): 0})
    result = run(domain, simulations=0)
    assert [item.P for item in result.rootstats] == [0.5, 0.5]
    assert len(result.rootstats) == 2


@pytest.mark.parametrize("simulations,c_puct,rewards,selected", [
    (4, 2, (1, 10), "a"),  # N=3 vs N=1; visits take precedence over Q.
    (2, 100, (1, 10), "b"),  # Equal N; larger Q wins.
    (2, 100, (1, 1), "a"),  # Equal N/Q; stable input order wins.
])
def test_final_selection_uses_visits_then_value_then_stable_order(simulations, c_puct, rewards, selected):
    domain = GraphDomain({"root": {"a": ("ta", rewards[0]), "b": ("tb", rewards[1])}},
                         terminal=("ta", "tb"))
    result = run(domain, simulations=simulations, c_puct=c_puct)
    assert result.candidate.action.name == selected


def test_partial_rollout_retains_costs_and_commit_reserve():
    ledger = BudgetLedger(BudgetLimits(env_steps=3), commit_reserve=Usage(env_steps=1))
    domain = GraphDomain({"root": {"a": ("a", 1)}, "a": {"b": ("b", 2)},
                          "b": {"finish": ("end", 4)}}, terminal=("end",), ledger=ledger)
    root = domain.state()
    result = MCTS(RunConfig(simulations=8, max_depth=1, rollout_depth=3, gamma=0.5)).search(domain, root)
    assert result.stop_reason == "budget_exhausted"
    assert result.budget_resources == ("env_steps",)
    assert result.simulations == result.partial_simulations == 1
    assert result.completed_simulations == 0
    assert (result.rootstats[0].N, result.rootstats[0].W, result.rootstats[0].partial_visits) == (1, 2, 1)
    assert ledger.used.env_steps == ledger.used.env_restores == ledger.used.env_snapshots == 2
    assert ledger.used.rollouts == 1
    assert root.research.actions == ()
    committed = domain.transition(root, result.candidate, phase=UsagePhase.COMMIT)
    assert committed.state.research.observation.text == "a"
    assert len(committed.state.research.actions) == 1
    assert ledger.used.env_steps == 3


def test_budget_rejection_before_any_transition_does_not_fabricate_visit():
    ledger = BudgetLedger(BudgetLimits(env_steps=0))
    domain = GraphDomain({"root": {"a": ("end", 7)}}, terminal=("end",), ledger=ledger)
    result = run(domain, simulations=4)
    assert result.stop_reason == "budget_exhausted"
    assert result.candidate.action.name == "a"
    assert result.simulations == result.partial_simulations == 1
    assert result.completed_simulations == 0
    assert result.rootstats[0].N == result.rootstats[0].W == 0
    assert ledger.used.env_steps == ledger.used.rollouts == 0


def test_budget_failure_in_evaluation_marks_only_known_reward_partial():
    ledger = BudgetLedger(BudgetLimits(judge_calls=0))
    domain = GraphDomain({"root": {"a": ("a", 2)}}, values={"a": 100}, ledger=ledger,
                         evaluation_cost=Usage(judge_calls=1))
    result = run(domain, simulations=4, rollout_depth=0)
    assert result.stop_reason == "budget_exhausted" and result.partial_simulations == 1
    assert result.rootstats[0].Q == 2 and result.rootstats[0].partial_visits == 1
    assert ledger.used.judge_calls == 0 and ledger.used.env_steps == 1


def test_root_candidate_budget_failure_returns_no_candidate():
    ledger = BudgetLedger(BudgetLimits(generation_calls=0))
    domain = GraphDomain({"root": {"a": ("end", 1)}}, ledger=ledger,
                         candidate_cost=Usage(generation_calls=1))
    result = run(domain)
    assert result.candidate is None and result.rootstats == ()
    assert result.stop_reason == "budget_exhausted" and result.simulations == 0
    assert ledger.used.generation_calls == ledger.used.env_steps == 0


def test_rollout_budget_zero_and_step_rejection_do_not_count_unstarted_rollouts():
    for limits in (BudgetLimits(rollouts=0), BudgetLimits(env_steps=1)):
        ledger = BudgetLedger(limits)
        domain = GraphDomain({"root": {"a": ("a", 1)}, "a": {"finish": ("end", 7)}},
                             terminal=("end",), ledger=ledger)
        result = run(domain, simulations=1, rollout_depth=2)
        assert result.stop_reason == "budget_exhausted"
        assert result.rootstats[0].Q == 1 and result.rootstats[0].partial_visits == 1
        assert ledger.used.env_steps == 1 and ledger.used.rollouts == 0


def test_rollout_action_followed_by_snapshot_failure_still_counts():
    class SnapshotFailureDomain(GraphDomain):
        def transition(self, state, candidate, *, phase=UsagePhase.SEARCH):
            if state.research.observation.text == "a":
                self.ledger.charge(Usage(env_steps=1), source="test.failed_snapshot", phase=phase)
                raise BudgetExceeded(("env_snapshots",), phase)
            return super().transition(state, candidate, phase=phase)

    domain = SnapshotFailureDomain({"root": {"a": ("a", 1)}, "a": {"finish": ("end", 99)}})
    result = run(domain, simulations=2, rollout_depth=1)
    assert result.partial_simulations == 1 and result.rootstats[0].Q == 1
    assert domain.ledger.used.env_steps == 2 and domain.ledger.used.rollouts == 1
    assert any(entry.source == "mcts.rollout" and entry.failed for entry in domain.ledger.entries)


def test_cached_path_budget_interruption_updates_known_edges_once():
    ledger = BudgetLedger(BudgetLimits(env_steps=1))
    domain = GraphDomain({"root": {"a": ("a", 1)}, "a": {"b": ("b", 2)}},
                         ledger=ledger, values={"a": 10})
    result = run(domain, simulations=3, max_depth=2, rollout_depth=0, gamma=0.5)
    assert result.simulations == 2 and result.completed_simulations == result.partial_simulations == 1
    assert result.rootstats[0].N == 2
    assert result.rootstats[0].W == 7  # First estimate 6; partial second traversal 1.
    assert result.rootstats[0].partial_visits == 1
    assert ledger.used.env_steps == 1


def test_domain_costs_are_never_recharged_by_search():
    domain = GraphDomain({"root": {"a": ("a", 1)}, "a": {"finish": ("end", 2)}},
                         terminal=("end",), candidate_cost=Usage(generation_calls=1, input_tokens=3),
                         evaluation_cost=Usage(judge_calls=1))
    run(domain, simulations=1, rollout_depth=1)
    assert domain.ledger.used.env_steps == 2
    assert domain.ledger.used.generation_calls == 2 and domain.ledger.used.input_tokens == 6
    assert domain.ledger.used.judge_calls == 0
    for entry in domain.ledger.entries:
        if entry.source.startswith("mcts."):
            data = entry.usage.model_dump()
            assert all(value == 0 for name, value in data.items() if name not in ("rollouts", "wall_seconds"))


def test_terminal_and_no_candidates_stop_without_simulation():
    terminal = GraphDomain({}, terminal=("root",))
    result = run(terminal)
    assert result.stop_reason == "terminal" and result.candidate is None
    assert terminal.calls == [] and terminal.ledger.used == Usage()
    empty = GraphDomain({})
    result = run(empty)
    assert result.stop_reason == "no_candidates" and result.candidate is None
    assert result.simulations == empty.ledger.used.rollouts == 0
    assert [call[0] for call in empty.calls] == ["candidates"]


def test_online_only_domain_is_rejected_before_calls_or_charges():
    domain = GraphDomain({"root": {"a": ("end", 1)}})
    domain.capabilities = EnvironmentCapabilities()
    with pytest.raises(UnsupportedCapability):
        run(domain)
    assert domain.calls == [] and domain.ledger.used == Usage()


def test_wall_budget_zero_prevents_domain_calls():
    domain = GraphDomain({"root": {"a": ("end", 1)}},
                         ledger=BudgetLedger(BudgetLimits(wall_seconds=0)))
    result = run(domain)
    assert result.stop_reason == "budget_exhausted" and result.budget_resources == ("wall_seconds",)
    assert domain.calls == [] and domain.ledger.used.env_steps == 0


def test_search_calls_do_not_reuse_prior_tree_statistics():
    domain = GraphDomain({"root": {"a": ("end", 1)}}, terminal=("end",))
    engine = MCTS(RunConfig(simulations=1))
    root = domain.state()
    first = engine.search(domain, root)
    second = engine.search(domain, root)
    assert first.rootstats[0].model_dump(exclude={"candidate"}) == second.rootstats[0].model_dump(exclude={"candidate"})
    assert first.candidate.action == second.candidate.action
    assert domain.ledger.used.env_steps == 2


def test_actual_wall_time_overage_stops_before_another_domain_call():
    class SlowReportedDomain(GraphDomain):
        def transition(self, state, candidate, *, phase=UsagePhase.SEARCH):
            result = super().transition(state, candidate, phase=phase)
            self.ledger.record(Usage(wall_seconds=10), source="test.measured_work", phase=phase)
            return result

    ledger = BudgetLedger(BudgetLimits(wall_seconds=1))
    domain = SlowReportedDomain({"root": {"a": ("end", 1)}}, terminal=("end",), ledger=ledger)
    result = run(domain, simulations=5)
    assert result.stop_reason == "budget_exhausted" and result.budget_resources == ("wall_seconds",)
    assert result.completed_simulations == 1 and result.partial_simulations == 0
    assert result.rootstats[0].Q == 1  # A returned terminal result is complete.
    assert len(transitions(domain)) == 1
    assert ledger.used.wall_seconds >= 10


def test_action_arguments_participate_in_deduplication():
    class ArgumentDomain(GraphDomain):
        def candidates(self, state, *, phase=UsagePhase.SEARCH):
            return (Candidate(action=Action(name="move", arguments={"x": 1, "y": 2}), prior=0.2),
                    Candidate(action=Action(name="move", arguments={"y": 2, "x": 1}), prior=0.3),
                    Candidate(action=Action(name="move", arguments={"x": 2, "y": 1}), prior=0.5))

    result = run(ArgumentDomain({}), simulations=0)
    assert len(result.rootstats) == 2
    assert [item.P for item in result.rootstats] == [0.5, 0.5]
    assert result.rootstats[0].candidate.sample_count == 2


def test_partial_backup_updates_every_completed_path_edge_once(monkeypatch):
    ledger = BudgetLedger(BudgetLimits(env_steps=2))
    domain = GraphDomain({"root": {"a": ("a", 1)}, "a": {"b": ("b", 2)},
                          "b": {"finish": ("end", 3)}}, terminal=("end",), ledger=ledger)
    engine = MCTS(RunConfig(simulations=4, max_depth=3, rollout_depth=0, gamma=0.5))
    original = engine._backup
    partial_stats = []

    def inspect_backup(path, rollout_rewards, tail, *, partial):
        original(path, rollout_rewards, tail, partial=partial)
        if partial:
            partial_stats.extend((edge.N, edge.W, edge.partial_visits) for edge in path)

    monkeypatch.setattr(engine, "_backup", inspect_backup)
    result = engine.search(domain, domain.state())
    assert result.completed_simulations == 2 and result.partial_simulations == 1
    assert partial_stats == [(3, 5, 1), (2, 4, 1)]
    assert result.rootstats[0].W == 5
    assert ledger.used.env_steps == 2
