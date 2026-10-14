"""Public contract and accounting invariants, without models or benchmarks."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, BudgetLimits, Candidate,
    EnvironmentCapabilities, EnvironmentSnapshot, EpisodeSpec, EvidenceKind,
    EvidenceRecord, EventRecord, ExperimentPlan, GenerationRequest,
    GenerationResponse, JSONLEventStore, JudgeRequest, JudgeResponse, JudgeRole,
    ModelIdentity, Observation, ResearchState, RunConfig, RunResult,
    SearchState, SearchTransition, StepResult, TerminationReason,
    UnsupportedCapability, Usage, UsagePhase, ValueEstimate,
)


def episode():
    return EpisodeSpec(episode_id="episode", environment="fixture", task="reach goal")


def state(**kwargs):
    return ResearchState(episode=episode(), observation=Observation(text="start"), **kwargs)


def evidence(evidence_id="e1", branch="root", **kwargs):
    return EvidenceRecord(evidence_id=evidence_id, episode_id="episode", branch_id=branch,
                          kind=EvidenceKind.OBSERVATION, origin="environment", source="fixture.step",
                          content="door is closed", raw={"door": {"open": False}}, **kwargs)


def test_nested_state_is_owned_frozen_and_serializes_as_json():
    payload = {"grid": [[1, 2]], "nested": {"visible": True}}
    observation = Observation(data=payload)
    original = ResearchState(episode=episode(), observation=observation)
    payload["grid"][0][0] = 99
    payload["nested"]["visible"] = False
    assert original.observation.data["grid"] == ((1, 2),)
    with pytest.raises(TypeError):
        original.observation.data["nested"]["visible"] = False
    with pytest.raises(TypeError):
        original.observation.data["grid"][0][0] = 3
    with pytest.raises(ValidationError):
        original.branch_id = "other"
    decoded = json.loads(original.model_dump_json())
    decoded["observation"]["data"]["grid"][0][0] = 100
    assert original.observation.data["grid"][0][0] == 1
    assert ResearchState.model_validate_json(original.model_dump_json()) == original


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), object(), b"opaque", {1: "not a JSON key"}])
def test_non_json_payloads_are_rejected(bad):
    with pytest.raises(ValidationError):
        Observation(data={"bad": bad})


def test_validated_copy_cannot_smuggle_mutable_payloads_or_invalid_budget():
    incoming = {"nested": [1]}
    copied = Observation().model_copy(update={"data": incoming}, deep=True)
    incoming["nested"].append(2)
    assert copied.data["nested"] == (1,)
    with pytest.raises(ValidationError):
        Usage().model_copy(update={"env_steps": -1})


def test_forks_keep_raw_evidence_and_independent_working_memory():
    root = state(evidence=(evidence(),), fact_ids=("e1",), working_memory=("old summary",))
    left = root.evolve(branch_id="left", evidence=root.evidence + (evidence("e2", "left"),),
                       working_memory=("compressed",))
    right = root.evolve(branch_id="right", working_memory=())
    assert len(root.evidence) == len(right.evidence) == 1
    assert len(left.evidence) == 2
    assert left.parent_id == root.state_id and left.version == 1
    assert left.state_id != right.state_id != root.state_id
    assert right.evidence[0].branch_id == "root"
    with pytest.raises(ValueError, match="append-only"):
        left.evolve(evidence=(), fact_ids=())
    with pytest.raises(ValueError, match="identity"):
        root.evolve(version=0)


def test_judge_and_inference_cannot_become_environment_facts():
    opinion = EvidenceRecord(episode_id="episode", branch_id="root", kind=EvidenceKind.JUDGE_OPINION,
                             origin="model", source="judge", content="probably solved")
    with pytest.raises(ValidationError, match="environment"):
        opinion.model_copy(update={"kind": EvidenceKind.OBSERVATION})
    with pytest.raises(ValidationError, match="Facts"):
        state(evidence=(opinion,), fact_ids=(opinion.evidence_id,))
    assert not state(evidence=(opinion,)).success
    with pytest.raises(ValidationError, match="episode"):
        state(evidence=(evidence().model_copy(update={"episode_id": "foreign"}),))


def test_snapshot_is_opaque_and_never_enters_research_json():
    snapshot = EnvironmentSnapshot(owner_id="fixture", payload={"hidden": "secret"})
    branch = SearchState(research=state(), snapshot=snapshot)
    assert "secret" not in branch.research.model_dump_json()
    assert "secret" not in repr(branch)
    with pytest.raises(ValidationError):
        Observation(data={"snapshot": snapshot})
    with pytest.raises(ValidationError):
        state().model_copy(update={"snapshot": snapshot})


def test_reserve_blocks_search_and_replay_but_allows_actual_commit():
    ledger = BudgetLedger(BudgetLimits(env_steps=5), commit_reserve=Usage(env_steps=1))
    ledger.charge(Usage(env_steps=3), source="adapter.step", phase=UsagePhase.SEARCH)
    ledger.charge(Usage(env_steps=1, replay_steps=1), source="adapter.replay", phase=UsagePhase.REPLAY)
    assert ledger.remaining(phase="search")["env_steps"] == 0
    with pytest.raises(BudgetExceeded):
        ledger.charge(Usage(env_steps=1), source="search", phase="search")
    with pytest.raises(BudgetExceeded):
        ledger.charge(Usage(env_steps=1), source="replay", phase="replay")
    assert ledger.used.env_steps == 4 and len(ledger.entries) == 2
    ledger.charge(Usage(env_steps=1), source="adapter.step", phase="commit")
    assert ledger.used.env_steps == 5
    assert ledger.remaining()["env_steps"] == 0


def test_commit_recovery_uses_explicit_parent_budget_and_preserves_replay_label():
    ledger = BudgetLedger(BudgetLimits(env_steps=2), commit_reserve=Usage(env_steps=2))
    ledger.charge(Usage(env_steps=1, replay_steps=1), source="adapter.replay", phase="replay",
                  budget_phase="commit")
    ledger.charge(Usage(env_steps=1), source="adapter.step", phase="commit")
    assert ledger.entries[0].phase == UsagePhase.REPLAY
    assert ledger.entries[0].budget_phase == "commit"
    with pytest.raises(ValueError, match="Only replay"):
        ledger.charge(Usage(), source="bad", phase="search", budget_phase="commit")


def test_branch_restore_never_rolls_back_shared_costs():
    ledger = BudgetLedger(BudgetLimits(env_steps=10))
    parent = EnvironmentSnapshot(owner_id="fixture", payload=0)
    position = parent.payload
    for direction in (1, -1):
        ledger.charge(Usage(env_restores=1), source="adapter.restore", phase="search")
        position = parent.payload
        ledger.charge(Usage(env_steps=1), source="adapter.step", phase="search")
        position += direction
    assert position == -1 and parent.payload == 0
    assert ledger.used.env_steps == 2 and ledger.used.env_restores == 2
    assert [entry.sequence for entry in ledger.entries] == [1, 2, 3, 4]


def test_failure_overage_is_recorded_and_upper_layers_do_not_recharge_response(tmp_path):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    ledger = BudgetLedger(BudgetLimits(generation_calls=1, output_tokens=2), event_store=store)
    ledger.charge(Usage(generation_calls=1, model_requests=1), source="backend.generate", phase="search")
    # Actual work remains accountable even when an external backend exceeds its cap or fails.
    ledger.record(Usage(output_tokens=3, wall_seconds=0.2), source="backend.generate",
                  phase="search", failed=True, metadata={"error": "backend failed"})
    assert ledger.used.generation_calls == 1
    assert ledger.used.output_tokens == 3 and ledger.over_budget == ("output_tokens",)
    assert ledger.entries[-1].failed
    assert not ledger.can_afford(Usage(output_tokens=1), phase="search")
    assert [event.payload["sequence"] for event in store.read()] == [1, 2]
    json.dumps(ledger.summary(), allow_nan=False)


def test_atomic_charges_cannot_oversubscribe_budget():
    ledger = BudgetLedger(BudgetLimits(env_steps=10))

    def attempt(_):
        try:
            ledger.charge(Usage(env_steps=1), source="worker", phase="search")
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(attempt, range(40)))
    assert sum(accepted) == ledger.used.env_steps == len(ledger.entries) == 10


def test_jsonl_is_append_only_and_reports_corrupt_rows(tmp_path):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    event = EventRecord(event_type="observation", episode_id="episode", branch_id="root",
                        payload={"raw": {"readings": [1, 2]}, "text": "line one\nline two"})
    store.append(event)
    JSONLEventStore(store.path).append(event.model_copy(update={"event_type": "failure"}))
    assert len(store.path.read_text().splitlines()) == 2
    assert store.read()[0] == event
    with store.path.open("a") as stream:
        stream.write('{"incomplete":')
    with pytest.raises(ValueError, match=":3"):
        store.read()


@pytest.mark.parametrize("mode", ["value_only", "policy_only", "joint", "unknown"])
def test_training_is_explicitly_rejected(mode):
    with pytest.raises(ValidationError, match="none"):
        RunConfig(learning_mode=mode)


def test_success_is_objective_and_distinct_from_budget_exhaustion():
    config = RunConfig()
    partial = RunResult(episode=episode(), config=config, final_state=state(), usage=Usage(env_steps=10),
                        termination_reason=TerminationReason.BUDGET_EXHAUSTED, answer="partial")
    assert not partial.success
    solved = RunResult(episode=episode(), config=config, final_state=state(success=True), usage=Usage(),
                       termination_reason=TerminationReason.SUCCESS)
    assert solved.success
    with pytest.raises(ValidationError, match="objective"):
        partial.model_copy(update={"termination_reason": TerminationReason.SUCCESS})
    with pytest.raises(ValidationError, match="objective"):
        solved.model_copy(update={"termination_reason": TerminationReason.BUDGET_EXHAUSTED})


def test_nonrecoverable_environments_reject_search_and_transitions_are_explicit():
    with pytest.raises(UnsupportedCapability):
        EnvironmentCapabilities().require_branching()
    EnvironmentCapabilities(deterministic_replay=True).require_branching()
    transition = SearchTransition(SearchState(state(success=True)), reward=1.0)
    assert transition.state.terminal
    with pytest.raises(ValueError):
        SearchTransition(SearchState(state()), reward=float("nan"))
    assert ValueEstimate(value=0, source="environment").value == 0


def test_public_payloads_roundtrip_without_python_only_objects():
    identity = ModelIdentity(backend="fixture", model="frozen-fixture")
    values = [
        Action(name="move", arguments={"direction": "left"}),
        Candidate(action=Action(name="move"), prior=0.5),
        ExperimentPlan(candidate_id="c", actions=(Action(name="move"),)),
        GenerationRequest(prompt="task", n=2),
        GenerationResponse(texts=("a", "b"), identity=identity, usage=Usage(generation_calls=1)),
        JudgeRequest(role=JudgeRole.CANDIDATE, context="task", items=("a", "b")),
        JudgeResponse(scores=(0.2, 0.8), identity=identity, usage=Usage(judge_calls=1)),
        StepResult(observation=Observation(text="goal"), success=True),
        RunConfig(),
    ]
    for value in values:
        assert type(value).model_validate_json(value.model_dump_json()) == value


def test_action_success_is_separate_and_observations_are_accountable():
    feedback = StepResult(observation=Observation(text="invalid move"), action_success=False)
    assert feedback.action_success is False and not feedback.done
    ledger = BudgetLedger(BudgetLimits(env_observations=1))
    ledger.charge(Usage(env_observations=1), source="adapter.refresh", phase="commit")
    assert ledger.used.env_observations == 1
    with pytest.raises(BudgetExceeded):
        ledger.charge(Usage(env_observations=1), source="adapter.refresh", phase="commit")
