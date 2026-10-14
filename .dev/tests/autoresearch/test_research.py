import json

import pytest

from jevgo.core import (
    Action, BudgetLedger, BudgetLimits, Candidate, EpisodeSpec, EvidenceKind,
    GenerationResponse, InvalidCandidate, JSONLEventStore, JudgeResponse, ModelFailure, ModelIdentity,
    Observation, RunConfig, StepResult, Usage, UsagePhase,
)
from jevgo.research import ResearchPolicy, initialize_state, public_state, record_transition


class Model:
    identity = ModelIdentity(backend="fixture", model="test")

    def __init__(self, texts=(), scores=()):
        self.texts, self.scores, self.requests = texts, scores, []

    def generate(self, request, *, ledger, phase):
        self.requests.append(request)
        usage = Usage(generation_calls=1, model_requests=1)
        ledger.charge(usage, source="test", phase=phase)
        return GenerationResponse(texts=self.texts, identity=self.identity, usage=usage)

    def score(self, request, *, ledger, phase):
        self.requests.append(request)
        usage = Usage(judge_calls=1, model_requests=1)
        ledger.charge(usage, source="test", phase=phase)
        return JudgeResponse(scores=self.scores or (1,) * len(request.items), identity=self.identity, usage=usage)


def proposal(name):
    return json.dumps({"action": {"name": name}, "hypothesis": "unverified " + name})


def state():
    return initialize_state(EpisodeSpec(environment="test", task="goal"), StepResult(
        observation=Observation(text="public", admissible_actions=(Action(name="a"), Action(name="b"))),
        info={"private_oracle": "SECRET"},
    ), priors=("unverified prior",))


def test_sampling_prior_and_soft_judge_mixture():
    generator = Model((proposal("a"), proposal("a"), proposal("a"), proposal("b")))
    judge = Model(scores=(0, 1))
    policy = ResearchPolicy(generator, judge, RunConfig(method="B3"))
    ledger = BudgetLedger()
    choices = policy.candidates(state(), ledger=ledger)
    assert [c.sample_count for c in choices] == [3, 1]
    assert [c.prior for c in choices] == [.375, .625]
    assert len(choices) == 2  # Zero JEV score is not hard pruning.
    assert ledger.used.generation_calls == ledger.used.judge_calls == 1
    assert "SECRET" not in generator.requests[0].prompt
    assert "SECRET" not in judge.requests[0].context


def test_no_judge_for_baselines_and_b0_single_sample():
    for method, n in (("B0", 1), ("B2", 4)):
        generator, judge = Model((proposal("a"),) * n), Model()
        policy = ResearchPolicy(generator, judge, RunConfig(method=method))
        choices = policy.candidates(state(), ledger=BudgetLedger())
        assert choices[0].prior == 1
        assert generator.requests[0].n == n
        assert not judge.requests


def test_invalid_and_illegal_output_never_forms_an_experiment():
    generator = Model(("bad", proposal("illegal"), "{}", '{"action":{"name":"a"},"success":true}'))
    ledger = BudgetLedger()
    with pytest.raises(InvalidCandidate):
        ResearchPolicy(generator, None, RunConfig(method="B2")).candidates(state(), ledger=ledger)
    assert ledger.used.generation_calls == 2
    assert ledger.used.env_steps == 0


def test_partial_valid_output_is_retained_without_repair():
    generator = Model(("bad", proposal("a"), proposal("b"), "{}"))
    choices = ResearchPolicy(generator, None, RunConfig(method="B2")).candidates(state(), ledger=BudgetLedger())
    assert [c.prior for c in choices] == [.5, .5]
    assert len(generator.requests) == 1


def test_failure_evidence_and_siblings_are_isolated():
    parent = state()
    candidate = Candidate(action=Action(name="a"), hypothesis="door opens")
    left = record_transition(parent, candidate, StepResult(
        observation=Observation(text="door stayed closed"), action_success=False,
    ), branch_id="left")
    right = record_transition(parent, candidate, StepResult(
        observation=Observation(text="door opened"), action_success=True,
    ), branch_id="right")
    assert left.evidence[:len(parent.evidence)] == parent.evidence
    assert left.evidence[-1].raw["action_success"] is False
    assert left.evidence[-1].branch_id == "left"
    assert left.evidence[-1].evidence_id not in right.fact_ids
    assert parent.actions == ()
    assert left.hypotheses == ("door opens",)
    assert all(e.kind == EvidenceKind.OBSERVATION for e in left.evidence if e.evidence_id in left.fact_ids)


def test_memory_selection_never_deletes_raw_observations():
    initial = state()
    child = record_transition(initial, Candidate(action=Action(name="a")), StepResult(observation=Observation(text="failure")))
    policy = ResearchPolicy(Model(), Model(), RunConfig(method="B3", options={"memory_records": 1}))
    result = policy.select_memory(child, ledger=BudgetLedger())
    assert result.evidence == child.evidence
    assert len(result.working_memory) == 1
    assert set(result.working_memory) <= {e.evidence_id for e in result.evidence}
    assert public_state(result)["memory"][0]["content"] == "unverified prior"


def test_judge_advice_cannot_turn_unfinished_task_into_success():
    initial = state()
    result = ResearchPolicy(Model(), Model(scores=(1,)), RunConfig()).assess_completion(initial, ledger=BudgetLedger())
    assert not result.success and not result.done
    assert result.fact_ids == initial.fact_ids
    assert result.evidence[-1].kind == EvidenceKind.JUDGE_OPINION
    assert result.evidence[-1].references == initial.fact_ids


def test_score_mismatch_fails_without_discarding_ingested_state():
    ingested = record_transition(state(), Candidate(action=Action(name="a")), StepResult(observation=Observation(text="failure")))
    policy = ResearchPolicy(Model(), Model(scores=(.5, .5)), RunConfig())
    with pytest.raises(ModelFailure):
        policy.assess_completion(ingested, ledger=BudgetLedger())
    assert ingested.evidence[-1].content == "failure"
    assert len(ingested.actions) == 1


def test_public_prompts_ignore_random_trace_identifiers():
    first, second = state(), state()
    assert first.state_id != second.state_id
    assert first.evidence[0].evidence_id != second.evidence[0].evidence_id
    assert public_state(first) == public_state(second)
    candidate = Candidate(action=Action(name="a"), hypothesis="test hypothesis")
    step = StepResult(observation=Observation(text="actual evidence"))
    first = record_transition(first, candidate, step, branch_id="root/random-one")
    second = record_transition(second, candidate, step, branch_id="root/random-two")
    assert public_state(first) == public_state(second)
    for value in (first.state_id, first.evidence[0].evidence_id, first.branch_id):
        assert value not in json.dumps(public_state(first))


def test_image_paths_are_stable_in_prompts_and_preserved_in_requests_and_evidence(tmp_path):
    requests, states = [], []
    for worker in ("random-worker-one", "random-worker-two"):
        paths = (f"/runs/{worker}/view.png", f"/runs/{worker}/detail.png")
        initial = initialize_state(EpisodeSpec(environment="test", task="goal"), StepResult(
            observation=Observation(text="same visible scene", image_paths=paths)))
        original = initial.model_dump(mode="json")
        generator, judge = Model((proposal("a"),)), Model()
        store = JSONLEventStore(tmp_path / worker / "events.jsonl")
        policy = ResearchPolicy(generator, judge, RunConfig(method="B1", candidate_count=1), events=store)
        policy.candidates(initial, ledger=BudgetLedger(), phase=UsagePhase.COMMIT)
        generation, scoring = generator.requests[0], judge.requests[0]
        assert generation.image_paths == scoring.image_paths == paths
        assert all(path not in generation.prompt and path not in scoring.context for path in paths)
        assert public_state(initial)["observation"]["image_paths"] == ["image_0", "image_1"]
        assert initial.model_dump(mode="json") == original
        assert initial.evidence[-1].raw["observation"]["image_paths"] == paths
        events = store.read()
        assert [event.event_type for event in events] == [
            "generation_request", "proposal_samples", "judge_request", "judge", "candidates"]
        for event, request in ((events[0], generation), (events[2], scoring)):
            assert event.payload["phase"] == "commit"
            assert event.model_dump(mode="json")["payload"]["request"] == request.model_dump(mode="json")
        assert all("request" not in event.payload for event in (events[1], events[3]))
        requests.append((generation, scoring))
        states.append(initial)
    assert public_state(states[0]) == public_state(states[1])
    assert requests[0][0].prompt == requests[1][0].prompt
    assert requests[0][1].context == requests[1][1].context


@pytest.mark.parametrize("backend", ["generation", "judge"])
@pytest.mark.parametrize("phase", [UsagePhase.SEARCH, UsagePhase.COMMIT])
def test_failed_backend_request_is_persisted_before_call(tmp_path, backend, phase):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    failure = ModelFailure("backend unavailable")
    initial = state()

    class FailingModel(Model):
        def fail(self, request, *, ledger, phase):
            self.requests.append(request)
            events = store.read()  # Read the persisted file inside the backend call.
            assert len(events) == 1
            event = events[0]
            assert event.event_type == backend + "_request"
            assert event.payload["phase"] == phase.value
            assert event.episode_id == initial.episode.episode_id
            assert event.branch_id == initial.branch_id
            assert event.model_dump(mode="json")["payload"]["request"] == request.model_dump(mode="json")
            raise failure

        generate = score = fail

    model = FailingModel()
    policy = ResearchPolicy(model, model, RunConfig(method="B1"), events=store)
    operation = policy.candidates if backend == "generation" else policy.assess_completion
    with pytest.raises(ModelFailure) as caught:
        operation(initial, ledger=BudgetLedger(), phase=phase)
    assert caught.value is failure
    assert len(model.requests) == len(store.read()) == 1


def test_repair_attempt_logs_each_request_once_before_response(tmp_path):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    generator = Model(("invalid JSON",))
    policy = ResearchPolicy(generator, None, RunConfig(method="B0"), events=store)
    with pytest.raises(InvalidCandidate):
        policy.candidates(state(), ledger=BudgetLedger())
    events = store.read()
    assert [event.event_type for event in events] == [
        "generation_request", "proposal_samples", "generation_request", "proposal_samples"]
    for attempt, (request_event, response_event) in enumerate(zip(events[::2], events[1::2])):
        assert request_event.payload["attempt"] == response_event.payload["attempt"] == attempt
        assert request_event.payload["phase"] == "search"
        assert request_event.payload["request"]["prompt"] == generator.requests[attempt].prompt
        assert "request" not in response_event.payload
    assert generator.requests[0].prompt != generator.requests[1].prompt
