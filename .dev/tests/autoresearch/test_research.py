import json
from pathlib import Path

import pytest

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, BudgetLimits, Candidate, EpisodeSpec, EvidenceKind,
    GenerationResponse, InvalidCandidate, JSONLEventStore, JudgeResponse, ModelFailure, ModelIdentity,
    Observation, RunConfig, StepResult, Usage, UsagePhase,
)
from jevgo.research import (
    ProposalValidationError, ResearchPolicy, initialize_state, parse_generated_proposal,
    public_state, record_transition,
)


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
    return json.dumps({"action": {"name": name, "arguments": {}},
                       "hypothesis": "unverified " + name, "rationale": "test proposal"})


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


@pytest.mark.parametrize("text,stage", [
    ('{"action":{},"action":{},"hypothesis":"h","rationale":"r"}', "json"),
    ('{"action":{"name":"a","arguments":{"nested":[{"x":1,"x":2}]}},"hypothesis":"h","rationale":"r"}', "json"),
    ('{"action":{"name":"a","name":"b","arguments":{}},"hypothesis":"h","rationale":"r"}', "json"),
    *[(proposal("a").replace('"arguments": {}', '"arguments": {"value":' + value + '}'), "json")
      for value in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999")],
    ('{}', "schema"),
    ('[]', "schema"),
    ('null', "schema"),
    ('{"action":{"name":"a","arguments":{}}}', "schema"),
    (proposal("a").replace(', "rationale": "test proposal"', ''), "schema"),
    (proposal("a").replace('"arguments": {}', '"arguments": {}, "extra": 1'), "schema"),
    (proposal("a")[:-1] + ', "prior":1}', "schema"),
    (proposal("a").replace(', "arguments": {}', ''), "schema"),
    *[(proposal("a").replace('"arguments": {}', '"arguments": ' + value), "schema")
      for value in ('[]', 'null', 'false', '"{}"')],
    *[(proposal("a").replace('"name": "a"', '"name": ' + value), "schema")
      for value in ('1', 'null', 'false', '""')],
    (proposal("a").replace('"unverified a"', '7'), "schema"),
    (proposal("a").replace('"test proposal"', 'null'), "schema"),
    (proposal("a")[:-1], "json"),
    ('prefix ' + proposal("a"), "json"),
    (proposal("a") + ' trailing', "json"),
    (proposal("a") + proposal("b"), "json"),
    ('```json\n' + proposal("a"), "json"),
    ('```\n' + proposal("a") + '\n```', "json"),
    ('```JSON\n' + proposal("a") + '\n```', "json"),
    ('```json\r\n' + proposal("a") + '\r\n```', "json"),
    (proposal("illegal"), "public_action"),
    (proposal("a").replace('"arguments": {}', '"arguments": {"invented":1}'), "public_action"),
])
def test_generated_text_contract_rejects_without_salvage(text, stage):
    with pytest.raises(ProposalValidationError) as caught:
        parse_generated_proposal(text, admissible_actions=state().observation.admissible_actions)
    assert caught.value.stage == stage
    assert caught.value.reason


def test_generated_text_complete_fence_nested_json_and_internal_defaults():
    action = Action(name="a", arguments={"a": [True, None, 1, 1.5, {"x": "unicode 🐍"}]})
    text = json.dumps({"action": action.model_dump(mode="json"), "hypothesis": "", "rationale": ""})
    for raw in (text, ' \n```json\n' + text + '\n``` \n'):
        assert parse_generated_proposal(raw, admissible_actions=(action,)).action == action
    assert parse_generated_proposal(proposal("unlisted")).action.name == "unlisted"
    # The generated-text boundary must not make internal convenience defaults mandatory.
    assert Candidate(action=Action(name="a")).hypothesis == ""
    assert Candidate(action={"name": "a"}).action.arguments == {}


def test_historical_p14_duplicate_rejected_and_legal_sample_accepted():
    fixture = Path(__file__).resolve().parents[2] / "fixtures/p14d-generated-proposals.json"
    cases = json.loads(fixture.read_text())["cases"]
    legal, duplicate = cases
    actions = tuple(Action.model_validate(a) for a in legal["admissible_actions"])
    assert parse_generated_proposal(legal["text"], admissible_actions=actions).action.name == "forward"
    assert json.loads(duplicate["text"])["action"]["name"] == "forward"  # Original last-wins gap.
    with pytest.raises(ProposalValidationError, match="Duplicate JSON key: hypothesis"):
        parse_generated_proposal(duplicate["text"], admissible_actions=actions)
    assert duplicate["source"]["proposal_line"] == 45


class SequenceModel(Model):
    def __init__(self, batches):
        super().__init__()
        self.batches = iter(batches)

    def generate(self, request, *, ledger, phase):
        self.texts = next(self.batches)
        return super().generate(request, ledger=ledger, phase=phase)


def test_partial_valid_sources_and_per_sample_rejection_are_retained(tmp_path):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    generator = Model((proposal("a"), "bad", proposal("a"), proposal("illegal")))
    choices = ResearchPolicy(generator, None, RunConfig(method="B2"), events=store).candidates(
        state(), ledger=BudgetLedger())
    assert len(generator.requests) == 1 and choices[0].sample_count == 2
    samples, candidates = store.read()[1:]
    assert samples.payload["texts"] == generator.texts
    assert [item["error_stage"] for item in samples.payload["samples"]] == [None, "json", None, "public_action"]
    assert samples.payload["valid_sample_count"] == 2
    assert samples.payload["unique_action_count"] == 1
    sources = candidates.payload["candidate_sources"][0]
    assert sources["candidate_id"] == choices[0].candidate_id
    assert sources["sources"] == ({"attempt": 0, "sample_index": 0}, {"attempt": 0, "sample_index": 2})


@pytest.mark.parametrize("strategy", ["generic", "bounded"])
def test_zero_valid_repair_is_separate_bounded_and_traceable(tmp_path, strategy):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    malformed = "first bad output " + "🐍" * 100 + "END-MUST-NOT-APPEAR"
    generator = SequenceModel(((malformed, "SECOND-BAD-SAMPLE"), (proposal("b"), proposal("b"))))
    config = RunConfig(method="B2", candidate_count=2, seed=11, options={
        "proposal_repair": strategy, "proposal_repair_text_chars": 20,
        "proposal_repair_error_chars": 8, "max_new_tokens": 256})
    ledger = BudgetLedger()
    choices = ResearchPolicy(generator, None, config, events=store).candidates(state(), ledger=ledger)
    first, repair = generator.requests
    assert [request.seed for request in generator.requests] == [11, 12]
    assert first.model_dump(exclude={"prompt", "seed"}) == repair.model_dump(exclude={"prompt", "seed"})
    assert repair.max_new_tokens == 256 and repair.n == 2 and repair.temperature == .7
    assert ledger.used.generation_calls == ledger.used.model_requests == 2 and ledger.used.env_steps == 0
    assert [c.action.name for c in choices] == ["b"] and choices[0].sample_count == 2
    events = store.read()
    assert events[1].payload["texts"][0] == malformed
    assert events[3].payload["texts"] == (proposal("b"), proposal("b"))
    assert events[4].payload["candidate_sources"][0]["sources"] == (
        {"attempt": 1, "sample_index": 0}, {"attempt": 1, "sample_index": 1})
    if strategy == "generic":
        assert repair.prompt == first.prompt + "\nThe previous proposals were invalid. Return only the required JSON object."
        assert events[2].payload["repair_source"] == {}
    else:
        diagnostic = json.JSONDecoder().raw_decode(repair.prompt.split("Unicode characters:\n")[1])[0]
        assert diagnostic["invalid_output"] == malformed[:20]
        assert len(diagnostic["error"]) == 8 and diagnostic["sample_index"] == 0
        assert "END-MUST-NOT-APPEAR" not in repair.prompt and "SECOND-BAD-SAMPLE" not in repair.prompt
        source = events[2].payload["repair_source"]
        assert source["source_sample_index"] == source["source_attempt"] == 0
        assert source["text_char_limit"] == source["text_chars"] == 20
        assert source["text_chars_total"] == len(malformed)


def test_bounded_repair_failure_stops_after_one_and_preserves_reasons(tmp_path):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    generator = Model(("bad",))
    ledger = BudgetLedger()
    policy = ResearchPolicy(generator, None, RunConfig(method="B0", options={"proposal_repair": "bounded"}), events=store)
    with pytest.raises(InvalidCandidate, match="one repair attempt"):
        policy.candidates(state(), ledger=ledger)
    assert len(generator.requests) == ledger.used.generation_calls == 2
    assert ledger.used.env_steps == 0
    assert [event.payload["samples"][0]["error_stage"] for event in store.read()
            if event.event_type == "proposal_samples"] == ["json", "json"]


def test_repair_budget_rejection_preserves_first_response_and_cost(tmp_path):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    ledger = BudgetLedger(BudgetLimits(generation_calls=1), event_store=store)
    generator = Model(("bad",))
    with pytest.raises(BudgetExceeded):
        ResearchPolicy(generator, None, RunConfig(method="B0"), events=store).candidates(state(), ledger=ledger)
    assert ledger.used.generation_calls == ledger.used.model_requests == 1 and ledger.used.env_steps == 0
    assert [e.payload["attempt"] for e in store.read() if e.event_type == "generation_request"] == [0, 1]
    assert len([e for e in store.read() if e.event_type == "proposal_samples"]) == 1


def test_failed_repair_preserves_backend_unknown_usage_and_raw_response(tmp_path):
    store = JSONLEventStore(tmp_path / "events.jsonl")
    ledger = BudgetLedger(event_store=store)

    class FailedRepair(Model):
        def generate(self, request, *, ledger, phase):
            if self.requests:
                self.requests.append(request)
                ledger.charge(Usage(generation_calls=1, model_requests=1, input_tokens=37),
                              source="fixture.failed", phase=phase, failed=True,
                              metadata={"usage_complete": False, "unknown_usage": ["output_tokens"]})
                raise ModelFailure("repair decode failed; output usage unknown")
            response = super().generate(request, ledger=ledger, phase=phase)
            return response.model_copy(update={"metadata": {"original_backend_note": "retained"}})

    generator = FailedRepair(("bad",))
    with pytest.raises(ModelFailure, match="repair decode failed"):
        ResearchPolicy(generator, None, RunConfig(method="B0"), events=store).candidates(state(), ledger=ledger)
    assert ledger.used.generation_calls == ledger.used.model_requests == 2
    assert ledger.used.input_tokens == 37 and ledger.used.env_steps == 0
    failure = ledger.summary()["entries"][-1]
    assert failure["failed"] and failure["metadata"]["unknown_usage"] == ["output_tokens"]
    response = next(e for e in store.read() if e.event_type == "proposal_samples")
    assert response.payload["texts"] == ("bad",)
    assert response.payload["metadata"]["original_backend_note"] == "retained"


@pytest.mark.parametrize("options", [
    {"proposal_repair": "unlimited"}, {"proposal_repair_text_chars": 0},
    {"proposal_repair_error_chars": True}, {"proposal_repair_text_chars": 1.5},
])
def test_invalid_repair_config_rejected_before_calls(options):
    generator = Model()
    with pytest.raises(ValueError):
        ResearchPolicy(generator, None, RunConfig(method="B0", options=options))
    assert not generator.requests
