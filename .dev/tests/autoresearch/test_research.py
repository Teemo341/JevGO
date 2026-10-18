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
    assert sources["sources"] == ({"attempt": 0, "source": "initial", "sample_index": 0}, {"attempt": 0, "source": "initial", "sample_index": 2})


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
        {"attempt": 1, "source": "repair", "sample_index": 0}, {"attempt": 1, "source": "repair", "sample_index": 1})
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


def minigrid_public_state():
    from jevgo.envs.minigrid import MiniGridAdapter
    image = [[[1, 0, 0] for _ in range(7)] for _ in range(7)]
    raw = {"image": image, "direction": 0, "mission": "public mission"}
    initial = initialize_state(EpisodeSpec(environment="minigrid", task="public task"),
                               StepResult(observation=MiniGridAdapter._observation(raw)))
    image[3][5] = [2, 5, 0]
    return record_transition(initial, Candidate(action=Action(name="forward"), hypothesis="I moved through the wall"),
                             StepResult(observation=MiniGridAdapter._observation(raw),
                                        action_success=False, info={"hidden_fingerprint": "PRIVATE_SECRET"}))


def test_observation_modes_keep_legacy_bytes_and_split_e1_from_e2():
    current = minigrid_public_state()
    legacy = public_state(current)
    assert legacy == public_state(current, public_observation="legacy")
    assert set(legacy) == {"task", "success_criteria", "observation", "actions", "memory"}
    assert legacy["observation"]["text"] == current.observation.text
    assert "data.image[x][y][channel]" not in legacy["observation"]["text"]
    e1 = public_state(current, public_observation="e1")
    e2 = public_state(current, public_observation="e1_e2")
    assert "data.image[x][y][channel]" in e1["observation"]["text"]
    assert e1["observation"] == e2["observation"]
    assert e1 == {key: value for key, value in e2.items() if key != "observation_feedback"}
    assert current.observation.model_dump(mode="json") == legacy["observation"]
    assert e2["observation_feedback"]["local_cell_changes"] == [
        {"x": 3, "y": 5, "before": [1, 0, 0], "after": [2, 5, 0]}]
    assert "PRIVATE_SECRET" not in json.dumps(e2)
    # This task changes observation expression, not model hypothesis semantics.
    assert "I moved through the wall" in [item["content"] for item in e2["memory"]]


def test_e2_is_immediate_feedback_even_when_memory_is_disabled_or_compressed():
    current = minigrid_public_state()
    expected = public_state(current, public_observation="e1_e2")["observation_feedback"]
    for use_memory, working_memory in ((False, current.working_memory), (True, ()), (False, ())):
        view = public_state(current.evolve(working_memory=working_memory), use_memory=use_memory,
                            public_observation="e1_e2")
        assert view["memory"] == []
        assert view["observation_feedback"] == expected
    assert expected["previous_observation_ref"] == "e0"
    assert expected["observation_ref"] == "e2"  # e1 is the unverified hypothesis.


def test_e2_unavailable_history_is_explicit_and_does_not_guess():
    current = minigrid_public_state()
    for malformed in (
        current.model_copy(update={"evidence": (), "fact_ids": (), "working_memory": ()}),
        current.model_copy(update={"branch_id": "other-branch"}),
    ):
        feedback = public_state(malformed, public_observation="e1_e2")["observation_feedback"]
        assert feedback["status"] == "unavailable"
        assert feedback["local_cell_changes"] == [] and feedback["changed_fields"] == []
    initial = initialize_state(EpisodeSpec(environment="test", task="goal"),
                               StepResult(observation=Observation(text="initial")))
    assert public_state(initial, public_observation="e1_e2")["observation_feedback"]["status"] == "initial"


def test_e2_rejects_an_archive_with_an_inserted_sibling_observation():
    parent = state()
    left = record_transition(parent, Candidate(action=Action(name="a")),
                             StepResult(observation=Observation(text="left sibling")), branch_id="left")
    right = record_transition(parent, Candidate(action=Action(name="b")),
                              StepResult(observation=Observation(text="right sibling")), branch_id="right")
    merged = right.model_copy(update={"evidence": parent.evidence + (left.evidence[-1], right.evidence[-1])})
    feedback = public_state(merged, public_observation="e1_e2")["observation_feedback"]
    assert feedback["status"] == "unavailable"
    assert feedback["previous_observation_ref"] is None and feedback["local_cell_changes"] == []


@pytest.mark.parametrize("mode", ["legacy", "e1", "e1_e2"])
@pytest.mark.parametrize("use_memory", [False, True])
def test_all_methods_and_judge_roles_share_selected_public_information(tmp_path, mode, use_memory):
    from jevgo.core import JudgeRole
    current = minigrid_public_state()
    expected = public_state(current, use_memory=use_memory, public_observation=mode)
    contexts = []
    for method in ("B0", "B1", "B2", "B3"):
        n = 1 if method == "B0" else 4
        generator, judge = Model((proposal("forward"),) * n), Model()
        policy = ResearchPolicy(generator, judge, RunConfig(method=method, use_memory=use_memory,
                                options={"public_observation": mode}),
                                events=JSONLEventStore(tmp_path / method / "events.jsonl"))
        ledger = BudgetLedger()
        policy.candidates(current, ledger=ledger)
        context = json.JSONDecoder().raw_decode(generator.requests[0].prompt.split("Public state:\n")[1])[0]
        assert context == expected
        contexts.append(context)
        if method in ("B1", "B3"):
            for role in JudgeRole:
                policy._scores(current, role, "public context test", ("item",), ledger, UsagePhase.SEARCH)
            assert all(json.loads(request.context.split("Public state:\n")[1]) == expected for request in judge.requests)
        else:
            assert not judge.requests
        request_events = [e for e in policy.events.read() if e.event_type.endswith("_request")]
        assert all(e.payload["public_observation"] == mode for e in request_events)
    assert contexts == [expected] * 4


def test_invalid_public_observation_mode_rejected_before_calls():
    generator = Model()
    with pytest.raises(ValueError, match="public_observation"):
        ResearchPolicy(generator, None, RunConfig(method="B0", options={"public_observation": "hidden"}))
    with pytest.raises(ValueError, match="public_observation"):
        public_state(state(), public_observation="hidden")
    assert not generator.requests


class BatchModel(Model):
    def __init__(self, batches):
        super().__init__()
        self.batches = list(batches)

    def generate(self, request, *, ledger, phase):
        self.texts = self.batches[len(self.requests)]
        return super().generate(request, ledger=ledger, phase=phase)


def coverage_policy(tmp_path, batches, *, method="B2", **options):
    generator = BatchModel(batches)
    store = JSONLEventStore(tmp_path / "events.jsonl")
    return ResearchPolicy(generator, None, RunConfig(method=method,
        options={"candidate_coverage": "bounded", **options}), events=store), generator, store


def test_bounded_coverage_merges_only_strict_samples_and_keeps_same_prompt(tmp_path):
    policy, generator, store = coverage_policy(tmp_path, [
        (proposal("a"),) * 4, (proposal("b"), proposal("a"), "bad", proposal("b"))])
    ledger = BudgetLedger()
    values = policy.candidates(state(), ledger=ledger, proposal_scope="root")
    assert [v.action.name for v in values] == ["a", "b"]
    assert [v.sample_count for v in values] == [5, 2]
    assert [v.prior for v in values] == pytest.approx([5 / 7, 2 / 7])
    assert generator.requests[0].prompt == generator.requests[1].prompt
    assert generator.requests[1].seed == generator.requests[0].seed + 1
    assert ledger.used.generation_calls == ledger.used.model_requests == 2
    events = store.read()
    requests = [e.payload for e in events if e.event_type == "generation_request"]
    assert [e["source"] for e in requests] == ["initial", "coverage"]
    assert [e["attempt"] for e in requests] == [0, 1]
    assert all(e["proposal_scope"] == "root" for e in requests)
    result = events[-1].payload
    assert result["sample_count"] == 8 and result["valid_sample_count"] == 7
    assert result["unique_action_count"] == 2 and not result["single_candidate"]
    assert result["coverage_rounds"] == 1 and result["coverage_status"] == "target_met_after_coverage"
    assert result["frequency_distribution"] == "same_original_prompt"
    assert sum(len(item["sources"]) for item in result["candidate_sources"]) == 7


@pytest.mark.parametrize("second", [("bad",) * 4, (proposal("a"),) * 4])
def test_coverage_cannot_add_repair_or_loop_until_satisfied(tmp_path, second):
    policy, generator, store = coverage_policy(tmp_path, [(proposal("a"),) * 4, second])
    values = policy.candidates(state(), ledger=BudgetLedger())
    assert len(generator.requests) == 2 and len(values) == 1
    assert values[0].sample_count == (4 if second[0] == "bad" else 8)
    assert store.read()[-1].payload["coverage_status"] == "coverage_target_unmet"


def test_initial_all_invalid_gets_repair_only_even_if_coverage_still_insufficient(tmp_path):
    policy, generator, store = coverage_policy(tmp_path, [("bad",) * 4, (proposal("a"),) * 4])
    values = policy.candidates(state(), ledger=BudgetLedger())
    requests = [e.payload for e in store.read() if e.event_type == "generation_request"]
    assert [e["source"] for e in requests] == ["initial", "repair"]
    assert generator.requests[1].prompt != generator.requests[0].prompt
    assert values[0].sample_count == 4
    result = store.read()[-1].payload
    assert result["coverage_rounds"] == 0 and result["coverage_status"] == "repair_only"
    assert result["frequency_distribution"] == "conditional_repair_prompt"


@pytest.mark.parametrize("scenario", ["off", "b0", "singleton", "already_two"])
def test_no_extra_call_when_disabled_inapplicable_or_target_met(tmp_path, scenario):
    n = 1 if scenario == "b0" else 4
    texts = (proposal("a"),) * n if scenario != "already_two" else (proposal("a"), proposal("b")) * 2
    policy, generator, _ = coverage_policy(tmp_path, [texts], method="B0" if scenario == "b0" else "B2",
                                            candidate_coverage="off" if scenario == "off" else "bounded")
    initial = state()
    if scenario == "singleton":
        initial = initial.evolve(observation=Observation(text="one action", admissible_actions=(Action(name="a"),)))
    policy.candidates(initial, ledger=BudgetLedger())
    assert len(generator.requests) == 1


@pytest.mark.parametrize("phase", [UsagePhase.SEARCH, UsagePhase.COMMIT])
def test_unspent_coverage_budget_refusal_preserves_candidates_and_commit_reserve(tmp_path, phase):
    policy, generator, store = coverage_policy(tmp_path, [(proposal("a"),) * 4, (proposal("b"),) * 4])
    ledger = BudgetLedger(BudgetLimits(generation_calls=2 if phase == UsagePhase.SEARCH else 1),
                          commit_reserve=Usage(generation_calls=1) if phase == UsagePhase.SEARCH else Usage())
    values = policy.candidates(state(), ledger=ledger, phase=phase)
    assert len(values) == 1 and values[0].action.name == "a"
    assert ledger.used.generation_calls == 1 and len(ledger.entries) == 1
    assert store.read()[-1].payload["coverage_status"] == "coverage_budget_limited"
    assert store.read()[-1].payload["generation_calls_attempted"] == 2
    assert store.read()[-1].payload["generation_calls_completed"] == 1
    assert any(e.event_type == "coverage_budget_limited" and not e.payload["charged"] for e in store.read())
    if phase == UsagePhase.SEARCH:
        assert ledger.remaining(phase=UsagePhase.COMMIT)["generation_calls"] == 1


@pytest.mark.parametrize("failure", ["charged_budget", "zero_entry_budget", "model_error"])
def test_coverage_errors_after_admission_are_not_swallowed(tmp_path, failure):
    class FailingCoverage(BatchModel):
        def generate(self, request, *, ledger, phase):
            if self.requests:
                self.requests.append(request)
                ledger.record(Usage(model_requests=int(failure != "zero_entry_budget")),
                              source="failed_partial", phase=phase, failed=True)
                if failure == "model_error":
                    raise ModelFailure("real backend failure")
                raise BudgetExceeded(("model_requests",), phase)
            return super().generate(request, ledger=ledger, phase=phase)
    generator = FailingCoverage([(proposal("a"),) * 4])
    policy = ResearchPolicy(generator, None, RunConfig(method="B2", options={"candidate_coverage": "bounded"}),
                            events=JSONLEventStore(tmp_path / "events.jsonl"))
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure if failure == "model_error" else BudgetExceeded):
        policy.candidates(state(), ledger=ledger)
    assert len(ledger.entries) == 2 and ledger.entries[-1].failed
    assert not any(e.event_type == "coverage_budget_limited" for e in policy.events.read())


def test_existing_measured_overage_cannot_be_hidden_as_optional_coverage(tmp_path):
    class OverBudget(BatchModel):
        def generate(self, request, *, ledger, phase):
            if self.requests:
                ledger.ensure_available(Usage(), phase=phase)
            response = super().generate(request, ledger=ledger, phase=phase)
            ledger.record(Usage(input_tokens=2), source="measured-overage", phase=phase)
            return response
    model = OverBudget([(proposal("a"),) * 4])
    policy = ResearchPolicy(model, None, RunConfig(method="B2", options={"candidate_coverage": "bounded"}))
    with pytest.raises(BudgetExceeded):
        policy.candidates(state(), ledger=BudgetLedger(BudgetLimits(input_tokens=1)))


def test_policy_rejects_over_eight_samples_without_changing_general_config():
    config = RunConfig(method="B2", candidate_count=5)
    assert config.candidate_count == 5  # General model API remains valid.
    with pytest.raises(ValueError, match="at most 4"):
        ResearchPolicy(Model(), None, config)
    b0 = ResearchPolicy(Model((proposal("a"),)), None, RunConfig(method="B0", candidate_count=99,
                         options={"candidate_coverage": "bounded"}))
    b0.candidates(state(), ledger=BudgetLedger())
    assert b0.generator.requests[0].n == 1


def test_coverage_all_invalid_repair_remains_bounded_and_fully_charged(tmp_path):
    policy, generator, store = coverage_policy(tmp_path, [("bad",) * 4, ("bad",) * 4])
    ledger = BudgetLedger()
    with pytest.raises(InvalidCandidate):
        policy.candidates(state(), ledger=ledger)
    assert ledger.used.generation_calls == ledger.used.model_requests == 2
    batches = [e.payload for e in store.read() if e.event_type == "proposal_samples"]
    assert [batch["source"] for batch in batches] == ["initial", "repair"]
    assert sum(batch["sample_count"] for batch in batches) == 8
    assert sum(batch["valid_sample_count"] for batch in batches) == 0


def test_unknown_public_action_set_still_uses_two_distinct_action_target(tmp_path):
    policy, generator, store = coverage_policy(tmp_path, [(proposal("a"),) * 4, (proposal("b"),) * 4])
    initial = state().evolve(observation=Observation(text="No action list was supplied"))
    values = policy.candidates(initial, ledger=BudgetLedger())
    assert len(values) == 2 and len(generator.requests) == 2
    assert store.read()[-1].payload["coverage_target"] == 2


def test_invalid_coverage_option_rejected_before_generation():
    model = Model()
    with pytest.raises(ValueError, match="candidate_coverage"):
        ResearchPolicy(model, None, RunConfig(method="B2", options={"candidate_coverage": "unbounded"}))
    assert not model.requests


def archived_state(count):
    initial = initialize_state(EpisodeSpec(environment="test", task="test evidence"), StepResult(
        observation=Observation(text="Old counterevidence: the door stayed closed",
            admissible_actions=(Action(name="a"), Action(name="b")))))
    for i in range(count - 1):
        initial = record_transition(initial, Candidate(action=Action(name="a")), StepResult(
            observation=Observation(text=f"Observation {i}",
                admissible_actions=(Action(name="a"), Action(name="b")))))
    return initial


@pytest.mark.parametrize("maximum", [4, 32])
@pytest.mark.parametrize("excess", [-2, 0, 5])
def test_g0_no_choice_skips_all_scoring_and_preserves_legacy_ids_order(tmp_path, maximum, excess):
    initial = archived_state(maximum + excess)
    judge = Model(scores=(0,))  # Would be invalid if called on a multi-item pool.
    ledger = BudgetLedger(BudgetLimits(judge_calls=0, model_requests=0, input_tokens=0, output_tokens=0))
    policy = ResearchPolicy(Model(), judge, RunConfig(method="B3", candidate_count=4,
        options={"memory_records": maximum}), events=JSONLEventStore(tmp_path / "events.jsonl"))
    selected = policy.select_memory(initial, ledger=ledger)
    assert selected.working_memory == tuple(e.evidence_id for e in initial.evidence[-maximum:])
    assert selected.evidence == initial.evidence
    assert ledger.used == Usage() and ledger.entries == () and not judge.requests
    event = policy.events.read()[-1].payload
    assert event["pool_capacity"] == maximum and event["retention_limit"] == maximum
    assert event["pool_ids"] == selected.working_memory == event["selected_ids"]
    assert event["scores"] is None and event["skip_reason"] == "pool_fits_retention"
    assert policy.mechanism_metadata()["memory"]["no_choice_skips"] == 1
    assert not policy.mechanism_metadata()["memory"]["scoring_applied"]


class ItemCostJudge(Model):
    def score(self, request, *, ledger, phase):
        self.requests.append(request)
        usage = Usage(judge_calls=1, model_requests=len(request.items),
                      input_tokens=7 * len(request.items), output_tokens=len(request.items))
        ledger.charge(usage, source="fixture.item-scoring", phase=phase)
        return JudgeResponse(scores=self.scores, identity=self.identity, usage=usage)


def test_g1_score_counterfactual_changes_actual_next_generation_prompt(tmp_path):
    initial = archived_state(4)
    # Simulate earlier compression: the history pool must still recall old evidence.
    initial = initial.evolve(working_memory=(initial.evidence[-1].evidence_id,))
    original = initial.model_dump(mode="json")
    results, prompts, events = [], [], []
    for index, scores in enumerate(((1, .2, .1, 0), (0, .2, .1, 1))):
        generator, judge = Model((proposal("a"),) * 4), ItemCostJudge(scores=scores)
        policy = ResearchPolicy(generator, judge, RunConfig(method="B3", judge_candidates=False,
            judge_stop=False, options={"memory_records": 2, "memory_pool_policy": "history_tail",
                                      "memory_pool_records": 4}),
            events=JSONLEventStore(tmp_path / str(index) / "events.jsonl"))
        ledger = BudgetLedger(BudgetLimits(model_requests=5, input_tokens=28, output_tokens=4))
        selected = policy.select_memory(initial, ledger=ledger)
        policy.candidates(selected, ledger=ledger)
        results.append(selected)
        prompts.append(json.loads(generator.requests[0].prompt.split("Public state:\n", 1)[1]))
        events.append(next(e.payload for e in policy.events.read() if e.event_type == "memory_selection"))
        assert ledger.used.model_requests == 5 and ledger.used.input_tokens == 28 and ledger.used.output_tokens == 4
        assert tuple(judge.requests[0].items) == tuple(e.content for e in initial.evidence)
        assert selected.evidence == initial.evidence
        assert policy.mechanism_metadata()["memory"]["scoring_applied"]
    assert initial.model_dump(mode="json") == original
    assert results[0].working_memory == tuple(initial.evidence[i].evidence_id for i in (0, 1))
    assert results[1].working_memory == tuple(initial.evidence[i].evidence_id for i in (1, 3))
    assert [item["content"] for item in prompts[0]["memory"]] == [initial.evidence[i].content for i in (0, 1)]
    assert [item["content"] for item in prompts[1]["memory"]] == [initial.evidence[i].content for i in (1, 3)]
    assert events[0]["pool_ids"] == events[1]["pool_ids"]
    assert events[0]["scores"] != events[1]["scores"]
    assert events[0]["pool_records"][0]["kind"] == "observation"
    assert events[0]["pool_records"][0]["source"] == "test"


@pytest.mark.parametrize("candidate_count", [1, 4])
def test_history_pool_is_independent_of_candidate_count_and_tail_uses_same_pool(tmp_path, candidate_count):
    initial = archived_state(8)
    selections, pools = [], []
    for selection in ("judge", "tail"):
        judge = ItemCostJudge(scores=(1,) * 6) if selection == "judge" else None
        policy = ResearchPolicy(Model(), judge, RunConfig(method="B3", candidate_count=candidate_count,
            judge_candidates=False, judge_stop=False, options={"memory_records": 4,
                "memory_pool_policy": "history_tail", "memory_pool_records": 6,
                "memory_selection": selection}), events=JSONLEventStore(tmp_path / selection / "events.jsonl"))
        ledger = BudgetLedger()
        selected = policy.select_memory(initial, ledger=ledger)
        selections.append(selected.working_memory)
        pools.append(policy.events.read()[-1].payload["pool_ids"])
        assert selected.evidence == initial.evidence
        assert ledger.used.model_requests == (6 if selection == "judge" else 0)
        if selection == "tail":
            assert policy.events.read()[-1].payload["skip_reason"] == "same_pool_tail_comparator"
    assert pools[0] == pools[1] == tuple(e.evidence_id for e in initial.evidence[-6:])
    assert selections[0] == tuple(e.evidence_id for e in initial.evidence[-6:-2])  # stable old-first ties
    assert selections[1] == tuple(e.evidence_id for e in initial.evidence[-4:])


def test_memory_pool_never_mutates_or_imports_sibling_evidence(tmp_path):
    parent = archived_state(3)
    branches = [record_transition(parent, Candidate(action=Action(name="a")),
        StepResult(observation=Observation(text=name)), branch_id=name) for name in ("left", "right")]
    for index, branch in enumerate(branches):
        policy = ResearchPolicy(Model(), ItemCostJudge(scores=(0, 0, 0, 1)), RunConfig(method="B3",
            options={"memory_records": 1, "memory_pool_policy": "history_tail", "memory_pool_records": 4}),
            events=JSONLEventStore(tmp_path / str(index) / "events.jsonl"))
        selected = policy.select_memory(branch, ledger=BudgetLedger())
        assert selected.working_memory == (branch.evidence[-1].evidence_id,)
        assert branches[1 - index].evidence[-1].evidence_id not in policy.events.read()[-1].payload["pool_ids"]
        assert selected.evidence == branch.evidence
    assert len(parent.evidence) == 3


@pytest.mark.parametrize("options", [
    {"memory_pool_policy": "history_tail"},
    {"memory_pool_policy": "history_tail", "memory_records": 4, "memory_pool_records": 3},
    {"memory_pool_policy": "history_tail", "memory_pool_records": True},
    {"memory_pool_policy": "history_tail", "memory_pool_records": 4.0},
    {"memory_pool_records": 64}, {"memory_selection": "unknown"},
    {"memory_pool_policy": "retained_plus_new"}, {"memory_records": 0},
])
def test_invalid_memory_policies_reject_before_model_calls(options):
    model = Model()
    with pytest.raises(ValueError):
        ResearchPolicy(model, model, RunConfig(options=options))
    assert not model.requests


def test_failed_memory_judge_keeps_archive_and_logs_incomplete_selection(tmp_path):
    class Fails(Model):
        def score(self, request, *, ledger, phase):
            ledger.record(Usage(model_requests=1, input_tokens=7), source="partial-failure", phase=phase, failed=True)
            raise ModelFailure("failed after first item")
    initial = archived_state(5)
    policy = ResearchPolicy(Model(), Fails(), RunConfig(options={"memory_records": 2,
        "memory_pool_policy": "history_tail", "memory_pool_records": 4}),
        events=JSONLEventStore(tmp_path / "events.jsonl"))
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure):
        policy.select_memory(initial, ledger=ledger)
    event = policy.events.read()[-1].payload
    assert event["status"] == "failed" and not event["selection_applied"]
    assert event["selected_ids"] is None and event["evidence_ids"] == initial.working_memory
    assert ledger.used.model_requests == 1 and ledger.used.input_tokens == 7
    assert policy.mechanism_metadata()["memory"]["score_calls_attempted"] == 1
    assert policy.mechanism_metadata()["memory"]["score_calls_completed"] == 0


def test_stop_opinions_never_enter_memory_pool_or_gain_control(tmp_path):
    initial = archived_state(2)
    judge = Model(scores=(1,))
    policy = ResearchPolicy(Model(), judge, RunConfig(options={"memory_records": 4}),
                            events=JSONLEventStore(tmp_path / "events.jsonl"))
    ledger = BudgetLedger()
    advised = policy.assess_completion(initial, ledger=ledger)
    retained = policy.select_memory(advised, ledger=ledger)
    assert len(judge.requests) == 1 and ledger.used.judge_calls == 1
    assert retained.working_memory == initial.working_memory
    assert retained.evidence == advised.evidence and not retained.done and not retained.success
    assert advised.evidence[-1].evidence_id not in policy.events.read()[-1].payload["pool_ids"]
    metadata = policy.mechanism_metadata()["stop"]
    assert metadata["mode"] == "advisory" and metadata["advice_records"] == 1
    assert not metadata["controls_success"] and not metadata["controls_action_or_search"]
