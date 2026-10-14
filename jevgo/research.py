"""AR-2/3/5/7: proposals and evidence, without conflating opinions with facts."""
from __future__ import annotations

import json
from typing import Iterable

from pydantic import ValidationError

from .core import (
    Action, BudgetLedger, Candidate, EpisodeSpec, EvidenceKind, EvidenceRecord,
    EventRecord, ExperimentPlan, GenerationBackend, GenerationRequest,
    InvalidCandidate, JSONLEventStore, JudgeBackend, JudgeRequest, JudgeRole,
    ModelFailure, ResearchState, RunConfig, StepResult, Usage, UsagePhase,
)


def _observation_record(episode: EpisodeSpec, result: StepResult, branch: str,
                        action: Action | None = None) -> EvidenceRecord:
    return EvidenceRecord(
        episode_id=episode.episode_id, branch_id=branch,
        kind=EvidenceKind.OBSERVATION, origin="environment", source=episode.environment,
        content=result.observation.text, raw=result.model_dump(mode="json"), action=action,
    )


def initialize_state(episode: EpisodeSpec, result: StepResult, *,
                     priors: Iterable[str] = ()) -> ResearchState:
    """Record reset feedback exactly; supplied general knowledge remains PRIOR."""
    observed = _observation_record(episode, result, "root")
    knowledge = tuple(EvidenceRecord(
        episode_id=episode.episode_id, branch_id="root", kind=EvidenceKind.PRIOR,
        origin="system", source="initial_knowledge", content=text,
    ) for text in priors)
    return ResearchState(
        episode=episode, observation=result.observation, evidence=knowledge + (observed,),
        fact_ids=(observed.evidence_id,),
        working_memory=tuple(e.evidence_id for e in knowledge + (observed,)),
        last_reward=result.reward, objective_score=result.objective_score,
        terminated=result.terminated, truncated=result.truncated, success=result.success,
    )


def concretize_candidate(candidate: Candidate) -> ExperimentPlan:
    """The first release uses one atomic operation per experiment/search edge."""
    return ExperimentPlan(
        candidate_id=candidate.candidate_id, actions=(candidate.action,),
        expected_observations=(candidate.hypothesis,) if candidate.hypothesis else (),
        failure_criteria=("Environment reports an invalid action or execution error",),
        estimated_usage=Usage(env_steps=1),
    )


def record_transition(state: ResearchState, candidate: Candidate, result: StepResult, *,
                      branch_id: str | None = None,
                      plan: ExperimentPlan | None = None) -> ResearchState:
    """Pure ingestion: callers save this result BEFORE optional model refinement."""
    branch = branch_id or state.branch_id
    plan = plan or concretize_candidate(candidate)
    if plan.candidate_id != candidate.candidate_id or plan.actions != (candidate.action,):
        raise InvalidCandidate("Each experiment must execute exactly its candidate's atomic action")
    records: tuple[EvidenceRecord, ...] = ()
    if candidate.hypothesis:
        records += (EvidenceRecord(
            episode_id=state.episode.episode_id, branch_id=branch,
            kind=EvidenceKind.HYPOTHESIS, origin="model", source="candidate",
            content=candidate.hypothesis, state_id=state.state_id, action=candidate.action,
        ),)
    observed = _observation_record(state.episode, result, branch, candidate.action)
    records += (observed,)
    return state.evolve(
        branch_id=branch, observation=result.observation, evidence=state.evidence + records,
        fact_ids=state.fact_ids + (observed.evidence_id,),
        hypotheses=state.hypotheses + ((candidate.hypothesis,) if candidate.hypothesis else ()),
        working_memory=state.working_memory + tuple(e.evidence_id for e in records),
        actions=state.actions + (candidate.action,), experiment_ids=state.experiment_ids + (plan.plan_id,),
        last_reward=result.reward, objective_score=result.objective_score,
        terminated=result.terminated, truncated=result.truncated, success=result.success,
    )


def public_state(state: ResearchState, *, use_memory: bool = True) -> dict:
    """Whitelist model-visible data; raw scorecards/info and snapshots stay private."""
    selected = set(state.working_memory) if use_memory else set()
    return {
        "task": state.episode.task, "success_criteria": state.episode.success_criteria,
        "observation": state.observation.model_dump(mode="json"),
        "actions": [a.model_dump(mode="json") for a in state.actions],
        "memory": [
            {"id": e.evidence_id, "kind": e.kind.value, "content": e.content,
             "source": e.source, "branch_id": e.branch_id, "references": list(e.references)}
            for e in state.evidence if e.evidence_id in selected
        ],
    }


class ResearchPolicy:
    """Frozen LLM proposals, optional JEV preferences, and traceable memory selection."""

    def __init__(self, generator: GenerationBackend, judge: JudgeBackend | None,
                 config: RunConfig, *, events: JSONLEventStore | None = None) -> None:
        self.generator, self.judge, self.config, self.events = generator, judge, config, events
        if config.method in ("B1", "B3") and any((config.judge_candidates,
                                                  config.judge_evidence, config.judge_stop)) and judge is None:
            raise ValueError("Enabled JEV roles require a judge backend")

    def _enabled(self, role: str) -> bool:
        return self.config.method in ("B1", "B3") and bool(getattr(self.config, "judge_" + role))

    def _event(self, name: str, state: ResearchState, **payload: object) -> None:
        if self.events:
            self.events.append(EventRecord(event_type=name, episode_id=state.episode.episode_id,
                                           branch_id=state.branch_id, payload=payload))

    def _context(self, state: ResearchState) -> str:
        return json.dumps(public_state(state, use_memory=self.config.use_memory), ensure_ascii=False)

    def _scores(self, state: ResearchState, role: JudgeRole, instruction: str,
                items: tuple[str, ...], ledger: BudgetLedger, phase: UsagePhase) -> tuple[float, ...]:
        if self.judge is None:
            raise ModelFailure("No judge backend")
        response = self.judge.score(JudgeRequest(
            role=role, context=instruction + "\nPublic state:\n" + self._context(state),
            items=items, image_paths=state.observation.image_paths,
        ), ledger=ledger, phase=phase)
        if len(response.scores) != len(items):
            raise ModelFailure("Judge scores must correspond one-to-one with items")
        if any(score < 0 for score in response.scores):
            raise ModelFailure("Research policy requires nonnegative preference scores")
        self._event("judge", state, role=role.value, items=list(items),
                    scores=list(response.scores), identity=response.identity.model_dump(mode="json"),
                    metadata=response.metadata)
        return response.scores

    def candidates(self, state: ResearchState, *, ledger: BudgetLedger,
                   phase: UsagePhase = UsagePhase.SEARCH) -> tuple[Candidate, ...]:
        if state.done:
            return ()
        n = 1 if self.config.method == "B0" else self.config.candidate_count
        prompt = (
            'Propose ONE next atomic action as a JSON object with exactly these fields: '
            '{"action":{"name":"...","arguments":{}},"hypothesis":"expected observation",'
            '"rationale":"reason"}. A hypothesis is unverified. Use an admissible action '
            'when the observation provides a list. Do not invent environment results.\nPublic state:\n'
            + self._context(state)
        )
        grouped: dict[str, Candidate] = {}
        allowed = {a.key() for a in state.observation.admissible_actions}
        errors: list[str] = []
        for attempt in range(2):
            response = self.generator.generate(GenerationRequest(
                prompt=prompt, system="You plan experiments using only the supplied public evidence.",
                n=n, seed=self.config.seed + len(state.actions) * 2 + attempt,
                image_paths=state.observation.image_paths,
                max_new_tokens=int(self.config.options.get("max_new_tokens", 256)),
            ), ledger=ledger, phase=phase)
            if len(response.texts) != n:
                raise ModelFailure("Generation response count differs from requested samples")
            for sample in response.texts:
                try:
                    text = sample.strip()
                    if text.startswith("```json\n") and text.endswith("```"):
                        text = text[8:-3].strip()
                    obj = json.loads(text)
                    if not isinstance(obj, dict) or set(obj) - {"action", "hypothesis", "rationale"}:
                        raise ValueError("Unexpected proposal fields")
                    candidate = Candidate(**obj)
                    key = candidate.action.key()
                    if allowed and key not in allowed:
                        raise ValueError("Action is not in the public admissible action set")
                    if key in grouped:
                        previous = grouped[key]
                        grouped[key] = previous.model_copy(update={"sample_count": previous.sample_count + 1})
                    else:
                        grouped[key] = candidate
                except (ValueError, TypeError, ValidationError) as exc:
                    errors.append(str(exc))
            self._event("proposal_samples", state, attempt=attempt, texts=list(response.texts),
                        errors=list(errors), identity=response.identity.model_dump(mode="json"))
            if grouped:
                break
            prompt += "\nThe previous proposals were invalid. Return only the required JSON object."
        if not grouped:
            raise InvalidCandidate("No valid proposals after one repair attempt: " + "; ".join(errors))
        values = tuple(grouped.values())
        total = sum(c.sample_count for c in values)
        priors = tuple(c.sample_count / total for c in values)
        scores: tuple[float, ...] | None = None
        if self._enabled("candidates"):
            scores = self._scores(
                state, JudgeRole.CANDIDATE,
                "Compare the plausibility and usefulness of proposed next actions. These are hypotheses, not facts.",
                tuple(json.dumps({"action": c.action.model_dump(mode="json"), "hypothesis": c.hypothesis})
                      for c in values), ledger, phase,
            )
            score_sum = sum(scores)
            preferences = tuple(s / score_sum for s in scores) if score_sum else tuple(1 / len(scores) for _ in scores)
            priors = tuple((p + q) / 2 for p, q in zip(priors, preferences))
        candidates = tuple(c.model_copy(update={"prior": p, "judge_score": scores[i] if scores else None})
                           for i, (c, p) in enumerate(zip(values, priors)))
        self._event("candidates", state, candidates=[c.model_dump(mode="json") for c in candidates],
                    prior_semantics="half_frequency_half_judge" if scores else "sample_frequency")
        return candidates

    def select_memory(self, state: ResearchState, *, ledger: BudgetLedger,
                      phase: UsagePhase = UsagePhase.SEARCH) -> ResearchState:
        # Selection operates on immutable, existing records; no model-written facts are accepted.
        records = tuple(e for e in state.evidence if e.kind != EvidenceKind.JUDGE_OPINION)
        if not self.config.use_memory:
            return state.evolve(working_memory=())
        maximum = self.config.options.get("memory_records", 32)
        if type(maximum) is not int or maximum < 1:
            raise ValueError("memory_records must be a positive integer")
        selected = records[-maximum:]
        if records and self._enabled("evidence"):
            # Bound individual scoring calls; older raw observations remain in the event/evidence archive.
            pool = records[-max(maximum, self.config.candidate_count):]
            scores = self._scores(
                state, JudgeRole.EVIDENCE,
                "Rate evidence relevance to the task. Failed experiments and counterevidence can be useful. "
                "Do not rate agreement with a preferred hypothesis.",
                tuple(e.content for e in pool), ledger, phase,
            )
            indices = sorted(range(len(pool)), key=lambda i: (-scores[i], i))[:maximum]
            selected = tuple(pool[i] for i in sorted(indices))
        successor = state.evolve(working_memory=tuple(e.evidence_id for e in selected))
        self._event("memory_selection", successor, evidence_ids=list(successor.working_memory),
                    raw_evidence_count=len(successor.evidence))
        return successor

    def assess_completion(self, state: ResearchState, *, ledger: BudgetLedger,
                          phase: UsagePhase = UsagePhase.SEARCH) -> ResearchState:
        if state.done or not self._enabled("stop"):
            return state
        scores = self._scores(
            state, JudgeRole.STOP,
            "Assess whether the public evidence is sufficient to answer the task. This advice cannot "
            "override the benchmark's objective success signal.",
            ("The available evidence is sufficient to finish the task.",), ledger, phase,
        )
        opinion = EvidenceRecord(
            episode_id=state.episode.episode_id, branch_id=state.branch_id,
            kind=EvidenceKind.JUDGE_OPINION, origin="model", source="jev_stop",
            content="Completion advice (uncalibrated): " + str(scores[0]),
            raw={"score": scores[0], "objective_success": state.success},
            references=state.fact_ids, state_id=state.state_id,
        )
        successor = state.evolve(evidence=state.evidence + (opinion,))
        self._event("completion_advice", successor, score=scores[0], objective_success=state.success)
        return successor
