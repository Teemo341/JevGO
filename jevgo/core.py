"""Version-one contracts for frozen-model autoresearch.

Only adapters charge environment operations; only backends charge model work.
Search and committed trajectories share one monotonic BudgetLedger.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, model_validator


def new_id() -> str:
    return uuid4().hex


class FrozenDict(Mapping[str, Any]):
    """An owned, recursively immutable JSON object, including nested arrays."""
    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_values", MappingProxyType({k: _freeze_json(v) for k, v in values.items()}))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("JSON objects are immutable")

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self)!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        return self


def _freeze_json(value: Any) -> Any:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return FrozenDict(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    raise ValueError(f"Expected finite JSON data, got {type(value).__name__}")


def _freeze_object(value: Any) -> FrozenDict:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a JSON object")
    return _freeze_json(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


JSONObject = Annotated[Any, BeforeValidator(_freeze_object), PlainSerializer(_thaw_json)]
JSONValue = Annotated[Any, BeforeValidator(_freeze_json), PlainSerializer(_thaw_json)]
Count = Annotated[int, Field(ge=0, strict=True)]
Finite = Annotated[float, Field(allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", validate_default=True)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Any:
        # Pydantic's default copy(update=...) bypasses validation and deep freezing.
        data = self.model_dump(mode="python")
        data.update(update or {})
        return type(self).model_validate(data)


class UsagePhase(str, Enum):
    COMMIT = "commit"
    SEARCH = "search"
    REPLAY = "replay"


class TerminationReason(str, Enum):
    SUCCESS = "success"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ENVIRONMENT_TERMINATED = "environment_terminated"
    ENVIRONMENT_TRUNCATED = "environment_truncated"
    ENVIRONMENT_FAILURE = "environment_failure"
    MODEL_FAILURE = "model_failure"
    INVALID_CANDIDATES = "invalid_candidates"
    NO_CANDIDATES = "no_candidates"
    MAX_STEPS = "max_steps"
    CANCELLED = "cancelled"


class ErrorCategory(str, Enum):
    BUDGET_EXHAUSTED = "budget_exhausted"
    ENVIRONMENT_FAILURE = "environment_failure"
    MODEL_FAILURE = "model_failure"
    INVALID_CANDIDATE = "invalid_candidate"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    CONFIGURATION = "configuration"


class JevGOError(RuntimeError):
    category = ErrorCategory.CONFIGURATION


class BudgetExceeded(JevGOError):
    category = ErrorCategory.BUDGET_EXHAUSTED

    def __init__(self, resources: tuple[str, ...], phase: UsagePhase) -> None:
        self.resources, self.phase = resources, phase
        super().__init__(f"Budget exhausted ({phase.value}): {', '.join(resources)}")


class EnvironmentFailure(JevGOError):
    category = ErrorCategory.ENVIRONMENT_FAILURE


class ModelFailure(JevGOError):
    category = ErrorCategory.MODEL_FAILURE


class InvalidCandidate(JevGOError):
    category = ErrorCategory.INVALID_CANDIDATE


class UnsupportedCapability(JevGOError):
    category = ErrorCategory.UNSUPPORTED_CAPABILITY


class Usage(FrozenModel):
    env_steps: Count = 0
    env_resets: Count = 0
    env_snapshots: Count = 0
    env_restores: Count = 0
    env_observations: Count = 0
    replay_steps: Count = 0
    generation_calls: Count = 0
    judge_calls: Count = 0
    model_requests: Count = 0
    input_tokens: Count = 0
    output_tokens: Count = 0
    rollouts: Count = 0
    wall_seconds: Nonnegative = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(**{key: getattr(self, key) + getattr(other, key) for key in type(self).model_fields})


class BudgetLimits(FrozenModel):
    """None is unlimited; zero forbids that resource."""
    env_steps: Count | None = None
    env_resets: Count | None = None
    env_snapshots: Count | None = None
    env_restores: Count | None = None
    env_observations: Count | None = None
    replay_steps: Count | None = None
    generation_calls: Count | None = None
    judge_calls: Count | None = None
    model_requests: Count | None = None
    input_tokens: Count | None = None
    output_tokens: Count | None = None
    rollouts: Count | None = None
    wall_seconds: Nonnegative | None = None


class EpisodeSpec(FrozenModel):
    episode_id: str = Field(default_factory=new_id)
    environment: str
    task: str
    seed: int = 0
    split: str = "development"
    success_criteria: str = "Environment reports objective success."
    options: JSONObject = Field(default_factory=dict)


class RunConfig(FrozenModel):
    learning_mode: Literal["none"] = "none"
    method: Literal["B0", "B1", "B2", "B3"] = "B3"
    seed: int = 0
    candidate_count: int = Field(default=4, ge=1)
    simulations: int = Field(default=8, ge=0)
    max_depth: int = Field(default=4, ge=1)
    rollout_depth: int = Field(default=2, ge=0)
    max_steps: int = Field(default=100, ge=1)
    c_puct: Nonnegative = 1.5
    gamma: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    budget: BudgetLimits = Field(default_factory=lambda: BudgetLimits(env_steps=1000))
    commit_reserve: Usage = Field(default_factory=lambda: Usage(env_steps=1))
    judge_candidates: bool = True
    judge_evidence: bool = True
    judge_stop: bool = True
    use_memory: bool = True
    options: JSONObject = Field(default_factory=dict)


class Action(FrozenModel):
    """One adapter-defined atomic action; name and arguments are public."""
    name: str = Field(min_length=1)
    arguments: JSONObject = Field(default_factory=dict)

    def key(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


class Observation(FrozenModel):
    """Only agent-visible data. Hidden world state belongs in a snapshot."""
    text: str = ""
    data: JSONObject = Field(default_factory=dict)
    admissible_actions: tuple[Action, ...] = ()
    image_paths: tuple[str, ...] = ()


class StepResult(FrozenModel):
    observation: Observation
    action_success: bool | None = None
    reward: Finite = 0.0
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    objective_score: Finite | None = None
    info: JSONObject = Field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated or self.success


class EvidenceKind(str, Enum):
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    JUDGE_OPINION = "judge_opinion"
    PRIOR = "prior"


class EvidenceRecord(FrozenModel):
    evidence_id: str = Field(default_factory=new_id)
    episode_id: str
    branch_id: str
    kind: EvidenceKind
    origin: Literal["environment", "model", "user", "system"]
    source: str
    content: str
    raw: JSONObject = Field(default_factory=dict)
    state_id: str | None = None
    action: Action | None = None
    references: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_origin(self) -> EvidenceRecord:
        if self.kind == EvidenceKind.OBSERVATION and self.origin != "environment":
            raise ValueError("Observed evidence must originate in the environment")
        if self.kind == EvidenceKind.JUDGE_OPINION and self.origin != "model":
            raise ValueError("Judge opinions must originate in a model")
        return self


class ResearchState(FrozenModel):
    episode: EpisodeSpec
    observation: Observation
    state_id: str = Field(default_factory=new_id)
    parent_id: str | None = None
    version: Count = 0
    branch_id: str = "root"
    evidence: tuple[EvidenceRecord, ...] = ()
    fact_ids: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    inferences: tuple[str, ...] = ()
    working_memory: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    actions: tuple[Action, ...] = ()
    experiment_ids: tuple[str, ...] = ()
    last_reward: Finite = 0.0
    objective_score: Finite | None = None
    terminated: bool = False
    truncated: bool = False
    success: bool = False

    @model_validator(mode="after")
    def validate_evidence(self) -> ResearchState:
        by_id = {item.evidence_id: item for item in self.evidence}
        if len(by_id) != len(self.evidence):
            raise ValueError("Evidence identifiers must be unique")
        if any(item.episode_id != self.episode.episode_id for item in self.evidence):
            raise ValueError("Evidence must belong to this episode")
        for evidence_id in self.fact_ids:
            if evidence_id not in by_id or by_id[evidence_id].kind != EvidenceKind.OBSERVATION:
                raise ValueError("Facts must reference retained environment observations")
        return self

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated or self.success

    def evolve(self, **changes: Any) -> ResearchState:
        """Create a versioned successor, preserving the complete evidence prefix."""
        reserved = {"state_id", "parent_id", "version", "episode"}.intersection(changes)
        if reserved:
            raise ValueError(f"Cannot override successor identity fields: {sorted(reserved)}")
        successor = self.model_copy(update={**changes, "state_id": new_id(), "parent_id": self.state_id,
                                            "version": self.version + 1})
        if successor.evidence[:len(self.evidence)] != self.evidence:
            raise ValueError("Raw evidence is append-only; compress working_memory instead")
        return successor


class Candidate(FrozenModel):
    candidate_id: str = Field(default_factory=new_id)
    action: Action
    hypothesis: str = ""
    rationale: str = ""
    prior: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)
    sample_count: int = Field(default=1, ge=1)
    judge_score: Finite | None = None


class ExperimentPlan(FrozenModel):
    plan_id: str = Field(default_factory=new_id)
    candidate_id: str
    actions: tuple[Action, ...] = Field(min_length=1)
    preconditions: tuple[str, ...] = ()
    expected_observations: tuple[str, ...] = ()
    failure_criteria: tuple[str, ...] = ()
    estimated_usage: Usage = Field(default_factory=Usage)


class EnvironmentCapabilities(FrozenModel):
    snapshot_restore: bool = False
    deterministic_replay: bool = False
    partial_observability: bool = True
    objective_success: bool = True
    details: JSONObject = Field(default_factory=dict)

    @property
    def can_branch(self) -> bool:
        return self.snapshot_restore or self.deterministic_replay

    def require_branching(self) -> None:
        if not self.can_branch:
            raise UnsupportedCapability("MCTS requires verified snapshot/restore or deterministic replay")


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Opaque adapter-owned handle. Never serialize it or include it in prompts."""
    owner_id: str
    payload: Any = field(repr=False, compare=False)
    snapshot_id: str = field(default_factory=new_id)


class ModelIdentity(FrozenModel):
    backend: str
    model: str
    revision: str | None = None
    adapter: str | None = None
    mode: str = "frozen"
    metadata: JSONObject = Field(default_factory=dict)


class GenerationRequest(FrozenModel):
    prompt: str
    system: str = ""
    n: int = Field(default=1, ge=1)
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: Nonnegative = 0.7
    seed: int = 0
    image_paths: tuple[str, ...] = ()
    options: JSONObject = Field(default_factory=dict)


class GenerationResponse(FrozenModel):
    texts: tuple[str, ...]
    identity: ModelIdentity
    usage: Usage
    metadata: JSONObject = Field(default_factory=dict)


class JudgeRole(str, Enum):
    CANDIDATE = "candidate"
    EVIDENCE = "evidence"
    STATE_VALUE = "state_value"
    STOP = "stop"


class JudgeRequest(FrozenModel):
    role: JudgeRole
    context: str
    items: tuple[str, ...] = Field(min_length=1)
    image_paths: tuple[str, ...] = ()
    options: JSONObject = Field(default_factory=dict)


class JudgeResponse(FrozenModel):
    """Higher scores mean greater preference. Scores are not objective success."""
    scores: tuple[Finite, ...]
    identity: ModelIdentity
    usage: Usage
    rationale: tuple[str, ...] = ()
    metadata: JSONObject = Field(default_factory=dict)


class EventRecord(FrozenModel):
    event_id: str = Field(default_factory=new_id)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str
    episode_id: str | None = None
    branch_id: str | None = None
    payload: JSONObject = Field(default_factory=dict)
    schema_version: Literal[1] = 1


class JSONLEventStore:
    """Append-only event stream. Reads fail explicitly on malformed/truncated rows."""
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def append(self, event: EventRecord) -> None:
        row = event.model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(row)
                stream.flush()

    def read(self) -> tuple[EventRecord, ...]:
        with self._lock:
            if not self.path.exists():
                return ()
            events = []
            with self.path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        events.append(EventRecord.model_validate_json(line))
                    except ValueError as error:
                        raise ValueError(f"Invalid JSONL event at {self.path}:{line_number}") from error
            return tuple(events)


class LedgerEntry(FrozenModel):
    sequence: int = Field(ge=1)
    source: str
    phase: UsagePhase
    budget_phase: Literal["commit", "search"]
    usage: Usage
    failed: bool = False
    metadata: JSONObject = Field(default_factory=dict)


class BudgetLedger:
    """One shared, monotonic ledger, intentionally absent from environment snapshots.

    charge() atomically admits and charges known costs BEFORE an operation.
    record() records measured costs AFTER an operation, including overages; it
    never discards actual work. Preflight unknown work using ensure_available().
    Failed work is retained. Restore/reset must never replace this ledger.
    """
    def __init__(self, limits: BudgetLimits | None = None, *, commit_reserve: Usage | None = None,
                 event_store: JSONLEventStore | None = None) -> None:
        self.limits = limits or BudgetLimits()
        self._used = Usage()
        self._entries: list[LedgerEntry] = []
        self._reserve = Usage()
        self._lock = RLock()
        self.event_store = event_store
        self.reserve_commit(commit_reserve or Usage())

    @staticmethod
    def _budget_phase(phase: UsagePhase | str, budget_phase: str | None) -> Literal["commit", "search"]:
        phase = UsagePhase(phase)
        chosen = budget_phase or ("commit" if phase == UsagePhase.COMMIT else "search")
        if chosen not in ("commit", "search"):
            raise ValueError("budget_phase must be commit or search")
        if phase != UsagePhase.REPLAY and chosen != phase.value:
            raise ValueError("Only replay entries may inherit another operation's budget phase")
        return chosen

    @property
    def used(self) -> Usage:
        with self._lock:
            return self._used

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    @property
    def commit_reserve(self) -> Usage:
        with self._lock:
            return self._reserve

    def reserve_commit(self, usage: Usage) -> None:
        """Replace the protected allowance; this does not spend or refund usage."""
        with self._lock:
            unavailable = tuple(key for key in Usage.model_fields
                                if getattr(self.limits, key) is not None
                                and getattr(self._used, key) + getattr(usage, key) > getattr(self.limits, key))
            if unavailable:
                raise BudgetExceeded(unavailable, UsagePhase.COMMIT)
            self._reserve = usage

    def remaining(self, *, phase: UsagePhase | str = UsagePhase.COMMIT,
                  budget_phase: str | None = None) -> dict[str, int | float | None]:
        selected = self._budget_phase(phase, budget_phase)
        with self._lock:
            return {key: None if getattr(self.limits, key) is None else max(
                0, getattr(self.limits, key) - getattr(self._used, key)
                - (getattr(self._reserve, key) if selected == "search" else 0)
            ) for key in Usage.model_fields}

    def ensure_available(self, usage: Usage, *, phase: UsagePhase | str = UsagePhase.COMMIT,
                         budget_phase: str | None = None) -> None:
        with self._lock:
            remaining = self.remaining(phase=phase, budget_phase=budget_phase)
            unavailable = tuple(key for key, value in remaining.items()
                                if value is not None and getattr(usage, key) > value)
            if unavailable:
                raise BudgetExceeded(unavailable, UsagePhase(phase))

    def can_afford(self, usage: Usage, *, phase: UsagePhase | str = UsagePhase.COMMIT,
                   budget_phase: str | None = None) -> bool:
        try:
            self.ensure_available(usage, phase=phase, budget_phase=budget_phase)
        except BudgetExceeded:
            return False
        return True

    def charge(self, usage: Usage, *, source: str, phase: UsagePhase | str = UsagePhase.COMMIT,
               budget_phase: str | None = None, failed: bool = False,
               metadata: Mapping[str, Any] | None = None) -> LedgerEntry:
        with self._lock:
            self.ensure_available(usage, phase=phase, budget_phase=budget_phase)
            return self.record(usage, source=source, phase=phase, budget_phase=budget_phase,
                               failed=failed, metadata=metadata)

    def record(self, usage: Usage, *, source: str, phase: UsagePhase | str = UsagePhase.COMMIT,
               budget_phase: str | None = None, failed: bool = False,
               metadata: Mapping[str, Any] | None = None) -> LedgerEntry:
        with self._lock:
            entry = LedgerEntry(sequence=len(self._entries) + 1, source=source, phase=phase,
                                budget_phase=self._budget_phase(phase, budget_phase), usage=usage,
                                failed=failed, metadata=metadata or {})
            self._used = self._used + usage
            self._entries.append(entry)
            if self.event_store is not None:
                self.event_store.append(EventRecord(event_type="usage", payload=entry.model_dump(mode="json")))
            return entry

    @property
    def over_budget(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(key for key in Usage.model_fields if getattr(self.limits, key) is not None
                         and getattr(self._used, key) > getattr(self.limits, key))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {"limits": self.limits.model_dump(mode="json"), "used": self._used.model_dump(mode="json"),
                    "commit_reserve": self._reserve.model_dump(mode="json"),
                    "entries": [entry.model_dump(mode="json") for entry in self._entries],
                    "over_budget": list(self.over_budget)}


@runtime_checkable
class EnvironmentAdapter(Protocol):
    @property
    def capabilities(self) -> EnvironmentCapabilities: ...
    def reset(self, spec: EpisodeSpec, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.COMMIT) -> StepResult: ...
    def step(self, action: Action, *, ledger: BudgetLedger,
             phase: UsagePhase = UsagePhase.COMMIT) -> StepResult: ...
    def snapshot(self, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> EnvironmentSnapshot: ...
    def restore(self, snapshot: EnvironmentSnapshot, *, ledger: BudgetLedger,
                phase: UsagePhase = UsagePhase.SEARCH) -> StepResult: ...
    def observe(self) -> Observation: ...
    def close(self) -> None: ...


@runtime_checkable
class GenerationBackend(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...
    def generate(self, request: GenerationRequest, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> GenerationResponse: ...
    def close(self) -> None: ...


@runtime_checkable
class JudgeBackend(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...
    def score(self, request: JudgeRequest, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.SEARCH) -> JudgeResponse: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SearchState:
    research: ResearchState
    snapshot: EnvironmentSnapshot | None = field(default=None, repr=False)

    @property
    def terminal(self) -> bool:
        return self.research.done


@dataclass(frozen=True, slots=True)
class SearchTransition:
    state: SearchState
    reward: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward):
            raise ValueError("Transition reward must be finite")


class ValueEstimate(FrozenModel):
    """Additional future return, excluding the reward on the incoming edge."""
    value: Finite
    source: Literal["environment", "heuristic"]
    rationale: str = ""


@runtime_checkable
class SearchDomain(Protocol):
    @property
    def ledger(self) -> BudgetLedger: ...
    @property
    def capabilities(self) -> EnvironmentCapabilities: ...
    def candidates(self, state: SearchState, *,
                   phase: UsagePhase = UsagePhase.SEARCH) -> tuple[Candidate, ...]: ...
    def transition(self, state: SearchState, candidate: Candidate, *,
                   phase: UsagePhase = UsagePhase.SEARCH) -> SearchTransition:
        """Restore this state's snapshot, execute ONE action, snapshot successor.

        This method owns restoration; MCTS must never restore a second time.
        It may skip restoration only when the adapter is known to be at this
        exact state. Nonrecoverable domains reject SEARCH before any mutation.
        """
        ...
    def evaluate(self, state: SearchState, *,
                 phase: UsagePhase = UsagePhase.SEARCH) -> ValueEstimate:
        """Estimate future return without executing environment actions.

        Terminal states return zero; their outcome is already on the incoming
        edge. Real rollout consists of repeated transition() calls by MCTS.
        """
        ...


class RunResult(FrozenModel):
    episode: EpisodeSpec
    config: RunConfig
    final_state: ResearchState
    termination_reason: TerminationReason
    usage: Usage
    answer: str = ""
    error: str | None = None
    events_path: str | None = None
    metadata: JSONObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_outcome(self) -> RunResult:
        if self.final_state.episode != self.episode:
            raise ValueError("Run result and final state must use the same episode")
        if (self.termination_reason == TerminationReason.SUCCESS) != self.final_state.success:
            raise ValueError("Success termination must match the objective environment signal")
        return self

    @property
    def success(self) -> bool:
        return self.termination_reason == TerminationReason.SUCCESS
