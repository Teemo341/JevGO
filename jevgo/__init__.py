"""JevGO: frozen-model autoresearch contracts and runtime components."""
from .core import (
    Action, BudgetExceeded, BudgetLedger, BudgetLimits, Candidate,
    EnvironmentAdapter, EnvironmentCapabilities, EnvironmentFailure, EnvironmentSnapshot,
    EpisodeSpec, ErrorCategory, EvidenceKind, EvidenceRecord, EventRecord, ExperimentPlan,
    GenerationBackend, GenerationRequest, GenerationResponse, InvalidCandidate,
    JevGOError, JSONLEventStore, JudgeBackend, JudgeRequest, JudgeResponse, JudgeRole,
    ModelFailure, ModelIdentity, Observation, ResearchState, RunConfig, RunResult,
    SearchDomain, SearchState, SearchTransition, StepResult, TerminationReason,
    UnsupportedCapability, Usage, UsagePhase, ValueEstimate,
)

__version__ = "0.1.0"
__all__ = [name for name in globals() if not name.startswith("_")]
