"""AR-0..7: a frozen-model research loop with distinct search and commit traces."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import traceback
from typing import Any

from .core import (
    BudgetExceeded, BudgetLedger, Candidate, EnvironmentAdapter, EnvironmentFailure,
    EpisodeSpec, EventRecord, GenerationBackend, InvalidCandidate, JSONLEventStore,
    JudgeBackend, ModelFailure, Observation, ResearchState, RunConfig, RunResult,
    SearchState, SearchTransition, TerminationReason, UnsupportedCapability,
    Usage, UsagePhase, ValueEstimate, new_id,
)
from .research import ResearchPolicy, concretize_candidate, initialize_state, record_transition
from .search.mcts import MCTS


def runtime_versions() -> dict:
    packages = {}
    for name in ("pydantic", "minigrid", "alfworld", "scienceworld", "discoveryworld",
                 "torch", "transformers", "semif-phase1", "vllm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                     cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "commit": commit, "tracked_files_dirty": dirty, "packages": packages}


def _write_json(path: Path, value: Any) -> None:
    import json
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class ResearchDomain:
    """Own restoration and all transitions; backends/adapters own their own charges."""
    def __init__(self, environment: EnvironmentAdapter, policy: ResearchPolicy,
                 ledger: BudgetLedger, events: JSONLEventStore) -> None:
        self.environment, self.policy, self.ledger, self.events = environment, policy, ledger, events
        self.current_state_id: str | None = None
        self.last_committed: ResearchState | None = None
        self._pending_search_budget: BudgetExceeded | None = None

    @property
    def capabilities(self):
        return self.environment.capabilities

    def _check_interruption(self, phase):
        if UsagePhase(phase) == UsagePhase.SEARCH and self._pending_search_budget is not None:
            raise self._pending_search_budget

    def candidates(self, state: SearchState, *, phase=UsagePhase.SEARCH):
        self._check_interruption(phase)
        return self.policy.candidates(state.research, ledger=self.ledger, phase=phase)

    def candidates_for_scope(self, state: SearchState, *, scope, phase=UsagePhase.SEARCH):
        self._check_interruption(phase)
        return self.policy.candidates(state.research, ledger=self.ledger, phase=phase,
                                      proposal_scope=scope)

    def transition(self, state: SearchState, candidate: Candidate, *, phase=UsagePhase.SEARCH):
        phase = UsagePhase(phase)
        self._check_interruption(phase)
        if phase == UsagePhase.COMMIT:
            self._pending_search_budget = None
        if phase == UsagePhase.SEARCH:
            self.capabilities.require_branching()
        if state.terminal:
            raise EnvironmentFailure("Cannot act from a terminal research state")
        if self.current_state_id != state.research.state_id:
            if state.snapshot is None:
                raise UnsupportedCapability("Restoration requires the parent state's snapshot")
            self.environment.restore(state.snapshot, ledger=self.ledger, phase=phase)
            self.current_state_id = state.research.state_id
        plan = concretize_candidate(candidate)
        step = self.environment.step(plan.actions[0], ledger=self.ledger, phase=phase)
        branch = "root" if phase == UsagePhase.COMMIT else state.research.branch_id + "/" + new_id()[:12]
        successor = record_transition(state.research, candidate, step, branch_id=branch, plan=plan)
        # The action has happened. Preserve it even if later optional scoring fails.
        if phase == UsagePhase.COMMIT:
            self.last_committed = successor
        self.current_state_id = successor.state_id
        self.events.append(EventRecord(event_type="transition", episode_id=successor.episode.episode_id,
            branch_id=branch, payload={"phase": phase.value, "parent_state_id": state.research.state_id,
                "candidate": candidate.model_dump(mode="json"), "plan": plan.model_dump(mode="json"),
                "step": step.model_dump(mode="json"), "state": successor.model_dump(mode="json")}))
        snapshot = None
        try:
            if not successor.done:
                successor = self.policy.select_memory(successor, ledger=self.ledger, phase=phase)
                if phase == UsagePhase.COMMIT:
                    self.last_committed = successor
                successor = self.policy.assess_completion(successor, ledger=self.ledger, phase=phase)
            self.current_state_id = successor.state_id
            if phase == UsagePhase.COMMIT:
                self.last_committed = successor
            if self.policy.config.method in ("B2", "B3") and self.capabilities.can_branch and not successor.done:
                snapshot = self.environment.snapshot(ledger=self.ledger, phase=phase)
        except BudgetExceeded as exc:
            if phase != UsagePhase.SEARCH:
                raise
            # Return the already observed edge first. On the next candidates or
            # evaluate call MCTS receives this interruption and performs exactly
            # one partial backup, including this edge's known reward. No further
            # search transition may run from the unsnapshotted successor.
            self._pending_search_budget = exc
            self.events.append(EventRecord(event_type="search_postprocessing_interrupted",
                episode_id=successor.episode.episode_id, branch_id=branch,
                payload={"state_id": successor.state_id, "known_reward": step.reward,
                         "resources": list(exc.resources)}))
        return SearchTransition(SearchState(successor, snapshot), step.reward)

    def evaluate(self, state: SearchState, *, phase=UsagePhase.SEARCH):
        self._check_interruption(phase)
        # Rollout rewards are accounted on edges. This explicit truncation bootstrap
        # never passes JEV plausibility off as a learned long-term value estimate.
        return ValueEstimate(value=0, source="environment" if state.terminal else "heuristic",
            rationale="Terminal continuation is zero" if state.terminal else "Zero bootstrap after bounded real rollout")


def commit_reserve(episode: EpisodeSpec, state: ResearchState, config: RunConfig) -> Usage:
    """Conservative reset/replay + atomic commit + successor snapshot allowance.

    Even when the current branch happens to equal root, retain the full allowance.
    Restore is never paid from search's protected reserve. Wall time is measured;
    only caller-provided time allowance can be reserved in advance.
    """
    h = len(state.actions)
    name = episode.environment.lower()
    if name not in ("minigrid", "alfworld", "scienceworld", "discoveryworld"):
        raise UnsupportedCapability(f"No verified restore-cost contract for {name}")
    reset_action = int(name == "scienceworld")
    observations = 8 if reset_action else 1
    needed = Usage(env_steps=h + 1 + reset_action, replay_steps=h + reset_action,
        env_resets=1, env_restores=1, env_snapshots=1, env_observations=(h + 2) * observations)
    return Usage(**{key: max(getattr(needed, key), getattr(config.commit_reserve, key)) for key in Usage.model_fields})


def _reason(state: ResearchState, default: TerminationReason) -> TerminationReason:
    if state.success:
        return TerminationReason.SUCCESS
    if state.truncated:
        return TerminationReason.ENVIRONMENT_TRUNCATED
    if state.terminated:
        return TerminationReason.ENVIRONMENT_TERMINATED
    return default


def run_episode(episode: EpisodeSpec, config: RunConfig, environment: EnvironmentAdapter,
                generator: GenerationBackend, judge: JudgeBackend | None = None, *,
                output_dir: Path | str, metadata: dict | None = None) -> RunResult:
    """Run B0/B1/B2/B3; output_dir must be new. All supplied resources are closed.

    B0 samples one sequential proposal. B1 selects maximum JEV score (stable
    frequency tie-break); B2/B3 select maximum root visit count from MCTS. JEV
    completion remains advice: only environment objective success ends in success.
    """
    output = Path(output_dir).resolve()
    events = None
    state = ResearchState(episode=episode, observation=Observation(text="Environment has not reset"))
    ledger = BudgetLedger(config.budget)
    reason, error, domain = TerminationReason.MAX_STEPS, None, None
    cleanup_errors: list[str] = []
    search_decisions = 0
    initialized = False
    try:
        output.mkdir(parents=True, exist_ok=False)
        initialized = True
        events = JSONLEventStore(output / "events.jsonl")
        ledger = BudgetLedger(config.budget, event_store=events)
        _write_json(output / "config.json", {"episode": episode.model_dump(mode="json"),
            "run": config.model_dump(mode="json"), "metadata": metadata or {},
            "generator": generator.identity.model_dump(mode="json"),
            "judge": judge.identity.model_dump(mode="json") if judge else None})
        _write_json(output / "versions.json", runtime_versions())
        policy = ResearchPolicy(generator, judge, config, events=events)
        domain = ResearchDomain(environment, policy, ledger, events)
        initial = environment.reset(episode, ledger=ledger, phase=UsagePhase.COMMIT)
        state = initialize_state(episode, initial)
        domain.last_committed = state
        domain.current_state_id = state.state_id
        events.append(EventRecord(event_type="initial_state", episode_id=episode.episode_id,
                                 branch_id="root", payload={"state": state.model_dump(mode="json")}))
        search = config.method in ("B2", "B3")
        if search:
            environment.capabilities.require_branching()
        snapshot = environment.snapshot(ledger=ledger, phase=UsagePhase.COMMIT) if search and not state.done else None
        root = SearchState(state, snapshot)
        for _ in range(config.max_steps):
            if root.terminal:
                break
            if search:
                ledger.reserve_commit(commit_reserve(episode, root.research, config))
                result = MCTS(config).search(domain, root)
                search_decisions += 1
                events.append(EventRecord(event_type="search", episode_id=episode.episode_id,
                    branch_id="root", payload={"state_id": root.research.state_id,
                                               "result": result.model_dump(mode="json"),
                                               "root_candidate_edges": len(result.rootstats),
                                               "root_visited_edges": sum(edge.N > 0 for edge in result.rootstats)}))
                candidate = result.candidate
                if candidate is None:
                    reason = (TerminationReason.BUDGET_EXHAUSTED if result.stop_reason == "budget_exhausted"
                              else TerminationReason.NO_CANDIDATES)
                    break
            else:
                candidates = domain.candidates_for_scope(root, scope="root", phase=UsagePhase.COMMIT)
                if not candidates:
                    reason = TerminationReason.NO_CANDIDATES
                    break
                if config.method == "B1" and config.judge_candidates:
                    candidate = max(candidates, key=lambda c: (c.judge_score if c.judge_score is not None else 0, c.prior))
                else:
                    candidate = max(candidates, key=lambda c: c.prior)
            # Never reuse a searched successor as the committed real trajectory.
            root = domain.transition(root, candidate, phase=UsagePhase.COMMIT).state
            state = root.research
            ledger.reserve_commit(Usage())
        reason = _reason(state, reason)
    except Exception as exc:
        if not initialized:
            raise  # An existing output directory is never reused or overwritten.
        if domain is not None and domain.last_committed is not None:
            state = domain.last_committed
        if isinstance(exc, BudgetExceeded):
            reason = TerminationReason.BUDGET_EXHAUSTED
        elif isinstance(exc, ModelFailure):
            reason = TerminationReason.MODEL_FAILURE
        elif isinstance(exc, InvalidCandidate):
            reason = TerminationReason.INVALID_CANDIDATES
        else:
            reason = TerminationReason.ENVIRONMENT_FAILURE
        reason = _reason(state, reason)
        error = f"{type(exc).__name__}: {exc}"
        if events:
            events.append(EventRecord(event_type="error", episode_id=episode.episode_id,
                payload={"error": error, "traceback": traceback.format_exc()}))
    except KeyboardInterrupt:
        if domain is not None and domain.last_committed is not None:
            state = domain.last_committed
        reason = _reason(state, TerminationReason.CANCELLED)
        error = "KeyboardInterrupt"
    finally:
        closed = set()
        for resource in (environment, generator, judge):
            if resource is not None and id(resource) not in closed:
                closed.add(id(resource))
                try:
                    resource.close()
                except Exception as exc:
                    cleanup_errors.append(f"{type(resource).__name__}: {exc}")
    result = RunResult(episode=episode, config=config, final_state=state, termination_reason=reason,
        usage=ledger.used, answer=state.observation.text, error=error,
        events_path=str(output / "events.jsonl"), metadata={**(metadata or {}),
            "generator": generator.identity.model_dump(mode="json"),
            "judge": judge.identity.model_dump(mode="json") if judge else None,
            "committed_steps": len(state.actions), "search_decisions": search_decisions,
            "over_budget": list(ledger.over_budget), "cleanup_errors": cleanup_errors,
            "wall_seconds_scope": "instrumented_operations_plus_search_overhead",
            "validation_level": (metadata or {}).get("validation_level", "unclassified")})
    _write_json(output / "ledger.json", ledger.summary())
    _write_json(output / "result.json", result.model_dump(mode="json"))
    if events:
        events.append(EventRecord(event_type="result", episode_id=episode.episode_id,
                                 payload={"result": result.model_dump(mode="json")}))
    return result
