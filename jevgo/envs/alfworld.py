"""ALFWorld TextWorld adapter with public text and verified deterministic replay."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from uuid import uuid4

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, EnvironmentCapabilities, EnvironmentFailure,
    EnvironmentSnapshot, EpisodeSpec, Observation, StepResult, UnsupportedCapability,
    Usage, UsagePhase,
)


@dataclass(frozen=True)
class _Checkpoint:
    history: tuple[tuple[Action, str], ...]
    fingerprint: str
    asset_hash: str


class ALFWorldAdapter:
    """One local .tw-pddl task; no hidden facts, walkthrough or planner in prompts.

    Episode options: game_file (or task), verify_replay=True. TextWorld's PDDL
    simulator has no copy support, so every restore resets and replays its prefix.
    An invalid command is still a paid step. Seed is recorded, while fixed PDDL
    tasks and shuffle=False have no randomized task sampling.
    """

    def __init__(self, *, assets_root: str | Path, max_steps: int = 100) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.assets_root = Path(assets_root).resolve()
        self.max_steps = max_steps
        self._owner = uuid4().hex
        self._env = None
        self._result = None
        self._raw = None
        self._history: tuple[tuple[Action, str], ...] = ()
        self._snapshots: dict[str, EnvironmentSnapshot] = {}
        self._verifying = False
        self.capabilities = EnvironmentCapabilities()

    @contextmanager
    def _operation(self, ledger, usage, source, phase, budget_phase=None):
        if ledger.over_budget:
            raise BudgetExceeded(ledger.over_budget, phase)
        metadata = {"purpose": "capability_verification"} if self._verifying else {}
        ledger.charge(usage, source=source, phase=phase, budget_phase=budget_phase, metadata=metadata)
        started, failed = perf_counter(), False
        try:
            yield
        except Exception as error:
            failed = True
            if isinstance(error, (BudgetExceeded, EnvironmentFailure, UnsupportedCapability)):
                raise
            raise EnvironmentFailure(f"{source} failed: {error}") from error
        finally:
            ledger.record(Usage(wall_seconds=perf_counter() - started), source=source,
                          phase=phase, budget_phase=budget_phase, failed=failed, metadata=metadata)

    def _convert(self, raw, reward=0.0, done=False, valid=None):
        # A whitelist is essential: GameState also contains _facts/_valid_actions.
        observation = Observation(text=str(raw.feedback),
            admissible_actions=tuple(Action(name=str(x)) for x in raw.admissible_commands))
        won = bool(raw.won)
        return StepResult(observation=observation, reward=float(reward),
            success=won, terminated=bool(done or won), action_success=valid,
            truncated=not (done or won) and len(self._history) >= self.max_steps,
            objective_score=float(won), info={"moves": int(raw.get("moves", 0))})

    def _fingerprint(self):
        private = sorted(str(x) for x in self._raw.get("_facts", ()))
        data = [self._result.model_dump(mode="json"), private, len(self._history)]
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def _reset(self, ledger, phase, budget_phase=None):
        with self._operation(ledger, Usage(env_resets=1, env_observations=1),
                             "alfworld.reset", phase, budget_phase):
            if self._env is None:
                os.environ.setdefault("ALFWORLD_DATA", str(self.assets_root / "data" / "alfworld"))
                import textworld
                from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
                # Fast Downward unlinks dlopen'ed .so files; NFS cannot remove those.
                previous_tmp = tempfile.tempdir
                try:
                    tempfile.tempdir = "/tmp"
                    self._env = textworld.start(str(self._game),
                        textworld.EnvInfos(won=True, admissible_commands=True, moves=True),
                        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
                finally:
                    tempfile.tempdir = previous_tmp
            self._history = ()
            self._raw = self._env.reset()
            self._result = self._convert(self._raw)
            return self._result

    def reset(self, spec: EpisodeSpec, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.COMMIT) -> StepResult:
        self.close()
        game = Path(str(spec.options.get("game_file", spec.task)))
        self._game = (self.assets_root / game).resolve() if not game.is_absolute() else game.resolve()
        if not self._game.is_file() or self._game.suffix != ".tw-pddl":
            raise EnvironmentFailure(f"A local ALFWorld .tw-pddl task is required: {self._game}")
        verify = spec.options.get("verify_replay", True)
        if type(verify) is not bool:
            raise ValueError("verify_replay must be boolean")
        self._asset_hash = hashlib.sha256(self._game.read_bytes()).hexdigest()
        self._verifying = verify
        try:
            ledger.ensure_available(Usage(env_resets=3 if verify else 1,
                env_observations=5 if verify else 1, env_steps=2 if verify else 0,
                replay_steps=1 if verify else 0, env_snapshots=2 if verify else 0,
                env_restores=2 if verify else 0), phase=phase)
            initial = self._reset(ledger, phase)
            if verify and not initial.done:
                if not initial.observation.admissible_actions:
                    raise EnvironmentFailure("No action available for replay verification")
                root = self._snapshot(ledger, phase)
                self._step(initial.observation.admissible_actions[0], ledger, phase)
                child = self._snapshot(ledger, phase)
                self._restore(child, ledger, phase)
                self._restore(root, ledger, phase)
                self._snapshots.clear()
                self.capabilities = EnvironmentCapabilities(deterministic_replay=True,
                    details={"replay_verified": True, "asset_sha256": self._asset_hash})
            return initial
        except Exception:
            self.close()
            raise
        finally:
            self._verifying = False

    def _step(self, action, ledger, phase, budget_phase=None):
        if self._env is None or self._result.done:
            raise EnvironmentFailure("Environment must be active and nonterminal")
        if action.arguments:
            raise EnvironmentFailure("ALFWorld actions use a full text command with no arguments")
        valid = action in self._result.observation.admissible_actions
        with self._operation(ledger, Usage(env_steps=1, env_observations=1,
                             replay_steps=int(phase == UsagePhase.REPLAY)),
                             "alfworld.step", phase, budget_phase):
            self._raw, reward, done = self._env.step(action.name)
            self._history += ((action, ""),)
            self._result = self._convert(self._raw, reward, done, valid)
            fingerprint = self._fingerprint()
            self._history = self._history[:-1] + ((action, fingerprint),)
            return self._result

    def step(self, action: Action, *, ledger: BudgetLedger,
             phase: UsagePhase = UsagePhase.COMMIT) -> StepResult:
        if UsagePhase(phase) == UsagePhase.SEARCH:
            self.capabilities.require_branching()
        return self._step(action, ledger, UsagePhase(phase))

    def _snapshot(self, ledger, phase):
        if self._env is None:
            raise EnvironmentFailure("Environment is closed")
        with self._operation(ledger, Usage(env_snapshots=1), "alfworld.snapshot", phase):
            snapshot = EnvironmentSnapshot(self._owner, _Checkpoint(
                self._history, self._fingerprint(), self._asset_hash))
            self._snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def snapshot(self, *, ledger: BudgetLedger, phase: UsagePhase = UsagePhase.SEARCH):
        self.capabilities.require_branching()
        return self._snapshot(ledger, phase)

    def _restore(self, snapshot, ledger, phase):
        if snapshot.owner_id != self._owner or self._snapshots.get(snapshot.snapshot_id) is not snapshot:
            raise EnvironmentFailure("Foreign, stale or forged snapshot")
        checkpoint = snapshot.payload
        if hashlib.sha256(self._game.read_bytes()).hexdigest() != checkpoint.asset_hash:
            raise EnvironmentFailure("Task file changed since snapshot")
        count = len(checkpoint.history)
        ledger.ensure_available(Usage(env_restores=1, env_resets=1, env_steps=count,
            replay_steps=count, env_observations=count + 1), phase=phase)
        ledger.charge(Usage(env_restores=1), source="alfworld.restore", phase=phase,
                      metadata={"purpose": "capability_verification"} if self._verifying else {})
        budget_phase = "commit" if UsagePhase(phase) == UsagePhase.COMMIT else "search"
        try:
            self._reset(ledger, UsagePhase.REPLAY, budget_phase)
            for action, fingerprint in checkpoint.history:
                self._step(action, ledger, UsagePhase.REPLAY, budget_phase)
                if self._fingerprint() != fingerprint:
                    raise EnvironmentFailure("ALFWorld replay diverged")
            if self._fingerprint() != checkpoint.fingerprint:
                raise EnvironmentFailure("ALFWorld restored state differs")
            return self._result
        except Exception:
            ledger.record(Usage(), source="alfworld.restore", phase=phase, failed=True)
            self.capabilities = EnvironmentCapabilities(details={"replay_verified": False})
            self.close()
            raise

    def restore(self, snapshot: EnvironmentSnapshot, *, ledger: BudgetLedger,
                phase: UsagePhase = UsagePhase.SEARCH) -> StepResult:
        self.capabilities.require_branching()
        return self._restore(snapshot, ledger, phase)

    def observe(self) -> Observation:
        if self._result is None:
            raise EnvironmentFailure("Environment is closed")
        return self._result.observation

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = self._raw = self._result = None
        self._history = ()
        self._snapshots.clear()
        self.capabilities = EnvironmentCapabilities()
