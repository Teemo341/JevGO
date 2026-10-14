"""ScienceWorld JVM adapter: explicit task variants, public observations, replay."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from time import perf_counter
from uuid import uuid4

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, EnvironmentCapabilities, EnvironmentFailure,
    EnvironmentSnapshot, EpisodeSpec, Observation, StepResult, Usage, UsagePhase,
)


@dataclass(frozen=True)
class _Checkpoint:
    history: tuple[tuple[Action, str], ...]
    fingerprint: str


class ScienceWorldAdapter:
    """An independent JVM per adapter. Reset/replay uses the same task/variation.

    Each upstream step reads eight observation fields: feedback, score, completed,
    moves, look, inventory, task description, and valid commands. Reset internally
    executes look around; that action and its eight reads are also charged. These
    reads do not all consume simulator moves. Cache reads are free.
    """
    OBSERVATIONS_PER_STEP = 8

    def __init__(self, *, assets_root: str | Path, max_steps: int = 100) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.assets_root, self.max_steps = Path(assets_root).resolve(), max_steps
        self._owner = uuid4().hex
        self._env = None
        self._result = None
        self._history = ()
        self._snapshots = {}
        self._verifying = False
        self.capabilities = EnvironmentCapabilities()
        self.last_process = None

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
            if isinstance(error, (BudgetExceeded, EnvironmentFailure)):
                raise
            raise EnvironmentFailure(f"{source} failed: {error}") from error
        finally:
            ledger.record(Usage(wall_seconds=perf_counter() - started), source=source,
                          phase=phase, budget_phase=budget_phase, failed=failed, metadata=metadata)

    @classmethod
    def _convert(cls, text, reward, done, info, limit):
        score, moves = float(info["score"]), int(info["moves"])
        success = score == 100
        truncated = moves > limit and not success and score >= 0
        public = {key: str(info[key]) for key in ("look", "inv", "taskDesc")}
        observation = Observation(text=str(text), data=public,
            admissible_actions=tuple(Action(name=str(action)) for action in info["valid"]))
        return StepResult(observation=observation, reward=float(reward), success=success,
            terminated=bool(done or success) and not truncated, truncated=truncated,
            objective_score=score, info={"moves": moves, "score": score,
                "raw_done": bool(done), "failure": score < 0})

    def _fingerprint(self):
        return hashlib.sha256(self._result.model_dump_json().encode()).hexdigest()

    def _reset(self, ledger, phase, budget_phase=None):
        with self._operation(ledger, Usage(env_resets=1, env_steps=1,
            replay_steps=int(phase == UsagePhase.REPLAY), env_observations=8),
            "scienceworld.reset", phase, budget_phase):
            if self._env is None:
                from scienceworld import ScienceWorldEnv
                java = self.assets_root / ".tools/java"
                if not (java / "bin/java").is_file():
                    raise EnvironmentFailure(f"Local Java runtime is missing: {java}")
                # Py4J resolves Java from JAVA_HOME. Set only during process launch.
                previous = os.environ.get("JAVA_HOME")
                os.environ["JAVA_HOME"] = str(java)
                try:
                    self._env = ScienceWorldEnv("", envStepLimit=self.max_steps)
                finally:
                    if previous is None:
                        os.environ.pop("JAVA_HOME", None)
                    else:
                        os.environ["JAVA_HOME"] = previous
                self.last_process = self._env._gateway.java_process
            self._env.load(self._task, self._variation, self._simplifications, generateGoldPath=False)
            text, info = self._env.reset()
            done = info["score"] < 0 or info["score"] == 100 or info["moves"] > self.max_steps
            self._result = self._convert(text, info["reward"], done, info, self.max_steps)
            self._history = ()
            return self._result

    def reset(self, spec: EpisodeSpec, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.COMMIT):
        self.close()
        self._task = spec.task
        self._variation = spec.options.get("variation", spec.seed)
        self._simplifications = spec.options.get("simplifications", "")
        verify = spec.options.get("verify_replay", True)
        if type(self._variation) is not int or self._variation < 0:
            raise ValueError("variation must be a nonnegative integer")
        if not isinstance(self._simplifications, str) or type(verify) is not bool:
            raise ValueError("simplifications must be text and verify_replay boolean")
        self._verifying = verify
        try:
            # Three resets (each has one internal action), one probe and one replay.
            ledger.ensure_available(Usage(env_resets=3 if verify else 1,
                env_steps=5 if verify else 1, env_observations=40 if verify else 8,
                replay_steps=3 if verify else 0, env_snapshots=2 if verify else 0,
                env_restores=2 if verify else 0), phase=phase)
            initial = self._reset(ledger, phase)
            if verify and not initial.done:
                root = self._snapshot(ledger, phase)
                self._step(Action(name="look around"), ledger, phase)
                child = self._snapshot(ledger, phase)
                self._restore(child, ledger, phase)
                self._restore(root, ledger, phase)
                self._snapshots.clear()
                self.capabilities = EnvironmentCapabilities(deterministic_replay=True,
                    details={"replay_verified": True, "task": self._task,
                        "variation": self._variation, "simplifications": self._simplifications,
                        "verification": "initial and one-action prefix; each restore checks all feedback"})
            return initial
        except Exception:
            self.close()
            raise
        finally:
            self._verifying = False

    def _step(self, action, ledger, phase, budget_phase=None):
        if self._env is None or self._result.done:
            raise EnvironmentFailure("ScienceWorld must be active and nonterminal")
        if action.arguments:
            raise EnvironmentFailure("ScienceWorld uses full text commands without arguments")
        with self._operation(ledger, Usage(env_steps=1, env_observations=8,
            replay_steps=int(phase == UsagePhase.REPLAY)), "scienceworld.step", phase, budget_phase):
            text, reward, done, info = self._env.step(action.name)
            self._result = self._convert(text, reward, done, info, self.max_steps)
            self._history += ((action, self._fingerprint()),)
            return self._result

    def step(self, action: Action, *, ledger: BudgetLedger, phase: UsagePhase = UsagePhase.COMMIT):
        if UsagePhase(phase) == UsagePhase.SEARCH:
            self.capabilities.require_branching()
        return self._step(action, ledger, UsagePhase(phase))

    def _snapshot(self, ledger, phase):
        if self._env is None:
            raise EnvironmentFailure("Environment is closed")
        with self._operation(ledger, Usage(env_snapshots=1), "scienceworld.snapshot", phase):
            snapshot = EnvironmentSnapshot(self._owner, _Checkpoint(self._history, self._fingerprint()))
            self._snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def snapshot(self, *, ledger: BudgetLedger, phase: UsagePhase = UsagePhase.SEARCH):
        self.capabilities.require_branching()
        return self._snapshot(ledger, phase)

    def _restore(self, snapshot, ledger, phase):
        if snapshot.owner_id != self._owner or self._snapshots.get(snapshot.snapshot_id) is not snapshot:
            raise EnvironmentFailure("Foreign, stale or forged snapshot")
        checkpoint = snapshot.payload
        n = len(checkpoint.history)
        ledger.ensure_available(Usage(env_restores=1, env_resets=1, env_steps=n + 1,
            replay_steps=n + 1, env_observations=(n + 1) * 8), phase=phase)
        ledger.charge(Usage(env_restores=1), source="scienceworld.restore", phase=phase,
                      metadata={"purpose": "capability_verification"} if self._verifying else {})
        budget_phase = "commit" if UsagePhase(phase) == UsagePhase.COMMIT else "search"
        try:
            self._reset(ledger, UsagePhase.REPLAY, budget_phase)
            for action, fingerprint in checkpoint.history:
                self._step(action, ledger, UsagePhase.REPLAY, budget_phase)
                if self._fingerprint() != fingerprint:
                    raise EnvironmentFailure("ScienceWorld replay diverged")
            if self._fingerprint() != checkpoint.fingerprint:
                raise EnvironmentFailure("ScienceWorld restored state differs")
            return self._result
        except Exception:
            ledger.record(Usage(), source="scienceworld.restore", phase=phase, failed=True)
            self.close()
            raise

    def restore(self, snapshot: EnvironmentSnapshot, *, ledger: BudgetLedger,
                phase: UsagePhase = UsagePhase.SEARCH):
        self.capabilities.require_branching()
        return self._restore(snapshot, ledger, phase)

    def observe(self):
        if self._result is None:
            raise EnvironmentFailure("Environment is closed")
        return self._result.observation

    def close(self):
        env, self._env = self._env, None
        self._result = None
        self._history = ()
        self._snapshots.clear()
        self.capabilities = EnvironmentCapabilities()
        if env is not None:
            process = env._gateway.java_process
            try:
                env.close()
            finally:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                env._obj_tree_tempdir.cleanup()
