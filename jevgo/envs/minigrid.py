"""MiniGrid's public local observations and budgeted, verified seed replay."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import pickle
from time import perf_counter
from typing import Any
from uuid import uuid4

import gymnasium as gym
import minigrid  # noqa: F401: registers the MiniGrid environments
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, EnvironmentCapabilities,
    EnvironmentFailure, EnvironmentSnapshot, EpisodeSpec, Observation,
    StepResult, UnsupportedCapability, Usage, UsagePhase,
)


_ACTIONS = tuple(Action(name=name) for name in (
    "left", "right", "forward", "pickup", "drop", "toggle", "done",
))
_ACTION_INDEX = {action.name: index for index, action in enumerate(_ACTIONS)}


def describe_public_observation(observation: Observation) -> str:
    """E1: explain the public 7x7 encoding without consulting a live world."""
    objects = ", ".join(f"{index}={name}" for index, name in sorted(IDX_TO_OBJECT.items()))
    colors = ", ".join(f"{index}={name}" for index, name in sorted(IDX_TO_COLOR.items()))
    states = ", ".join(f"{index}={name}" for name, index in sorted(STATE_TO_IDX.items(), key=lambda p: p[1]))
    return observation.text + (
        "\nPublic MiniGrid encoding (zero-based local coordinates): "
        "data.image[x][y][channel], x=0..6 left to right, y=0..6 far to near; "
        "channel 0=object, 1=color, 2=door state. Text grid rows iterate y and columns iterate x. "
        "The agent is at local (x=3,y=6); the cell directly forward is (x=3,y=5). "
        "The agent's local cell encodes the carried object or empty, not an agent marker. "
        "The local view rotates with the agent: decreasing local y always means forward, "
        "not necessarily global north. The public compass is data.direction: "
        "0=east, 1=south, 2=west, 3=north. Left/right turn the agent; forward acts toward "
        "the local front cell. Local coordinates describe view slots, not persistent world locations. "
        f"Object codes: {objects}. Color codes: {colors}. Door-state codes: {states}; "
        "interpret channel 2 this way only for doors. Unseen (object 0) means unknown, "
        "not empty; no contents behind occlusion are inferred."
    )


@dataclass(frozen=True)
class _Event:
    action: Action
    public: str
    private: str


@dataclass(frozen=True)
class _Checkpoint:
    seed: int
    history: tuple[_Event, ...]
    public: str
    private: str


class MiniGridAdapter:
    """Text-only 7x7 agent view, with no privileged state in Observation.

    Episode options: env_id (otherwise task, then constructor default),
    max_steps, verify_replay (default True). Verification costs three resets,
    two steps (one replay), two snapshots, two restores and five observations.
    Disabling verification permits online execution but disables branching.
    """

    def __init__(self, env_id: str = "MiniGrid-Empty-5x5-v0", *,
                 max_steps: int | None = None) -> None:
        self._default_id = env_id
        self._default_max_steps = max_steps
        self._owner = uuid4().hex
        self._env: Any = None
        self._result: StepResult | None = None
        self._history: tuple[_Event, ...] = ()
        self._snapshots: dict[str, EnvironmentSnapshot] = {}
        self._seed = 0
        self._initial: tuple[str, str] | None = None
        self._verifying = False
        self.capabilities = EnvironmentCapabilities(details={"replay_verified": False})

    @property
    def _metadata(self):
        return {"purpose": "capability_verification"} if self._verifying else {}

    @staticmethod
    def _preflight(ledger, usage, phase):
        if ledger.over_budget:
            raise BudgetExceeded(ledger.over_budget, phase)
        ledger.ensure_available(usage, phase=phase)

    @contextmanager
    def _operation(self, ledger, usage, source, phase, budget_phase=None):
        if ledger.over_budget:
            raise BudgetExceeded(ledger.over_budget, phase)
        ledger.charge(usage, source=source, phase=phase, budget_phase=budget_phase,
                      metadata=self._metadata)
        started, failed = perf_counter(), False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            ledger.record(Usage(wall_seconds=perf_counter() - started), source=source,
                          phase=phase, budget_phase=budget_phase, failed=failed,
                          metadata=self._metadata)

    @staticmethod
    def _observation(raw: dict) -> Observation:
        # Whitelist these three upstream fields; never copy raw info or env state.
        image = np.asarray(raw["image"])
        if image.shape != (7, 7, 3):
            raise EnvironmentFailure("MiniGrid requires a local 7x7x3 symbolic observation")
        direction, mission = int(raw["direction"]), str(raw["mission"])
        rows = []
        for y in range(7):
            cells = []
            for x in range(7):
                kind, color, state = map(int, image[x, y])
                name = IDX_TO_OBJECT[kind]
                if name not in ("unseen", "empty"):
                    name = f"{IDX_TO_COLOR[color]} {name}"
                if kind == 4:  # Door's public state encoding.
                    name += " " + ("open", "closed", "locked")[state]
                cells.append(name)
            rows.append(" | ".join(cells))
        text = (f"Mission: {mission}\nDirection: {direction} "
                "(0=east, 1=south, 2=west, 3=north).\n"
                "Local view: columns left to right; rows far to near; "
                "agent at bottom center facing upward. Unseen cells are unknown.\n"
                + "\n".join(rows))
        return Observation(text=text, data={"image": image.tolist(),
                           "direction": direction, "mission": mission},
                           admissible_actions=_ACTIONS)

    def _world_bytes(self) -> bytes:
        env = self._env.unwrapped
        # Includes box contents and carried object attributes, unlike env.hash().
        return pickle.dumps((env.grid, env.carrying, tuple(env.agent_pos),
                             int(env.agent_dir), env.mission), protocol=5)

    def _fingerprints(self) -> tuple[str, str]:
        assert self._result is not None
        env = self._env.unwrapped
        wrappers = []
        wrapper = self._env
        while wrapper is not env:
            if hasattr(wrapper, "_elapsed_steps"):
                wrappers.append(wrapper._elapsed_steps)
            wrapper = wrapper.env
        private = self._world_bytes() + pickle.dumps(
            (env.step_count, env.np_random.bit_generator.state, wrappers), protocol=5)
        return (hashlib.sha256(self._result.model_dump_json().encode()).hexdigest(),
                hashlib.sha256(private).hexdigest())

    def _reset_world(self, ledger, phase, *, env_id=None, max_steps=None,
                     budget_phase=None) -> StepResult:
        with self._operation(ledger, Usage(env_resets=1, env_observations=1),
                             "minigrid.reset", phase, budget_phase):
            if env_id is not None:
                kwargs = {"agent_view_size": 7, "render_mode": None}
                if max_steps is not None:
                    kwargs["max_steps"] = max_steps
                self._env = gym.make(env_id, **kwargs)
                if not isinstance(self._env.unwrapped, MiniGridEnv):
                    raise EnvironmentFailure("The selected environment is not MiniGrid")
            raw, _ = self._env.reset(seed=self._seed)
            self._history = ()
            self._result = StepResult(observation=self._observation(raw))
            return self._result

    def reset(self, spec: EpisodeSpec, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.COMMIT) -> StepResult:
        phase = UsagePhase(phase)
        verify = spec.options.get("verify_replay", True)
        if type(verify) is not bool:
            raise ValueError("verify_replay must be a boolean")
        env_id = spec.options.get("env_id", spec.task or self._default_id)
        max_steps = spec.options.get("max_steps", self._default_max_steps)
        if not isinstance(env_id, str) or not env_id.startswith("MiniGrid-"):
            raise ValueError("env_id/task must be a registered MiniGrid environment ID")
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            raise ValueError("max_steps must be a positive integer")
        self.close()
        self._seed = spec.seed
        self._verifying = verify
        started = None
        prior_wall = ledger.used.wall_seconds
        failed = False
        try:
            self._preflight(ledger, Usage(env_resets=3 if verify else 1,
                env_observations=5 if verify else 1, env_steps=2 if verify else 0,
                replay_steps=1 if verify else 0, env_snapshots=2 if verify else 0,
                env_restores=2 if verify else 0), phase)
            started = perf_counter()
            initial = self._reset_world(ledger, phase, env_id=env_id, max_steps=max_steps)
            self._initial = self._fingerprints()
            if verify:
                root = self._snapshot(ledger, phase)
                self._step(Action(name="left"), ledger, phase)
                turned = self._snapshot(ledger, phase)
                self._restore(turned, ledger, phase)
                self._restore(root, ledger, phase)
                self._snapshots.clear()  # Internal probes are not public handles.
                self.capabilities = EnvironmentCapabilities(deterministic_replay=True,
                    details={"replay_verified": True, "mechanism": "seed_and_action_replay"})
            else:
                self.capabilities = EnvironmentCapabilities(
                    details={"replay_verified": False, "reason": "verification_disabled"})
            return initial
        except Exception as error:
            failed = True
            self.close()
            if isinstance(error, (BudgetExceeded, EnvironmentFailure, UnsupportedCapability)):
                raise
            raise EnvironmentFailure("MiniGrid reset or replay verification failed") from error
        finally:
            if started is not None:
                overhead = max(0, perf_counter() - started - (ledger.used.wall_seconds - prior_wall))
                ledger.record(Usage(wall_seconds=overhead), source="minigrid.initialize",
                              phase=phase, failed=failed, metadata=self._metadata)
            self._verifying = False

    def _step(self, action, ledger, phase, budget_phase=None) -> StepResult:
        with self._operation(ledger, Usage(env_steps=1, env_observations=1,
                             replay_steps=int(phase == UsagePhase.REPLAY)),
                             "minigrid.step", phase, budget_phase):
            before = self._world_bytes()
            raw, reward, terminated, truncated, _ = self._env.step(_ACTION_INDEX[action.name])
            self._result = StepResult(observation=self._observation(raw),
                action_success=self._world_bytes() != before, reward=float(reward),
                terminated=bool(terminated), truncated=bool(truncated),
                success=bool(terminated and reward > 0), objective_score=float(reward))
            public, private = self._fingerprints()
            self._history += (_Event(action, public, private),)
            return self._result

    def step(self, action: Action, *, ledger: BudgetLedger,
             phase: UsagePhase = UsagePhase.COMMIT) -> StepResult:
        self.observe()  # Reject use before reset or after a failed restoration.
        if self._result.done:
            raise EnvironmentFailure("MiniGrid episode is finished; reset or restore before stepping")
        if action.name not in _ACTION_INDEX or action.arguments:
            ledger.record(Usage(), source="minigrid.invalid_action", phase=phase, failed=True)
            return StepResult(observation=self.observe(), action_success=False,
                              info={"reason": "Unknown action or unexpected arguments"})
        try:
            return self._step(action, ledger, UsagePhase(phase))
        except BudgetExceeded:
            raise
        except Exception as error:
            self.close()
            raise EnvironmentFailure("MiniGrid step failed; reset is required") from error

    def _snapshot(self, ledger, phase) -> EnvironmentSnapshot:
        with self._operation(ledger, Usage(env_snapshots=1), "minigrid.snapshot", phase):
            public, private = self._fingerprints()
            snapshot = EnvironmentSnapshot(owner_id=self._owner, payload=_Checkpoint(
                self._seed, self._history, public, private))
            self._snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    def snapshot(self, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> EnvironmentSnapshot:
        self.capabilities.require_branching()
        self.observe()
        return self._snapshot(ledger, UsagePhase(phase))

    def _restore(self, snapshot, ledger, phase) -> StepResult:
        if (not isinstance(snapshot, EnvironmentSnapshot) or snapshot.owner_id != self._owner
                or self._snapshots.get(snapshot.snapshot_id) is not snapshot):
            raise EnvironmentFailure("Invalid, foreign, or expired MiniGrid snapshot")
        saved = snapshot.payload
        n = len(saved.history)
        self._preflight(ledger, Usage(env_restores=1, env_resets=1, env_steps=n,
                        replay_steps=n, env_observations=n + 1), phase)
        ledger.charge(Usage(env_restores=1), source="minigrid.restore", phase=phase,
                      metadata=self._metadata)
        started, prior_wall = perf_counter(), ledger.used.wall_seconds
        failed = False
        budget_phase = "commit" if phase == UsagePhase.COMMIT else "search"
        try:
            self._seed = saved.seed
            self._reset_world(ledger, UsagePhase.REPLAY, budget_phase=budget_phase)
            if self._fingerprints() != self._initial:
                raise EnvironmentFailure("MiniGrid initial replay state diverged")
            for event in saved.history:
                self._step(event.action, ledger, UsagePhase.REPLAY, budget_phase)
                if self._fingerprints() != (event.public, event.private):
                    raise EnvironmentFailure("MiniGrid action replay diverged")
            if self._fingerprints() != (saved.public, saved.private):
                raise EnvironmentFailure("MiniGrid restored state diverged")
            return self._result
        except Exception as error:
            failed = True
            self.close()
            self.capabilities = EnvironmentCapabilities(
                details={"replay_verified": False, "reason": "restore_failed"})
            if isinstance(error, BudgetExceeded):
                raise
            raise EnvironmentFailure("MiniGrid replay mismatch/failure; branching disabled; reset required") from error
        finally:
            # Charge only restoration overhead, not time already measured by reset/step.
            overhead = max(0, perf_counter() - started - (ledger.used.wall_seconds - prior_wall))
            ledger.record(Usage(wall_seconds=overhead), source="minigrid.restore",
                          phase=phase, failed=failed, metadata=self._metadata)

    def restore(self, snapshot: EnvironmentSnapshot, *, ledger: BudgetLedger,
                phase: UsagePhase = UsagePhase.SEARCH) -> StepResult:
        self.capabilities.require_branching()
        return self._restore(snapshot, ledger, UsagePhase(phase))

    def observe(self) -> Observation:
        if self._result is None or self._env is None:
            raise EnvironmentFailure("MiniGrid has no valid cached observation; reset is required")
        return self._result.observation

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env, self._result, self._initial = None, None, None
        self._history = ()
        self._snapshots.clear()
        self.capabilities = EnvironmentCapabilities(details={"replay_verified": False})
