"""Explicit CPU integration fixture, never a pretrained or general-purpose model."""
from __future__ import annotations

import json
from time import perf_counter
from typing import Callable

from jevgo.core import (
    Action, BudgetLedger, GenerationRequest, GenerationResponse, JudgeRequest,
    JudgeResponse, JudgeRole, ModelFailure, ModelIdentity, UnsupportedCapability,
    Usage, UsagePhase,
)

FIXTURE_TASK = "MiniGrid-Empty-5x5-v0"


class DeterministicBackend:
    """Public-history script for one MiniGrid fixture; no hidden world-state access.

    Alternative scripted_actions are explicit testing inputs, not benchmark
    policies. Beyond the script, its final action repeats so exploratory branches
    remain executable. Zero token costs mean no model/tokenizer exists here.
    """
    def __init__(self, *, task: str = FIXTURE_TASK,
                 scripted_actions: tuple[str, ...] | list[str] = ("forward", "forward", "right", "forward", "forward")) -> None:
        if task != FIXTURE_TASK:
            raise UnsupportedCapability(f"DeterministicBackend only supports the CPU fixture {FIXTURE_TASK}")
        if (not isinstance(scripted_actions, (list, tuple)) or not scripted_actions
                or any(name not in ("left", "right", "forward", "pickup", "drop", "toggle", "done")
                       for name in scripted_actions)):
            raise ValueError("scripted_actions must be a nonempty list of MiniGrid action names")
        self.task, self.scripted_actions = task, tuple(scripted_actions)
        self.closed = False
        self.identity = ModelIdentity(backend="deterministic_fixture", model="minigrid-public-history-script",
                                      revision="fixture-v1", metadata={"validation_level": "cpu_fixture",
                                      "task": task, "scripted_actions": list(scripted_actions),
                                      "pretrained": False, "benchmark_effectiveness_claim": False})

    def close(self) -> None:
        self.closed = True

    def _state(self, text: str, image_paths: tuple[str, ...]) -> dict:
        if self.closed:
            raise ModelFailure("Deterministic fixture backend is closed")
        if image_paths:
            raise UnsupportedCapability("The deterministic fixture is text-only")
        try:
            tail = text.split("Public state:\n", 1)[1]
            state, _ = json.JSONDecoder().raw_decode(tail.lstrip())
            if state["task"] != self.task or not isinstance(state["actions"], list):
                raise ValueError("Unexpected task or public action history")
            for action in state["actions"]:
                Action.model_validate(action)
            return state
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelFailure("Deterministic fixture requires its MiniGrid public-state JSON") from error

    def _target(self, state: dict) -> str:
        return self.scripted_actions[min(len(state["actions"]), len(self.scripted_actions) - 1)]

    @staticmethod
    def _invoke(operation: Callable, *, kind: str, ledger: BudgetLedger, phase: UsagePhase):
        fixed = Usage(model_requests=1, **{kind: 1})
        ledger.charge(fixed, source="deterministic_fixture", phase=phase,
                      metadata={"validation_level": "cpu_fixture"})
        started, failed = perf_counter(), True
        try:
            value = operation()
            failed = False
        finally:
            measured = Usage(wall_seconds=perf_counter() - started)
            ledger.record(measured, source="deterministic_fixture.result", phase=phase, failed=failed,
                          metadata={"usage_complete": True, "token_accounting": "no_model_or_tokenizer"})
        return value, fixed + measured

    def generate(self, request: GenerationRequest, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> GenerationResponse:
        state = self._state(request.prompt, request.image_paths)
        target = self._target(state)
        def operation():
            names = (target, "left", "right", "forward")
            return tuple(json.dumps({"action": {"name": names[index % len(names)], "arguments": {}},
                                     "hypothesis": "Scripted CPU fixture action; outcome is unverified.",
                                     "rationale": "Fixed public-history integration fixture."})
                         for index in range(request.n))
        texts, usage = self._invoke(operation, kind="generation_calls", ledger=ledger, phase=phase)
        return GenerationResponse(texts=texts, identity=self.identity, usage=usage,
                                  metadata={"validation_level": "cpu_fixture", "usage_complete": True,
                                            "token_accounting": "no_model_or_tokenizer", "script_action": target})

    def score(self, request: JudgeRequest, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.SEARCH) -> JudgeResponse:
        state = self._state(request.context, request.image_paths)
        target = self._target(state)
        def operation():
            if request.role == JudgeRole.CANDIDATE:
                try:
                    return tuple(float(Action.model_validate(json.loads(item)["action"]).name == target)
                                 for item in request.items)
                except (KeyError, TypeError, ValueError) as error:
                    raise ModelFailure("Fixture candidate items must contain action JSON") from error
            return (1.0 if request.role == JudgeRole.EVIDENCE else 0.0,) * len(request.items)
        scores, usage = self._invoke(operation, kind="judge_calls", ledger=ledger, phase=phase)
        return JudgeResponse(scores=scores, identity=self.identity, usage=usage,
                             metadata={"validation_level": "cpu_fixture", "role": request.role.value,
                                       "objective_success": False, "calibrated": False, "usage_complete": True,
                                       "score_scale": "scripted_fixture_preference", "token_accounting": "no_model_or_tokenizer"})
