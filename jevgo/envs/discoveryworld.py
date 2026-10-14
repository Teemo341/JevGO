"""Isolated DiscoveryWorld workers with cached observations and guarded replay."""
from __future__ import annotations

import base64
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import select
import subprocess
import sys
import tempfile
from time import perf_counter
from uuid import uuid4
import zlib

from jevgo.core import (
    Action, BudgetExceeded, BudgetLedger, EnvironmentCapabilities, EnvironmentFailure,
    EnvironmentSnapshot, EpisodeSpec, Observation, StepResult, Usage, UsagePhase,
)


_UI_KEYS = ("agentLocation", "world_steps", "inventoryObjects", "accessibleEnvironmentObjects",
            "nearbyObjects", "nearbyAgents", "discoveryFeed", "dialog_box",
            "extended_action_message", "lastActionMessage")
_ORACLE_KEYS = {"fullscorecard", "scorecard", "criticalhypotheses", "criticalquestions",
                "scorenormalized", "completed", "completedsuccessfully", "score", "maxscore"}


def _public(value):
    if isinstance(value, dict):
        return {str(k): _public(v) for k, v in value.items()
                if str(k).lower().replace("_", "") not in _ORACLE_KEYS}
    if isinstance(value, (tuple, list)):
        return [_public(v) for v in value]
    return value


def _public_ui(raw):
    ui = {key: _public(raw[key]) for key in _UI_KEYS if key in raw}
    ui["taskProgress"] = [{key: str(task.get(key, "")) for key in ("taskName", "description")}
                          for task in raw.get("taskProgress", ())]
    return ui


def _task_status(tasks):
    return (bool(tasks) and all(task.get("completed") is True for task in tasks),
            bool(tasks) and all(task.get("completedSuccessfully") is True for task in tasks))


def _admissible(ui, schemas, locations):
    dialog = ui.get("dialog_box", {})
    if dialog.get("is_in_dialog"):
        return tuple(Action(name="DIALOG_CHOICE", arguments={"chosen_dialog_option_int": int(key)})
                     for key in dialog.get("dialogOptions", {}) if str(key).isdigit())
    inventory = [item["uuid"] for item in ui.get("inventoryObjects", ())]
    accessible = [item["uuid"] for item in ui.get("accessibleEnvironmentObjects", ())]
    objects = sorted(set(inventory + accessible))
    actions = []
    for name, schema in schemas.items():
        if name in ("MOVE_DIRECTION", "ROTATE_DIRECTION"):
            actions.extend(Action(name=name, arguments={"arg1": direction})
                           for direction in ("north", "east", "south", "west"))
        elif name == "TELEPORT_TO_LOCATION":
            actions.extend(Action(name=name, arguments={"arg1": place}) for place in locations)
        elif name == "TELEPORT_TO_OBJECT":
            actions.extend(Action(name=name, arguments={"arg1": obj}) for obj in objects)
        elif name == "DISCOVERY_FEED_GET_POST_BY_ID":
            feed = ui.get("discoveryFeed", {})
            posts = feed.get("posts", []) + feed.get("scientific_articles", [])
            actions.extend(Action(name=name, arguments={"arg1": post_id})
                           for post_id in sorted({post["postID"] for post in posts}))
        elif schema["args"] == []:
            actions.append(Action(name=name))
        elif schema["args"] == ["arg1"]:
            choices = inventory if name == "DROP" else accessible if name == "PICKUP" else objects
            actions.extend(Action(name=name, arguments={"arg1": obj}) for obj in choices)
        elif schema["args"] == ["arg1", "arg2"]:
            first = inventory if name == "PUT" else objects
            actions.extend(Action(name=name, arguments={"arg1": a, "arg2": b})
                           for a in first for b in objects if a != b)
    return tuple(actions)


def _json_default(value):
    if isinstance(value, set):
        return sorted(value, key=str)
    if hasattr(value, "uuid"):
        return {"object_uuid": value.uuid}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "name"):
        return str(value.name)
    raise TypeError(f"Unsupported private fingerprint value: {type(value).__name__}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_json_default,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _private_fingerprint(api):
    # Internal engine history is trusted local pickle, never supplied by a user.
    history = pickle.loads(zlib.decompress(api.world.worldHistory[-1]))
    history.pop("runtime_seconds", None)
    objects = {}
    for column in api.world.grid:
        for tile in column:
            for layer in tile["layers"].values():
                for obj in layer:
                    for child in obj.getAllContainedObjectsAndParts():
                        objects[child.uuid] = child
    rng = [(key, hashlib.sha256(pickle.dumps(obj.rng.getstate())).hexdigest())
           for key, obj in sorted(objects.items())]
    return _digest({"world": history, "object_rng": rng, "world_rng": api.world.rng.getstate(),
                    "uuid_rng": api.world.uuidGenerator.random.getstate(), "api_rng": api.r.getstate(),
                    "acted": api.agentsThatHaveActedThisStep,
                    "ui": [{"messages": ui.messageQueueText, "dialog": ui.dialogToDisplay,
                            "last_message": ui.lastActionMessage} for ui in api.ui]})


def _worker_main():
    """One API per process. stdout is exclusively a JSON-lines reply channel."""
    protocol = sys.stdout
    with redirect_stdout(sys.stderr):
        from PIL import Image
        import discoveryworld.constants as constants
        api, last_observed = None, None
        events = Path("events.jsonl").open("a", encoding="utf-8")

        def event(operation, **details):
            events.write(json.dumps({"operation": operation, **details}) + "\n")
            events.flush()

        def observe(config, outcome=None):
            nonlocal last_observed
            step = api.world.getStepCounter()
            if last_observed == step:
                raise RuntimeError("A second real observation in the same tick is forbidden")
            raw = api.getAgentObservation(0)
            last_observed = step
            event("observe", world_steps=step)
            if raw["errors"]:
                raise RuntimeError("DiscoveryWorld observation failed")
            payload = base64.b64decode(raw["vision"]["base64_no_grid"].split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(payload)) as frame:
                rgb = frame.convert("RGB")
                pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            image_path = None
            if config["modality"] == "vl":
                path = Path(f"observation-{step}.png").resolve()
                path.write_bytes(payload)
                image_path = str(path)
            completed, success = _task_status(raw["ui"].get("taskProgress", []))
            return {"ui": _public_ui(raw["ui"]), "completed": completed, "success": success,
                    "action_success": None if outcome is None else outcome.get("success", False),
                    "action_errors": [] if outcome is None else outcome.get("errors", []),
                    "pixels": pixels, "private": _private_fingerprint(api), "image_path": image_path,
                    "schemas": api.listKnownActions(),
                    "locations": sorted(api.listTeleportLocationsDict())}

        try:
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request["operation"] == "load":
                        if api is not None:
                            raise RuntimeError("A worker may load only one scenario")
                        config = request["config"]
                        resource = Path(config["assets_root"]) if config["assets_root"] else None
                        if resource and (resource / "assets" / "spriteIndex.json").is_file():
                            constants.ASSETS_PATH = str(resource / "assets")
                            constants.DATA_PATH = str(resource / "data")
                        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
                        api = DiscoveryWorldAPI(threadID=0)
                        if not api.loadScenario(config["scenario"], config["difficulty"],
                                                randomSeed=config["seed"], numUserAgents=1):
                            raise RuntimeError("DiscoveryWorld scenario load failed")
                        event("load", scenario=config["scenario"], seed=config["seed"])
                        event("tick", initial=True, world_steps=api.world.getStepCounter())
                        response = observe(config)
                    elif request["operation"] == "step":
                        action = request["action"]
                        wire = dict(action["arguments"])
                        if action["name"] != "DIALOG_CHOICE":
                            wire["action"] = action["name"]
                        outcome = api.performAgentAction(0, wire)
                        event("action", action=action, success=outcome.get("success", False),
                              world_steps=api.world.getStepCounter())
                        tick = api.tick()
                        event("tick", initial=False, world_steps=api.world.getStepCounter())
                        if not tick["success"] or tick["errors"]:
                            raise RuntimeError("DiscoveryWorld tick failed")
                        response = observe(config, outcome)
                    else:
                        raise RuntimeError("Unknown worker operation")
                    protocol.write(json.dumps({"ok": True, "response": response}) + "\n")
                    protocol.flush()
                except BaseException as error:
                    event("failure", category=type(error).__name__)
                    protocol.write(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}) + "\n")
                    protocol.flush()
                    break
        finally:
            events.close()
            if api is not None:
                import pygame
                pygame.quit()


@dataclass
class _Session:
    process: subprocess.Popen
    directory: Path
    log: object
    result: StepResult | None = None
    initial: tuple[str, str] | None = None
    fingerprint: tuple[str, str] | None = None
    history: tuple = ()


@dataclass(frozen=True)
class _Checkpoint:
    spec: EpisodeSpec
    history: tuple
    initial: tuple[str, str]
    fingerprint: tuple[str, str]


class _ReplayMismatch(EnvironmentFailure):
    def __init__(self, category, stage):
        self.mismatch_category, self.stage = category, stage
        super().__init__(f"DiscoveryWorld {category} at {stage}; branching disabled")


class DiscoveryWorldAdapter:
    """Scenario seed replay is advertised only after a fresh-worker comparison.

    Default text observations expose public UI only. Output directories are
    created lazily at reset; every new worker owns a unique cwd. assets_root may
    point to package assets/data, otherwise the installed package supplies them.
    """

    def __init__(self, *, assets_root=None, output_dir=None, max_steps=100, timeout=120.0):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        self.assets_root = Path(assets_root).resolve() if assets_root else None
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.max_steps, self.timeout = max_steps, timeout
        self._owner = uuid4().hex
        self._session = None
        self._spec = None
        self._snapshots = {}
        self._verifying = False
        self.capabilities = EnvironmentCapabilities(details={"replay_verified": False})

    @contextmanager
    def _operation(self, ledger, usage, source, phase, budget_phase=None, **metadata):
        if self._verifying:
            metadata["purpose"] = "capability_verification"
        ledger.charge(usage, source=source, phase=phase, budget_phase=budget_phase, metadata=metadata)
        started, failed = perf_counter(), False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            ledger.record(Usage(wall_seconds=perf_counter() - started), source=source,
                          phase=phase, budget_phase=budget_phase, failed=failed, metadata=metadata)

    def _rpc(self, session, request):
        try:
            session.process.stdin.write(json.dumps(request) + "\n")
            session.process.stdin.flush()
            ready, _, _ = select.select([session.process.stdout], [], [], self.timeout)
            if not ready:
                raise EnvironmentFailure(f"DiscoveryWorld worker timed out after {self.timeout:g}s")
            line = session.process.stdout.readline()
            if not line:
                raise EnvironmentFailure("DiscoveryWorld worker exited without a reply")
            reply = json.loads(line)
            if not reply["ok"]:
                raise EnvironmentFailure(f"DiscoveryWorld worker failed: {reply['error']}")
            return reply["response"]
        except (OSError, ValueError) as error:
            raise EnvironmentFailure("DiscoveryWorld worker communication failed") from error

    def _accept(self, session, raw, *, step=False):
        ui = raw["ui"]
        actions = _admissible(ui, raw["schemas"], raw["locations"])
        observation = Observation(
            text=json.dumps(ui, ensure_ascii=False, sort_keys=True, indent=2),
            data={"ui": ui, "action_schema": raw["schemas"], "teleport_locations": raw["locations"]},
            admissible_actions=actions,
            image_paths=(raw["image_path"],) if raw["image_path"] else ())
        count = len(session.history) + int(step)
        success, completed = raw["success"], raw["completed"]
        result = StepResult(observation=observation, action_success=raw["action_success"],
                            reward=float(success) if step else 0, success=success,
                            terminated=completed, truncated=count >= self.max_steps and not (success or completed),
                            objective_score=float(success),
                            info={"world_steps": int(ui["world_steps"]), "action_errors": raw["action_errors"]})
        comparable = result.model_dump(mode="json")
        comparable["observation"]["image_paths"] = []
        fingerprint = (_digest([comparable, raw["pixels"] if self._modality == "vl" else None]),
                       raw["private"])
        session.result, session.fingerprint = result, fingerprint
        return result

    def _new_session(self, ledger, phase, budget_phase=None):
        session = log = None
        with self._operation(ledger, Usage(env_resets=1, env_observations=1),
                             "discoveryworld.reset", phase, budget_phase, initial_ticks=1):
            try:
                if self.output_dir:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                directory = Path(tempfile.mkdtemp(prefix="discoveryworld-worker-", dir=self.output_dir))
                log = (directory / "worker.log").open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment.update(SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                                   PYGAME_HIDE_SUPPORT_PROMPT="1", MPLCONFIGDIR=str(directory / "matplotlib"),
                                   PYTHONHASHSEED=str(self._spec.seed % (2**32)))
                environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + environment.get("PYTHONPATH", "")
                process = subprocess.Popen([sys.executable, "-B", "-u", "-m", __name__, "--worker"],
                    cwd=directory, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log, text=True, bufsize=1)
                session = _Session(process, directory, log)
                raw = self._rpc(session, {"operation": "load", "config": {
                    "scenario": self._spec.task or "Tutorial", "difficulty": self._difficulty,
                    "seed": self._spec.seed, "modality": self._modality,
                    "assets_root": str(self.assets_root) if self.assets_root else None}})
                self._accept(session, raw)
                session.initial = session.fingerprint
                return session
            except Exception as error:
                if session:
                    self._close_session(session)
                elif log:
                    log.close()
                if isinstance(error, (BudgetExceeded, EnvironmentFailure)):
                    raise
                raise EnvironmentFailure("DiscoveryWorld worker startup failed") from error

    @staticmethod
    def _close_session(session):
        if session.process.poll() is None:
            session.process.terminate()
        try:
            session.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            session.process.kill()
            session.process.wait(timeout=5)
        session.process.stdin.close()
        session.process.stdout.close()
        session.log.close()

    def _step(self, session, action, ledger, phase, budget_phase=None):
        with self._operation(ledger, Usage(env_steps=1, env_observations=1,
                             replay_steps=int(phase == UsagePhase.REPLAY)),
                             "discoveryworld.step", phase, budget_phase, ticks=1):
            raw = self._rpc(session, {"operation": "step", "action": action.model_dump(mode="json")})
            self._accept(session, raw, step=True)
            session.history += ((action, session.fingerprint),)
            return session.result

    def _capture(self, session, ledger, phase):
        with self._operation(ledger, Usage(env_snapshots=1), "discoveryworld.snapshot", phase):
            snapshot = EnvironmentSnapshot(owner_id=self._owner, payload=_Checkpoint(
                self._spec, session.history, session.initial, session.fingerprint))
            self._snapshots[snapshot.snapshot_id] = snapshot
            return snapshot

    @staticmethod
    def _compare(actual, expected, stage):
        if actual[0] != expected[0]:
            raise _ReplayMismatch("public_observation_mismatch", stage)
        if actual[1] != expected[1]:
            raise _ReplayMismatch("private_state_or_rng_mismatch", stage)

    def _replay(self, saved, ledger, phase):
        count = len(saved.history)
        ledger.ensure_available(Usage(env_restores=1, env_resets=1, env_steps=count,
            replay_steps=count, env_observations=count + 1), phase=phase)
        metadata = {"purpose": "capability_verification"} if self._verifying else {}
        ledger.charge(Usage(env_restores=1), source="discoveryworld.restore", phase=phase, metadata=metadata)
        started, prior_wall, failed = perf_counter(), ledger.used.wall_seconds, False
        session = None
        budget_phase = "commit" if phase == UsagePhase.COMMIT else "search"
        try:
            session = self._new_session(ledger, UsagePhase.REPLAY, budget_phase)
            self._compare(session.initial, saved.initial, "initial")
            for index, (action, fingerprint) in enumerate(saved.history, 1):
                self._step(session, action, ledger, UsagePhase.REPLAY, budget_phase)
                self._compare(session.fingerprint, fingerprint, f"step_{index}")
            self._compare(session.fingerprint, saved.fingerprint, "restored")
            return session
        except Exception as error:
            failed = True
            if session:
                self._close_session(session)
            if isinstance(error, (BudgetExceeded, EnvironmentFailure)):
                raise
            raise EnvironmentFailure("DiscoveryWorld replay failed") from error
        finally:
            overhead = max(0, perf_counter() - started - (ledger.used.wall_seconds - prior_wall))
            ledger.record(Usage(wall_seconds=overhead), source="discoveryworld.restore",
                          phase=phase, failed=failed, metadata=metadata)

    def reset(self, spec: EpisodeSpec, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.COMMIT) -> StepResult:
        verify = spec.options.get("verify_replay", True)
        modality = spec.options.get("modality", "text")
        if type(verify) is not bool or modality not in ("text", "vl"):
            raise ValueError("verify_replay must be boolean; modality must be text or vl")
        self._difficulty = str(spec.options.get("difficulty", "Normal"))
        self._modality = modality
        phase = UsagePhase(phase)
        self.close()
        self._spec, self._verifying = spec, verify
        try:
            ledger.ensure_available(Usage(env_resets=3 if verify else 1,
                env_observations=5 if verify else 1, env_steps=2 if verify else 0,
                replay_steps=1 if verify else 0, env_snapshots=2 if verify else 0,
                env_restores=2 if verify else 0), phase=phase)
            self._session = self._new_session(ledger, phase)
            details = {"replay_verified": False, "modality": modality, "scenario": spec.task or "Tutorial",
                       "difficulty": self._difficulty, "seed": spec.seed,
                       "verification_scope": "fresh process; public UI/pixels; world/object/UUID RNG; message queues"}
            if verify:
                probe = restored = None
                try:
                    root = self._capture(self._session, ledger, phase)
                    probe = self._replay(root.payload, ledger, phase)
                    self._step(probe, Action(name="ROTATE_DIRECTION", arguments={"arg1": "north"}), ledger, phase)
                    turned = self._capture(probe, ledger, phase)
                    restored = self._replay(turned.payload, ledger, phase)
                except _ReplayMismatch as error:
                    details.update(reason=error.mismatch_category, verification_stage=error.stage)
                else:
                    details["replay_verified"] = True
                finally:
                    for session in (probe, restored):
                        if session:
                            self._close_session(session)
                    self._snapshots.clear()
            else:
                details["reason"] = "verification_disabled"
            self.capabilities = EnvironmentCapabilities(deterministic_replay=details["replay_verified"],
                                                        details=details)
            return self._session.result  # The original worker was never perturbed.
        except Exception as error:
            self.close()
            if isinstance(error, (BudgetExceeded, EnvironmentFailure)):
                raise
            raise EnvironmentFailure("DiscoveryWorld reset failed") from error
        finally:
            self._verifying = False

    def step(self, action: Action, *, ledger: BudgetLedger,
             phase: UsagePhase = UsagePhase.COMMIT) -> StepResult:
        self.observe()
        if self._session.result.done:
            raise EnvironmentFailure("DiscoveryWorld episode is finished")
        if action.key() not in {item.key() for item in self.observe().admissible_actions}:
            ledger.record(Usage(), source="discoveryworld.invalid_action", phase=phase, failed=True)
            return StepResult(observation=self.observe(), action_success=False,
                              info={"reason": "Action is outside the public admissible action set"})
        try:
            return self._step(self._session, action, ledger, UsagePhase(phase))
        except BudgetExceeded:
            raise
        except Exception as error:
            self.close()
            if isinstance(error, EnvironmentFailure):
                raise
            raise EnvironmentFailure("DiscoveryWorld step failed") from error

    def snapshot(self, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> EnvironmentSnapshot:
        self.capabilities.require_branching()
        self.observe()
        return self._capture(self._session, ledger, UsagePhase(phase))

    def restore(self, snapshot: EnvironmentSnapshot, *, ledger: BudgetLedger,
                phase: UsagePhase = UsagePhase.SEARCH) -> StepResult:
        self.capabilities.require_branching()
        if (not isinstance(snapshot, EnvironmentSnapshot) or snapshot.owner_id != self._owner
                or self._snapshots.get(snapshot.snapshot_id) is not snapshot):
            raise EnvironmentFailure("Invalid, foreign, or expired DiscoveryWorld snapshot")
        try:
            restored = self._replay(snapshot.payload, ledger, UsagePhase(phase))
        except BudgetExceeded:
            raise
        except EnvironmentFailure as error:
            self.capabilities = EnvironmentCapabilities(details={"replay_verified": False,
                "reason": getattr(error, "mismatch_category", "worker_failure")})
            raise
        self._close_session(self._session)
        self._session = restored
        return restored.result

    def observe(self) -> Observation:
        if self._session is None or self._session.result is None:
            raise EnvironmentFailure("DiscoveryWorld has no cached observation; reset is required")
        return self._session.result.observation

    def close(self):
        if self._session:
            self._close_session(self._session)
        self._session = None
        self._snapshots.clear()
        self.capabilities = EnvironmentCapabilities(details={"replay_verified": False})


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker_main()
