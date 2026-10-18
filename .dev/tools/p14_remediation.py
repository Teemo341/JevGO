"""Finite P14 remediation experiments; manifests are saved before model loading.

Development only. Run once per immutable manifest, using the checkout's code,
shared local assets, original frozen models, and explicitly assigned GPUs.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jevgo.core import (Action, BudgetLedger, BudgetLimits, Candidate, EpisodeSpec,
                        GenerationRequest, InvalidCandidate, JSONLEventStore, JudgeRole,
                        ResearchState, RunConfig, UsagePhase)
from jevgo.envs.minigrid import MiniGridAdapter
from jevgo.research import ResearchPolicy, initialize_state, record_transition
from jevgo.runner import run_episode


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_status": subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True),
            "source_root": str(ROOT), "cwd": str(Path.cwd()), "python": sys.executable,
            "argv": sys.argv, "hostname": platform.node(),
            "device_environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")},
            "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT / "jevgo/research.py", ROOT / "jevgo/runner.py"]}}


class TraceBackend:
    """Log complete logical calls, including failures; reuse one owned model.

    The real backend exclusively owns charging. The runner may close this view;
    the experiment owner closes the underlying backend after all conditions.
    """
    def __init__(self, backend, output, *, semif=False):
        self.backend, self.output, self.semif = backend, output, semif
        self.identity = backend.identity
        self.number = 0
        self.request_origin = None

    def close(self):
        pass

    def call(self, method, request, *, ledger, phase=UsagePhase.SEARCH):
        self.number += 1
        row = {"call": self.number, "method": method, "phase": UsagePhase(phase).value,
               "request": request.model_dump(mode="json"), "started_unix": time.time()}
        if method == "generate":
            origin = self.request_origin
            for path in (self.output / "episode/events.jsonl", self.output / "probe-events.jsonl"):
                if path.exists():
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    last = next((e for e in reversed(rows) if e["event_type"] == "generation_request"), None)
                    if last and last["payload"]["request"] == row["request"]:
                        payload = last["payload"]
                        origin = {"event_id": last["event_id"], "attempt": payload.get("attempt"),
                                  "source": payload.get("source", "repair" if payload.get("attempt") == 1 else "initial"),
                                  "repair": payload.get("source") == "repair" if "source" in payload else payload.get("attempt") == 1}
                        break
            row["generation_origin"] = origin
            self.backend.p14_generation_origin = origin
        append(self.output / "logical.jsonl", {"event": "request", **row})
        before, started = len(ledger.entries), time.monotonic()
        captured = []
        if self.semif and method == "generate":
            from p14_semif_pretrial import capture_generation
            context = capture_generation(self.backend)
        else:
            context = nullcontext(captured)
        try:
            with context as captured:
                response = getattr(self.backend, method)(request, ledger=ledger, phase=phase)
            row["response"] = response.model_dump(mode="json")
            return response
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            row.update(elapsed_seconds=time.monotonic() - started,
                       ledger_entries=[e.model_dump(mode="json") for e in ledger.entries[before:]],
                       generation_token_evidence=captured)
            append(self.output / "logical.jsonl", {"event": "finished", **row})

    def generate(self, request, **kwargs):
        return self.call("generate", request, **kwargs)

    def score(self, request, **kwargs):
        return self.call("score", request, **kwargs)


def fixed_state(condition, ledger):
    """Tester-created public states; control actions never become online proposals."""
    if "state_file" in condition:
        path = Path(condition["state_file"])
        if sha(path) != condition["state_sha256"]:
            raise ValueError("Fixed public-state source hash changed")
        return ResearchState.model_validate_json(path.read_text())
    episode = EpisodeSpec(environment="minigrid", task=condition.get("task", "MiniGrid-Empty-5x5-v0"),
                          seed=condition.get("seed", 0), episode_id=condition["id"],
                          options={"max_steps": 20})
    env = MiniGridAdapter(env_id=episode.task, max_steps=20)
    try:
        state = initialize_state(episode, env.reset(episode, ledger=ledger, phase=UsagePhase.COMMIT))
        for name in condition.get("setup_actions", []):
            candidate = Candidate(action=Action(name=name), hypothesis="The public observation may change after this action.",
                                  rationale="Tester-provided state preparation; not a generator output.")
            state = record_transition(state, candidate, env.step(candidate.action, ledger=ledger, phase=UsagePhase.COMMIT))
        return state
    finally:
        env.close()


def execute_condition(condition, backend, output):
    output.mkdir(parents=True, exist_ok=False)
    write(output / "condition.json", condition)
    ledger = BudgetLedger(BudgetLimits.model_validate(condition["budget"]))
    trace = TraceBackend(backend, output, semif=condition["model_key"] == "semif")
    if hasattr(backend, "journal"):
        backend.journal = output / "http.jsonl"
        backend.number = 0
    config = RunConfig.model_validate({"method": "B1", "candidate_count": 4,
        "judge_candidates": False, "judge_evidence": False, "judge_stop": False,
        "budget": condition["budget"], "seed": condition.get("seed", 0), **condition.get("run", {})})
    events = JSONLEventStore(output / "probe-events.jsonl")
    policy = ResearchPolicy(trace, trace, config, events=events)
    started = time.monotonic()
    result = {"id": condition["id"], "kind": condition["kind"], "model_key": condition["model_key"],
              "effectiveness_evidence": False, "attempt": 1}
    try:
        kind = condition["kind"]
        if kind == "episode":
            episode = EpisodeSpec(environment="minigrid", task=condition.get("task", "MiniGrid-Empty-5x5-v0"),
                                  seed=condition.get("seed", 0), episode_id=condition["id"], options={"max_steps": 20})
            env = MiniGridAdapter(env_id=episode.task, max_steps=20)
            run = run_episode(episode, config, env, trace, trace, output_dir=output / "episode",
                metadata={"scope": "P14 finite development loop", "model_key": condition["model_key"],
                          "max_attempts": 1, "environment_horizon": 20, "validation_level": "real_model_attempt",
                          "model_identity": trace.identity.model_dump(mode="json")})
            result.update(termination_reason=run.termination_reason.value, objective_success=run.final_state.success,
                          actions=[a.name for a in run.final_state.actions], error=run.error,
                          usage=run.usage.model_dump(mode="json"), result_path=str(output / "episode/result.json"))
            ledger_data = json.loads((output / "episode/ledger.json").read_text())
        elif kind == "request":
            trace.request_origin = condition.get("generation_origin")
            response = trace.generate(GenerationRequest.model_validate(condition["request"]), ledger=ledger, phase=UsagePhase.COMMIT)
            from jevgo.research import parse_generated_proposal
            validations = []
            for text in response.texts:
                try:
                    candidate = parse_generated_proposal(text, admissible_actions=tuple(Action.model_validate(a) for a in condition["allowed_actions"]))
                    validations.append({"valid": True, "action": candidate.action.model_dump(mode="json")})
                except (ValueError, InvalidCandidate) as exc:
                    validations.append({"valid": False, "error": str(exc)})
            result.update(validations=validations, source=condition.get("source"))
            ledger_data = ledger.summary()
        else:
            state = fixed_state(condition, ledger)
            write(output / "state.json", state.model_dump(mode="json"))
            if kind == "candidates":
                candidates = policy.candidates(state, ledger=ledger, phase=UsagePhase.COMMIT)
                result["candidates"] = [c.model_dump(mode="json") for c in candidates]
            elif kind == "judge":
                actions = condition["candidate_actions"]
                items = tuple(json.dumps({"action": {"name": a, "arguments": {}},
                    "hypothesis": "The public observation may change after this action."}) for a in actions)
                scores = policy._scores(state, JudgeRole.CANDIDATE,
                    "Compare the plausibility and usefulness of proposed next actions. These are hypotheses, not facts.",
                    items, ledger, UsagePhase.COMMIT)
                result.update(candidate_origin="tester_provided_not_generator_not_online_policy",
                              actions=actions, scores=list(scores))
            elif kind == "memory":
                selected = policy.select_memory(state, ledger=ledger, phase=UsagePhase.COMMIT)
                write(output / "selected-state.json", selected.model_dump(mode="json"))
                result.update(before_memory=list(state.working_memory), retained_ids=list(selected.working_memory))
                candidates = policy.candidates(selected, ledger=ledger, phase=UsagePhase.COMMIT)
                result["candidates"] = [c.model_dump(mode="json") for c in candidates]
            else:
                raise ValueError(f"Unknown condition kind {kind}")
            ledger_data = ledger.summary()
        result.setdefault("status", "executed")
    except Exception as exc:
        result.update(status="execution_failed", error={"type": type(exc).__name__, "message": str(exc)})
        ledger_data = ledger.summary()
    result.update(usage=ledger_data["used"], elapsed_seconds=time.monotonic() - started,
                  unknown_usage_entries=[e for e in ledger_data["entries"] if e["metadata"].get("usage_complete") is False],
                  over_budget=ledger_data.get("over_budget"))
    write(output / "ledger.json", ledger_data)
    write(output / "result.json", result)
    return result


def validate_manifest(manifest):
    if manifest.get("max_attempts") != 1 or not manifest.get("conditions"):
        raise ValueError("Declare one attempt and a nonempty finite condition list")
    names = [c["id"] for c in manifest["conditions"]]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("Condition IDs must be unique safe basenames")
    for c in manifest["conditions"]:
        if c["model_key"] != manifest["model"]["backend"]:
            raise ValueError("One fixed backend per batch")
        limits = BudgetLimits.model_validate(c["budget"])
        if limits.model_requests is None or limits.input_tokens is None or limits.output_tokens is None:
            raise ValueError("Every condition needs finite request/input/output budgets")
        if c.get("run", {}).get("learning_mode", "none") != "none":
            raise ValueError("P14 only supports frozen inference")
        if "budget" in c.get("run", {}):
            raise ValueError("Declare budget once at condition level; run.budget is forbidden")
    return manifest


def execute(manifest_path, output, assets):
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    info = provenance()
    if info["tracked_status"]:
        raise ValueError("Commit tracked source changes before real execution")
    write(output / "execution-record.json", {**info, "manifest_sha256": sha(manifest_path),
          "manifest": manifest, "max_attempts": 1, "created_before_model_load": True})
    backend, results, started = None, [], time.monotonic()
    try:
        from jevgo.factory import build_backend, versions
        write(output / "versions.json", versions())
        if manifest["model"]["backend"] == "jev_9b":
            from p14_context import GuardedBackend, Lengths, service_admission
            from jevgo.models.autotrust import AutoTrustBackend
            class AuditedBackend(GuardedBackend):
                def generate(self, request, **kwargs):
                    origin = getattr(self, "p14_generation_origin", None) or {}
                    self.logical = {"role": "generation", "repair": origin.get("repair"),
                                    "generation_origin": origin}
                    return AutoTrustBackend.generate(self, request, **kwargs)
            options = dict(manifest["model"])
            key = options.pop("backend")
            write(output / "service-admission.json", service_admission(options["base_url"], options["max_model_len"]))
            backend = AuditedBackend(key, assets_root=assets, journal=output / "http.jsonl", lengths=Lengths(assets), **options)
        else:
            backend = build_backend(manifest["model"], assets_root=assets)
        write(output / "model-identity.json", backend.identity.model_dump(mode="json"))
        for c in manifest["conditions"]:
            result = execute_condition(c, backend, output / c["id"])
            results.append(result)
            write(output / "summary.json", {"conditions": results, "complete": False})
            print(json.dumps({"id": c["id"], "status": result["status"], "usage": result["usage"]}), flush=True)
    except Exception as exc:
        write(output / "batch-error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if backend is not None:
            backend.close()
        write(output / "summary.json", {"conditions": results, "complete": len(results) == len(manifest["conditions"]),
            "end_to_end_seconds": time.monotonic() - started, "actual_model_attempts": len(results),
            "missing_conditions": [c["id"] for c in manifest["conditions"][len(results):]],
            "effectiveness_verified": False, "original_precision": True, "learning_mode": "none"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    if not args.execute:
        print("Manifest valid; no model loaded. Supply --execute after GPU allocation.")
        return
    execute(args.manifest, args.output, args.assets_root)


if __name__ == "__main__":
    main()
