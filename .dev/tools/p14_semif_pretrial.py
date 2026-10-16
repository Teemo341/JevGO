"""P14b/c development-only preparation and exact frozen-SemIf request replay.

Run from a checkout with its own source and an explicit shared assets root.
Preparation is CPU-only. Replay requires --execute and a separately allocated GPU.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

from jevgo.core import Action, BudgetLedger, BudgetLimits, Candidate, GenerationRequest, Usage, UsagePhase

SOURCES = (
    ("semif-matrix/semif-B0-full-ecaea88c689bda82", "3ad4b1a7b59543969a4fe1763aa48e9f"),
    ("semif-matrix/semif-B0-full-ecaea88c689bda82", "6432f6271b844060961457016a8af695"),
    ("semif-matrix-seed1/semif-B0-full-2811d54277003347", "e53dd1dc79c849cea34e460e6195ceec"),
    ("semif-matrix-seed1/semif-B0-full-2811d54277003347", "ca58918c24d84eddaf9043280f2b4467"),
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    return {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "cwd": str(Path.cwd()), "source_root": str(ROOT), "python": sys.executable,
            "tool_sha256": sha256(Path(__file__)), "argv": sys.argv, "hostname": platform.node(),
            "device_environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")}}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def public_actions(request: dict) -> list[dict]:
    # Decode just the saved public state. A captured repair suffix is retained in
    # the request but does not form part of the JSON state.
    state, _ = json.JSONDecoder().raw_decode(request["prompt"].split("\nPublic state:\n", 1)[1])
    return state["observation"]["admissible_actions"]


def validate_proposal(text: str, allowed: list[dict] | None) -> dict:
    """No salvage: only the production parser's complete outer fence is stripped.

    The exact prompt schema is stricter than Candidate's optional defaults.
    """
    result = {"json_valid": False, "schema_valid": None, "public_action_legal": None,
              "passed": False, "error_stage": None, "error": None}
    cleaned = text.strip()
    if cleaned.startswith("```json\n") and cleaned.endswith("```"):
        cleaned = cleaned[8:-3].strip()
    try:
        obj = json.loads(cleaned, object_pairs_hook=_unique_pairs, parse_constant=_bad_constant)
        result["json_valid"] = True
    except (ValueError, TypeError) as exc:
        return {**result, "error_stage": "json", "error": str(exc)}
    try:
        if not isinstance(obj, dict) or set(obj) != {"action", "hypothesis", "rationale"}:
            raise ValueError("Proposal must contain exactly action, hypothesis and rationale")
        if not isinstance(obj["action"], dict) or set(obj["action"]) != {"name", "arguments"}:
            raise ValueError("Action must contain exactly name and arguments")
        candidate = Candidate.model_validate(obj)
        result["schema_valid"] = True
    except (ValueError, TypeError) as exc:
        return {**result, "schema_valid": False, "error_stage": "schema", "error": str(exc)}
    if allowed is None or not allowed:
        return {**result, "error_stage": "public_action", "error": "Public action legality unknown: no admissible set"}
    keys = {Action.model_validate(action).key() for action in allowed}
    legal = candidate.action.key() in keys
    return {**result, "public_action_legal": legal, "passed": legal,
            "error_stage": None if legal else "public_action",
            "error": None if legal else "Action is not in the saved public admissible set"}


def extract_case(events_path: Path, proposal_id: str) -> dict:
    rows = [json.loads(line) for line in events_path.read_text().splitlines()]
    index = next(i for i, row in enumerate(rows) if row["event_id"] == proposal_id)
    proposal = rows[index]
    if proposal["event_type"] != "proposal_samples" or len(proposal["payload"]["texts"]) != 1:
        raise ValueError("Expected one saved proposal sample")
    request_index = max(i for i in range(index) if rows[i]["event_type"] == "generation_request")
    request_event = rows[request_index]
    if proposal["payload"]["attempt"] != request_event["payload"]["attempt"]:
        raise ValueError("Request/proposal attempt mismatch")
    usage = Usage()
    for row in rows[request_index + 1:index]:
        if row["event_type"] == "usage":
            usage += Usage.model_validate(row["payload"]["usage"])
    request = request_event["payload"]["request"]
    GenerationRequest.model_validate(request)
    allowed = public_actions(request)
    return {"case_id": f"seed{rows[index]['episode_id'].rsplit('=', 1)[-1]}-attempt{proposal['payload']['attempt']}",
            "source": {"events_path": str(events_path), "events_sha256": sha256(events_path),
                       "request_line": request_index + 1, "request_event_id": request_event["event_id"],
                       "proposal_line": index + 1, "proposal_event_id": proposal_id},
            "request": request, "public_actions": allowed,
            "original_text": proposal["payload"]["texts"][0],
            "original_usage": usage.model_dump(mode="json"),
            "original_identity": proposal["payload"]["identity"],
            "original_validation": validate_proposal(proposal["payload"]["texts"][0], allowed),
            "historical_finish_reason": None, "historical_eos_token_ids": None,
            "historical_stop_evidence": "at_limit_only" if usage.output_tokens == request["max_new_tokens"] else "shorter_than_limit_only"}


def prepare(evidence_root: Path, output_root: Path) -> dict:
    output = output_root / "b" / "prepared"
    output.mkdir(parents=True, exist_ok=False)
    cases = [extract_case(evidence_root / relative / "events.jsonl", event_id) for relative, event_id in SOURCES]
    manifest = {"schema_version": 1, "stage": "prepared_no_inference", "provenance": provenance(),
                "evidence_root": str(evidence_root), "output_reserves": [256, 512], "cases": cases,
                "limitations": ["Historical finish_reason and EOS IDs were not saved.",
                                "Four selected failures do not estimate population error rates."]}
    write_json(output / "replay-manifest.json", manifest)
    for case in cases:
        folder = output / case["case_id"]
        folder.mkdir()
        write_json(folder / "saved-request.json", case["request"])
        (folder / "original-output.txt").write_text(case["original_text"], encoding="utf-8")
    source_configs = {}
    for relative, _ in SOURCES:
        config_path = evidence_root / relative / "config.json"
        source_configs[relative] = {"path": str(config_path), "sha256": sha256(config_path)}
    for stage, reserves, seeds, simulations, max_steps, budget in (
        ("b", [256, 512], [0, 1], [8], 6, {"env_steps": 100, "model_requests": 64, "input_tokens": 100000, "output_tokens": 16000}),
        ("c", [256, 512], [0], [8, 2], 2, {"env_steps": 500, "model_requests": 500, "input_tokens": 1000000, "output_tokens": 100000}),
    ):
        configs = output_root / stage / "conditions"
        configs.mkdir(parents=True, exist_ok=False)
        for reserve in reserves:
            for seed in seeds:
                for simulation in simulations:
                    relative = SOURCES[0 if seed == 0 else 2][0]
                    source = json.loads((evidence_root / relative / "config.json").read_text())
                    payload = {"episode": source["episode"], "run": source["run"], "model": {"backend": "semif", "max_model_len": 4096, "dtype": "bfloat16"}}
                    payload["episode"]["episode_id"] = f"p14{stage}-output{reserve}-seed{seed}-sim{simulation}"
                    payload["run"].update(method="B0" if stage == "b" else "B3", learning_mode="none",
                                          seed=seed, max_steps=max_steps, candidate_count=4, simulations=simulation,
                                          max_depth=4, rollout_depth=2, budget=budget)
                    payload["run"]["options"] = {"memory_records": 4, "max_new_tokens": reserve}
                    write_json(configs / f"output{reserve}-seed{seed}-sim{simulation}.json", payload)
    write_json(output / "source-configs.json", source_configs)
    return {"manifest": str(output / "replay-manifest.json"), "cases": len(cases), "planned_probes": 8,
            "b_conditions": str(output_root / "b" / "conditions"), "c_conditions": str(output_root / "c" / "conditions")}


def stop_evidence(tokens: list[int], maximum: int, eos_ids: list[int]) -> dict:
    positions = [i for i, token in enumerate(tokens) if token in eos_ids]
    terminal = bool(tokens and tokens[-1] in eos_ids)
    at_limit = len(tokens) == maximum
    if terminal:
        inference = "terminal_eos_at_limit" if at_limit else "terminal_eos_before_limit"
    elif at_limit:
        inference = "at_limit_without_terminal_eos"
    else:
        inference = "shorter_without_terminal_eos_stop_unknown"
    return {"actual_output_tokens": len(tokens), "requested_output_tokens": maximum,
            "effective_eos_token_ids": eos_ids, "eos_positions": positions,
            "terminal_eos": terminal, "at_output_limit": at_limit,
            "finish_reason": None, "stop_evidence": inference,
            "finish_reason_status": "Transformers generate did not supply a finish_reason; token evidence is reported separately"}


@contextmanager
def capture_generation(backend):
    """Observe the existing generate call without changing arguments or RNG."""
    original = backend.model.generate
    had_override = "generate" in backend.model.__dict__
    old_override = backend.model.__dict__.get("generate")
    captures = []

    def observed(*args, **kwargs):
        item = {"completed": False}
        captures.append(item)
        prompt = kwargs["input_ids"][0].detach().cpu().tolist()
        configuration = getattr(backend.model, "generation_config", None)
        eos = kwargs.get("eos_token_id", getattr(configuration, "eos_token_id", None))
        eos_ids = [] if eos is None else list(eos) if isinstance(eos, (list, tuple)) else [eos]
        item.update(prompt_token_ids=prompt,
                    model_generation_config=configuration.to_dict() if configuration is not None else None,
                    invocation_options={key: value for key, value in kwargs.items()
                                        if key not in {"input_ids", "attention_mask", "streamer"}})
        output = original(*args, **kwargs)
        tokens = output.sequences[0].detach().cpu().tolist()[len(prompt):]
        item.update(completed=True, generated_token_ids=tokens,
                    **stop_evidence(tokens, kwargs["max_new_tokens"], eos_ids))
        return output

    backend.model.generate = observed
    try:
        yield captures
    finally:
        if had_override:
            backend.model.generate = old_override
        else:
            delattr(backend.model, "generate")


def replay(manifest_path: Path, output: Path, assets_root: Path) -> dict:
    from jevgo.models.semif import SemIfBackend
    manifest = json.loads(manifest_path.read_text())
    if len(manifest["cases"]) != 4 or manifest["output_reserves"] != [256, 512]:
        raise ValueError("P14b requires the four fixed cases and both 256/512 arms")
    output.mkdir(parents=True, exist_ok=False)
    from jevgo.factory import versions
    write_json(output / "versions.json", versions())
    write_json(output / "provenance.json", {**provenance(), "assets_root": str(assets_root),
                                            "manifest_sha256": sha256(manifest_path)})
    ledger = BudgetLedger(BudgetLimits(model_requests=8, input_tokens=100000, output_tokens=16000))
    backend = None
    results = []
    error = None
    started = time.monotonic()
    try:
        backend = SemIfBackend(assets_root=assets_root, max_model_len=4096, dtype="bfloat16")
        write_json(output / "identity.json", backend.identity.model_dump(mode="json"))
        for case in manifest["cases"]:
            if backend.identity.revision != case["original_identity"]["revision"]:
                raise ValueError("Historical/current model revisions differ")
            for maximum in manifest["output_reserves"]:
                folder = output / f"{case['case_id']}-output{maximum}"
                folder.mkdir()
                request_payload = {**case["request"], "max_new_tokens": maximum}
                request = GenerationRequest.model_validate(request_payload)
                write_json(folder / "request.json", request_payload)
                write_json(folder / "source.json", case)
                before = ledger.used
                entry_start = len(ledger.entries)
                row = {"case_id": case["case_id"], "max_new_tokens": maximum, "error": None}
                captures = []
                try:
                    with capture_generation(backend) as captures:
                        response = backend.generate(request, ledger=ledger, phase=UsagePhase.COMMIT)
                    # Preserve actual model output before observer assertions, so
                    # an instrumentation failure cannot discard the evidence.
                    write_json(folder / "response.json", response.model_dump(mode="json"))
                    (folder / "raw-output.txt").write_text(response.texts[0], encoding="utf-8")
                    if len(captures) != 1 or not captures[0]["completed"]:
                        raise ValueError("Expected exactly one captured complete model request")
                    capture = captures[0]
                    if capture["actual_output_tokens"] != response.usage.output_tokens:
                        raise ValueError("Captured output token count differs from backend accounting")
                    if len(capture["prompt_token_ids"]) != response.usage.input_tokens:
                        raise ValueError("Captured input token count differs from backend accounting")
                    row.update(validation=validate_proposal(response.texts[0], case["public_actions"]),
                               token_evidence=capture, usage=response.usage.model_dump(mode="json"),
                               matches_historical_text=response.texts[0] == case["original_text"])
                except Exception as exc:
                    row.update(error=f"{type(exc).__name__}: {exc}", captures=captures)
                row["ledger_used_before"] = before.model_dump(mode="json")
                row["ledger_used_after"] = ledger.used.model_dump(mode="json")
                details = [entry.metadata for entry in ledger.entries[entry_start:]]
                row["usage_complete"] = all(detail.get("usage_complete", True) and not detail.get("unknown_usage") for detail in details)
                row["unknown_usage"] = sorted({field for detail in details for field in detail.get("unknown_usage", ())})
                write_json(folder / "result.json", row)
                results.append(row)
                write_json(output / "partial-results.json", results)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if backend is not None:
            backend.close()
        write_json(output / "ledger.json", ledger.summary())
    completed = sum(row["error"] is None for row in results)
    result = {"stage": "completed" if error is None and completed == 8 else "incomplete", "error": error,
              "attempted_probes": len(results), "completed_probes": completed,
              "results": results, "usage": ledger.used.model_dump(mode="json"),
              "usage_complete": all(entry.metadata.get("usage_complete", True) and not entry.metadata.get("unknown_usage") for entry in ledger.entries),
              "all_eight_format_passed": len(results) == 8 and all(row.get("validation", {}).get("passed", False) for row in results),
              "end_to_end_seconds_including_load": time.monotonic() - started,
              "research_effectiveness": "pending"}
    write_json(output / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--evidence-root", required=True, type=Path)
    prep.add_argument("--output-root", required=True, type=Path)
    run = commands.add_parser("replay")
    run.add_argument("--corpus", "--manifest", dest="manifest", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--assets-root", required=True, type=Path)
    run.add_argument("--execute", action="store_true", help="Required after GPU ownership/admission is confirmed")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.evidence_root.resolve(), args.output_root.resolve())
    else:
        if not args.execute:
            parser.error("Replay loads the frozen model onto GPU; pass --execute only after allocation")
        result = replay(args.manifest.resolve(), args.output.resolve(), args.assets_root.resolve())
        write_json(args.output / "command-result.json", {"argv": sys.argv, "exit_code": 1 if result["stage"] == "incomplete" else 0})
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, ensure_ascii=False))
    return 0 if result.get("stage") != "incomplete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
