"""Reconcile P14 SemIf cost by actual committed action, with initialization/tail.

Every ledger entry belongs to exactly one segment and one cost category. Restores
and their replay actions remain separate from search/commit atomic transitions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jevgo.core import Usage
from jevgo.evaluation import audit_run


def _total(entries) -> dict:
    total = Usage()
    for entry in entries:
        total += Usage.model_validate(entry["usage"])
    return total.model_dump(mode="json")


def _category(entry, role):
    source, phase = entry["source"], entry["phase"]
    if source.startswith("semif."):
        if source.startswith("semif.generate") or source.startswith("semif.model.generate"):
            return "generation"
        if source.startswith("semif.score") or source.startswith("semif.direct"):
            if role is None:
                raise ValueError("SemIf judge usage without a preceding judge_request role")
            return "judge_" + role
    if phase == "replay" or source.endswith(".restore"):
        return "restore_replay"
    if source.startswith("mcts."):
        return "search_overhead_and_rollout_count"
    if source.startswith("minigrid."):
        return "search_environment" if phase == "search" else "commit_environment"
    return "other"


def audit_semif_calls(events: list[dict]) -> list[dict]:
    """Independently tie charged logical calls/internal requests to request spans.

    A logged request may be rejected before the backend charges or sends it.
    Such attempts remain explicit zero-request rows; no successful cost is inferred.
    """
    spans = []
    active = None
    for line, event in enumerate(events, 1):
        kind, payload = event["event_type"], event["payload"]
        if kind in ("generation_request", "judge_request"):
            active = {"request_event_line": line, "kind": kind, "request": payload["request"],
                      "entries": [], "response": None}
            spans.append(active)
        elif kind == "usage" and payload["source"].startswith("semif."):
            if active is None:
                raise ValueError("SemIf charge has no owning request span")
            active["entries"].append(payload)
        elif kind in ("proposal_samples", "judge") and active is not None:
            expected = "proposal_samples" if active["kind"] == "generation_request" else "judge"
            if kind != expected:
                raise ValueError("Request/response event types disagree")
            active["response"] = payload
            active["response_event_line"] = line
            active = None
    result = []
    for span in spans:
        entries, request, response = span["entries"], span["request"], span["response"]
        if not entries:  # Also supports CPU fixture runs without pretending they are SemIf.
            if response is not None and response.get("identity", {}).get("backend") == "semif":
                raise ValueError("SemIf response has no corresponding charged backend call")
            result.append({"request_event_line": span["request_event_line"],
                           "kind": span["kind"], "role": request.get("role"),
                           "response_observed": response is not None,
                           "semif_charged_usage": Usage().model_dump(mode="json"),
                           "internal_request_count": 0,
                           "status": "request_logged_no_semif_charge" if response is None else "non_semif_response_no_semif_charge"})
            continue
        generation = span["kind"] == "generation_request"
        fixed_source = "semif.model.generate" if generation else "semif.direct"
        logical_field = "generation_calls" if generation else "judge_calls"
        charged = [entry for entry in entries if entry["source"] == fixed_source]
        returned = [entry for entry in entries if entry["source"] == fixed_source + ".result"]
        total = _total(entries)
        if total[logical_field] != 1 or total["model_requests"] != len(charged):
            raise ValueError("Logical call/internal request accounting differs from source charges")
        if len(charged) != len(returned):
            raise ValueError("Model request/result charge pairing incomplete")
        for charge, returned_entry in zip(charged, returned):
            detail = returned_entry["metadata"]
            if charge["usage"]["input_tokens"] != detail["input_tokens"]:
                raise ValueError("Internal request input tokens disagree with result metadata")
            if returned_entry["usage"]["output_tokens"] != detail["output_tokens"]:
                raise ValueError("Internal request output tokens disagree with result metadata")
        if response is not None:
            if generation:
                expected = request["n"]
                if len(response["texts"]) != expected:
                    raise ValueError("Proposal sample count differs from generation request n")
            else:
                expected = 1 if request["role"] == "candidate" and len(request["items"]) > 1 else len(request["items"])
                metadata = response["metadata"]
                if metadata["num_model_requests"] != expected or len(metadata["requests"]) != expected:
                    raise ValueError("Judge response metadata disagrees with role/item request count")
                if metadata.get("role") != request["role"] or len(response["scores"]) != len(request["items"]):
                    raise ValueError("Judge role or score slots changed")
                for detail, charge, returned_entry in zip(metadata["requests"], charged, returned):
                    if detail != returned_entry["metadata"] or detail["input_tokens"] != charge["usage"]["input_tokens"]:
                        raise ValueError("Judge internal-request metadata does not match ledger source/result entries")
            if len(charged) != expected:
                raise ValueError("Successful logical response has the wrong number of actual internal requests")
        result.append({"request_event_line": span["request_event_line"],
                       "response_event_line": span.get("response_event_line"), "kind": span["kind"],
                       "role": request.get("role"), "response_observed": response is not None,
                       "semif_charged_usage": total, "internal_request_count": len(charged),
                       "status": "complete_response" if response is not None else "started_without_complete_response"})
    return result


def partition_events(events: list[dict]) -> list[dict]:
    """Partition using actual phase changes plus committed transition events.

    A commit's postprocessing belongs to that commit until the next search
    operation. Any search/commit attempt without a transition is an uncommitted
    tail, including budget stops. Initial reset/verification remains initialization.
    """
    segments = [{"kind": "initialization", "entries": [], "events": [], "commits": []}]
    segment = segments[0]
    initialized = False
    active_role = None
    for line, event in enumerate(events, 1):
        kind, payload = event["event_type"], event["payload"]
        parent_phase = None
        if kind == "usage":
            parent_phase = payload["budget_phase"]
        elif kind in ("generation_request", "judge_request"):
            parent_phase = payload.get("phase")
        # Before initial_state even commit/replay verification is initialization.
        if initialized and parent_phase == "search" and (segment["kind"] == "initialization" or segment["commits"]):
            segment = {"kind": "attempt", "entries": [], "events": [], "commits": []}
            segments.append(segment)
        elif initialized and kind == "generation_request" and parent_phase == "commit" and (segment["kind"] == "initialization" or segment["commits"]):
            segment = {"kind": "attempt", "entries": [], "events": [], "commits": []}
            segments.append(segment)
        if kind == "initial_state":
            initialized = True
        if kind == "judge_request":
            active_role = payload["request"]["role"]
        elif kind == "generation_request":
            active_role = None
        if kind == "usage":
            entry = dict(payload)
            entry["event_line"] = line
            entry["category"] = _category(payload, active_role)
            segment["entries"].append(entry)
        else:
            segment["events"].append({"line": line, "event_type": kind, "payload": payload})
        if kind == "transition" and payload["phase"] == "commit":
            segment["commits"].append({"event_line": line, "state_id": payload["state"]["state_id"],
                                       "action": payload["candidate"]["action"],
                                       "objective_success": payload["step"]["success"]})
    result = []
    commit_index = 0
    for segment in segments:
        if len(segment["commits"]) > 1:
            raise ValueError("Multiple commits in a segment; phase boundary inference is invalid")
        if segment["commits"]:
            commit_index += 1
            label = "committed_step"
        else:
            label = "initialization" if segment["kind"] == "initialization" else "uncommitted_tail"
        groups = defaultdict(list)
        phases = defaultdict(list)
        for entry in segment["entries"]:
            groups[entry["category"]].append(entry)
            phases[entry["phase"] + "/" + entry["budget_phase"]].append(entry)
        searches = [event for event in segment["events"] if event["event_type"] == "search"]
        interruptions = [event for event in segment["events"] if event["event_type"] == "search_postprocessing_interrupted"]
        transitions = [event for event in segment["events"] if event["event_type"] == "transition"]
        request_events = [event for event in segment["events"] if event["event_type"] in ("generation_request", "judge_request")]
        result.append({"kind": label, "committed_step": commit_index if segment["commits"] else None,
                       "commit": segment["commits"][0] if segment["commits"] else None,
                       "usage": _total(segment["entries"]),
                       "categories": {key: _total(values) for key, values in groups.items()},
                       "phase_usage": {key: _total(values) for key, values in phases.items()},
                       "search_transition_count": sum(event["payload"]["phase"] == "search" for event in transitions),
                       "commit_transition_count": len(segment["commits"]),
                       "requested_generation_calls": sum(event["event_type"] == "generation_request" for event in request_events),
                       "requested_judge_calls_by_role": {role: sum(event["event_type"] == "judge_request" and event["payload"]["request"]["role"] == role for event in request_events)
                                                         for role in ("candidate", "evidence", "state_value", "stop")},
                       "requested_judge_items_by_role": {role: sum(len(event["payload"]["request"].get("items", ())) for event in request_events
                                                                  if event["event_type"] == "judge_request" and event["payload"]["request"]["role"] == role)
                                                        for role in ("candidate", "evidence", "state_value", "stop")},
                       "search_results": searches, "partial_backup_interruptions": interruptions,
                       "ledger_sequences": [entry["sequence"] for entry in segment["entries"]]})
    return result


def analyze(directory: Path) -> dict:
    audited = audit_run(directory)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    calls = audit_semif_calls(events)
    segments = partition_events(events)
    reconstructed = Usage()
    for segment in segments:
        reconstructed += Usage.model_validate(segment["usage"])
    if reconstructed != Usage.model_validate(audited["usage"]):
        # Float addition order can differ when aggregating independently.
        import math
        if any(not math.isclose(getattr(reconstructed, field), audited["usage"][field], rel_tol=1e-10, abs_tol=1e-8) for field in Usage.model_fields):
            raise ValueError("Segment totals do not reconcile with event/ledger/result audit")
    if sum(segment["kind"] == "committed_step" for segment in segments) != audited["committed_steps"]:
        raise ValueError("Segment commit count differs from independently audited actual actions")
    result = audited["result"]
    blocking = []
    for event in events:
        if event["event_type"] == "search":
            search = event["payload"]["result"]
            if search.get("stop_reason") == "budget_exhausted":
                blocking.append({"source": "search", "details": search})
        if event["event_type"] == "error":
            blocking.append({"source": "error", "details": event["payload"]["error"]})
    return {"source_directory": str(directory.resolve()), "termination_reason": result.termination_reason.value,
            "error": result.error, "objective_success": result.final_state.success,
            "committed_steps": audited["committed_steps"], "cost_complete": audited["cost_complete"],
            "within_budget": audited["within_budget"], "usage": audited["usage"],
            "phase_usage": audited["phase_usage"], "reconciliation_passed": True,
            "request_span_reconciliation": calls,
            "segments": segments, "budget_or_error_evidence": blocking,
            "closed_loop_commit_observed": audited["committed_steps"] > 0,
            "completed_requested_commits_or_objective_success": audited["committed_steps"] == result.config.max_steps or result.final_state.success,
            "limitations": ["Instrumented wall_seconds excludes model loading and is not end-to-end latency.",
                            "A partial committed loop or pre-commit budget stop does not prove task success.",
                            "Logical charged calls and actual internal requests are separate Usage fields.",
                            "Replay env_steps already belong to total env_steps; do not add them twice."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("termination_reason", "committed_steps", "objective_success", "cost_complete", "reconciliation_passed")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
