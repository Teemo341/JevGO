"""Independently reconcile finite P14 condition costs and mechanism exposure."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def totals(entries):
    result = defaultdict(float)
    for entry in entries:
        for field, value in entry["usage"].items():
            result[field] += value
    return {k: v if k == "wall_seconds" else int(v) for k, v in result.items()}


def summarize_condition(path):
    result, ledger = read(path / "result.json"), read(path / "ledger.json")
    logical = rows(path / "logical.jsonl")
    requests = [r for r in logical if r["event"] == "request"]
    finished = [r for r in logical if r["event"] == "finished"]
    accounted = totals(ledger["entries"])
    ledger_matches = all(math.isclose(accounted.get(k, 0), v, rel_tol=1e-10, abs_tol=1e-7)
                         for k, v in ledger["used"].items())
    role_entries, unknown = defaultdict(list), []
    for call in finished:
        role = call["request"].get("role", "generation")
        if role == "generation":
            role += ":" + (call.get("generation_origin") or {}).get("source", "unspecified")
        role_entries[role].extend(call["ledger_entries"])
        for entry in call["ledger_entries"]:
            if entry["metadata"].get("usage_complete") is False:
                unknown.append({"call": call["call"], "entry": entry})
    model_cost = totals([e for call in finished for e in call["ledger_entries"]])
    fields = ("model_requests", "input_tokens", "output_tokens", "generation_calls", "judge_calls")
    model_matches = all(model_cost.get(k, 0) == ledger["used"][k] for k in fields)
    event_path = path / "episode/events.jsonl" if (path / "episode/events.jsonl").exists() else path / "probe-events.jsonl"
    events = rows(event_path)
    proposals = [e["payload"] for e in events if e["event_type"] == "proposal_samples"]
    candidates = [e["payload"] for e in events if e["event_type"] == "candidates"]
    searches = [e["payload"]["result"] for e in events if e["event_type"] == "search"]
    memories = [e["payload"] for e in events if e["event_type"] == "memory_selection"]
    http = rows(path / "http.jsonl")
    raw_http = [e for e in http if e["event"] == "request_finished" and e.get("network_sent")]
    raw_known = {"input_tokens": 0, "output_tokens": 0}
    raw_unknown = []
    for row in raw_http:
        try:
            body = json.loads(row["raw_response"])
            usage = body.get("usage") or {}
        except (KeyError, ValueError, AttributeError):
            usage = {}
        for remote, local in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
            value = usage.get(remote)
            if type(value) is int and value >= 0:
                raw_known[local] += value
            else:
                raw_unknown.append({"request_number": row["request_number"], "field": local})
    checks = {"ledger_sum_matches": ledger_matches,
              "logical_request_finish_pairs_match": [r["call"] for r in requests] == [r["call"] for r in finished],
              "logical_model_cost_matches_ledger": model_matches}
    if http:
        checks.update(http_request_count_matches=len(raw_http) == ledger["used"]["model_requests"],
                      http_known_tokens_match=all(raw_known[k] == ledger["used"][k] for k in raw_known))
    observation = None
    if (path / "state.json").exists():
        observation = read(path / "state.json")["observation"]
    elif events:
        initial = next((e for e in events if e["event_type"] == "initial_state"), None)
        if initial:
            observation = initial["payload"]["state"]["observation"]
    digest = lambda obj: hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"path": str(path.resolve()), "result": result, "checks": checks,
            "cost_by_role": {k: totals(v) for k, v in role_entries.items()}, "unknown_usage": unknown,
            "http_unknown_usage": raw_unknown, "model_cost_complete": not unknown and not raw_unknown,
            "samples": sum(len(p["texts"]) for p in proposals),
            "strict_valid_samples": sum(sum(s["accepted"] for s in p.get("samples", [])) for p in proposals),
            "candidate_calls": len(candidates), "candidate_unique_action_counts": [len(p["candidates"]) for p in candidates],
            "singleton_calls": sum(len(p["candidates"]) == 1 for p in candidates),
            "root_searches": [{"candidate_edges": len(s["rootstats"]),
                                "visited_edges": sum(edge["N"] > 0 for edge in s["rootstats"]),
                                "stats": s["rootstats"]} for s in searches],
            "memory_selections": memories,
            "stop_advice_events": sum(e["event_type"] == "completion_advice" for e in events),
            "public_initial_observation_sha256": digest(observation) if observation else None,
            "effectiveness_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions, batches = [], []
    for record in sorted(args.root.glob("*/execution-record.json")):
        execution = read(record)
        manifest = execution.get("manifest")
        if not isinstance(manifest, dict) or "conditions" not in manifest:
            continue
        batches.append({"path": str(record.resolve()), "commit": execution["commit"],
                        "conditions": [c["id"] for c in manifest["conditions"]]})
        for condition in manifest["conditions"]:
            path = record.parent / condition["id"]
            if (path / "result.json").exists():
                conditions.append(summarize_condition(path))
            else:
                conditions.append({"path": str(path.resolve()), "not_executed_or_incomplete": True,
                                   "planned_condition": condition})
    completed = [c for c in conditions if "result" in c]
    data = {"batches": batches, "conditions": conditions,
            "actual_usage": totals([{"usage": c["result"]["usage"]} for c in completed]),
            "accounting_checks_pass": all(all(c["checks"].values()) for c in completed),
            "all_usage_complete": all(c["model_cost_complete"] for c in completed),
            "scope": "New finite policy experiments only; excludes capacity probes, CPU setup, historical inference and model/service startup.",
            "P15_started": False, "effectiveness_verified": False}
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"conditions": len(conditions), "complete_results": len(completed),
                      "accounting_checks_pass": data["accounting_checks_pass"], "actual_usage": data["actual_usage"]}))


if __name__ == "__main__":
    main()
