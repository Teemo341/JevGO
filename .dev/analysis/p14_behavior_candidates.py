#!/usr/bin/env python3
"""CPU-only, source-linked audit of frozen P14 candidate generation artifacts.

No model loading, tokenizer invocation, network access, or mutation of inputs.
Run with explicit --repo, --artifacts and --output after moving this file.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(value).hexdigest()


def read_events(path):
    return [(i, json.loads(line)) for i, line in enumerate(path.open(), 1)]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_lines(path, values):
    with path.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def production_parse(text, allowed, Candidate):
    """Mirror research.py's parser, separately from independent strict validation."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict) or set(obj) - {"action", "hypothesis", "rationale"}:
            raise ValueError("Unexpected proposal fields")
        candidate = Candidate(**obj)
        if allowed and candidate.action.key() not in allowed:
            raise ValueError("Action is not in the public admissible action set")
        return candidate, None
    except (ValueError, TypeError) as error:
        return None, str(error)


def audit(condition, relative, http_relative, args, strict, Candidate, Action):
    root = args.artifacts
    path = root / relative / "events.jsonl"
    events = read_events(path)
    config = json.loads((root / relative / "config.json").read_text())
    out = args.output / condition
    out.mkdir(parents=True, exist_ok=True)
    http = {}
    if http_relative:
        for no, row in read_events(root / http_relative / "http.jsonl"):
            if row.get("role") == "generation" and row["event"] == "request_finished":
                http[row["request_number"]] = (no, row)
    http_order = iter(sorted(http))
    candidate_parents = defaultdict(set)
    candidate_transition_lines = defaultdict(list)
    for no, event in events:
        if event["event_type"] == "transition":
            payload = event["payload"]
            cid = payload["candidate"]["candidate_id"]
            candidate_parents[cid].add(payload["parent_state_id"])
            candidate_transition_lines[cid].append(no)
    calls, samples = [], []
    active = None
    committed = 0
    incoming = {}
    for no, event in events:
        kind, payload = event["event_type"], event["payload"]
        branch = event["branch_id"]
        if kind == "initial_state":
            incoming[branch] = {"line": no, "kind": kind, "state": payload["state"]}
        elif kind == "transition":
            incoming[branch] = {"line": no, "kind": kind, **payload}
            if payload["phase"] == "commit":
                committed += 1
        elif kind == "generation_request":
            assert active is None or active.get("candidate_event_line"), (condition, no)
            request = payload["request"]
            public = json.JSONDecoder().raw_decode(request["prompt"].split("Public state:\n", 1)[1])[0]
            active = {
                "condition": condition, "call_index": len(calls) + 1,
                "events_path": str(path), "generation_event_line": no,
                "generation_event_id": event["event_id"], "branch_id": branch,
                "scope": "root" if branch == "root" else "branch",
                "planning_commit_number": committed + 1,
                "absolute_action_depth": len(public["actions"]),
                "search_depth_from_current_root": len(public["actions"]) - committed,
                "attempt": payload["attempt"], "request": request,
                "public_state": public, "prompt_sha256": digest(request["prompt"].encode()),
                "observation_sha256": digest(public["observation"]),
                "seed_formula_matches": request["seed"] == config["run"]["seed"] + 2 * len(public["actions"]) + payload["attempt"],
                "implementation_child_seeds": [request["seed"] + i for i in range(request["n"])],
                "child_seed_evidence": "backend source; child RNG states not journaled",
                "incoming_transition_or_initial": incoming.get(branch),
                "sample_numbers": [], "charged_model_generation_requests": 0,
            }
            if http:
                number = next(http_order)
                http_line, item = http[number]
                raw = json.loads(item["raw_response"])
                choices = sorted(raw["choices"], key=lambda choice: choice["index"])
                payload_http = item["request"]
                matches = {
                    "n": payload_http["n"] == request["n"],
                    "seed": payload_http["seed"] == request["seed"],
                    "temperature": payload_http["temperature"] == request["temperature"],
                    "max_tokens": payload_http["max_tokens"] == request["max_new_tokens"],
                    "messages": payload_http["messages"] == [
                        {"role": "system", "content": request["system"]},
                        {"role": "user", "content": request["prompt"]}],
                    "choice_indices": [c["index"] for c in choices] == list(range(request["n"])),
                    "thinking_disabled": payload_http.get("chat_template_kwargs") == {"enable_thinking": False},
                }
                assert all(matches.values()), (condition, no, matches)
                active.update(http_request_number=number, http_finished_line=http_line,
                              http_path=str(root / http_relative / "http.jsonl"),
                              actual_http_payload=payload_http, http_payload_matches=matches,
                              http_aggregate_usage=raw["usage"], http_choices=choices)
            calls.append(active)
        elif active is not None and kind == "usage":
            if payload["source"] == "semif.model.generate":
                active["charged_model_generation_requests"] += payload["usage"]["model_requests"]
        elif kind == "proposal_samples":
            assert active is not None
            active["proposal_event_line"] = no
            active["proposal_event_errors"] = payload["errors"]
            assert len(payload["texts"]) == active["request"]["n"]
            allowed = strict.public_actions(active["request"])
            allowed_keys = {Action(**a).key() for a in allowed}
            grouped = {}
            for index, text in enumerate(payload["texts"]):
                production, error = production_parse(text, allowed_keys, Candidate)
                validation = strict.validate_proposal(text, allowed)
                sample = {
                    "sample_number": len(samples) + 1, "call_index": active["call_index"],
                    "events_path": str(path), "proposal_event_line": no, "choice_index": index,
                    "raw_text": text, "raw_sha256": digest(text.encode()),
                    "strict": validation, "production_valid": production is not None,
                    "production_error": error,
                    "action": production.action.model_dump(mode="json") if production else None,
                    "hypothesis": production.hypothesis if production else None,
                    "rationale": production.rationale if production else None,
                    "finish_reason": None,
                    "choice_actual_tokens": None,
                }
                if http:
                    choice = active["http_choices"][index]
                    assert choice["message"]["content"] == text
                    sample.update(finish_reason=choice["finish_reason"],
                                  stop_reason=choice.get("stop_reason"),
                                  http_path=active["http_path"],
                                  http_finished_line=active["http_finished_line"],
                                  http_request_number=active["http_request_number"])
                if production:
                    key = production.action.key()
                    if key not in grouped:
                        grouped[key] = {"action": sample["action"], "hypothesis": sample["hypothesis"],
                                        "rationale": sample["rationale"], "sample_count": 0,
                                        "source_sample_number": sample["sample_number"]}
                    grouped[key]["sample_count"] += 1
                active["sample_numbers"].append(sample["sample_number"])
                samples.append(sample)
            active["recomputed_grouped"] = grouped
        elif kind == "candidates":
            assert active is not None
            candidates = payload["candidates"]
            expected = active["recomputed_grouped"]
            assert [Action(**c["action"]).key() for c in candidates] == list(expected)
            valid_count = sum(c["sample_count"] for c in expected.values())
            score_sum = sum(c["judge_score"] or 0 for c in candidates)
            for candidate in candidates:
                group = expected[Action(**candidate["action"]).key()]
                for field in ("action", "hypothesis", "rationale", "sample_count"):
                    assert group[field] == candidate[field], (condition, no, field)
                preference = (candidate["judge_score"] / score_sum) if score_sum else 1 / len(candidates)
                prior = .5 * (candidate["sample_count"] / valid_count + preference)
                assert abs(candidate["prior"] - prior) < 1e-12
            parents = set().union(*(candidate_parents[c["candidate_id"]] for c in candidates))
            assert len(parents) <= 1, (condition, no, parents)
            active.update(candidate_event_line=no, final_candidates=candidates,
                          exact_generation_state_id=next(iter(parents)) if parents else None,
                          state_id_evidence="parent_state_id of transition using this emitted candidate; null if unexecuted",
                          candidate_transition_lines={c["candidate_id"]: candidate_transition_lines[c["candidate_id"]] for c in candidates},
                          policy_valid_samples=valid_count,
                          strict_valid_samples=sum(samples[i - 1]["strict"]["passed"] for i in active["sample_numbers"]),
                          distinct_legal_action_count=len(expected), final_candidate_count=len(candidates),
                          final_action_names=[c["action"]["name"] for c in candidates],
                          raw_unique_texts=len({samples[i - 1]["raw_sha256"] for i in active["sample_numbers"]}),
                          parser_dedup_and_prior_match=True)
            if not http:
                assert active["charged_model_generation_requests"] == active["request"]["n"]
    assert all(c.get("candidate_event_line") for c in calls)
    if http:
        assert len(http) == len(calls)
    seed_groups = []
    for depth in sorted({c["absolute_action_depth"] for c in calls}):
        group = [c for c in calls if c["absolute_action_depth"] == depth]
        seed_groups.append({"absolute_action_depth": depth, "call_indices": [c["call_index"] for c in group],
                            "base_seeds": sorted({c["request"]["seed"] for c in group}),
                            "distinct_branches": len({c["branch_id"] for c in group}),
                            "distinct_prompts": len({c["prompt_sha256"] for c in group}),
                            "child_seeds": sorted({s for c in group for s in c["implementation_child_seeds"]})})
    identical_inputs = defaultdict(list)
    for call in calls:
        identical_inputs[digest(call["request"])].append(call)
    repeats = []
    for group in identical_inputs.values():
        if len(group) > 1:
            repeats.append({"call_indices": [c["call_index"] for c in group],
                            "distinct_output_tuples": len({tuple(samples[i-1]["raw_sha256"] for i in c["sample_numbers"]) for c in group})})
    transitions = [(no, e["payload"]) for no, e in events if e["event_type"] == "transition"]
    commits = []
    by_candidate = {c["candidate_id"]: (call, c) for call in calls for c in call["final_candidates"]}
    for no, transition in transitions:
        if transition["phase"] == "commit":
            call, c = by_candidate[transition["candidate"]["candidate_id"]]
            source = call["recomputed_grouped"][Action(**c["action"]).key()]["source_sample_number"]
            commits.append({"transition_event_line": no, "action": c["action"],
                            "call_index": call["call_index"], "source_sample_number": source,
                            "strict_source_passed": samples[source-1]["strict"]["passed"],
                            "action_success": transition["step"].get("action_success"),
                            "objective_success": transition["step"]["success"]})
    summary = {
        "condition": condition, "events_path": str(path), "source_runtime_commit": "955df143e7e2db059cb870e8d4fed9e9741621a2",
        "logical_generation_calls": len(calls), "raw_samples": len(samples),
        "repair_calls": sum(c["attempt"] > 0 for c in calls),
        "strict_valid": sum(s["strict"]["passed"] for s in samples),
        "policy_valid": sum(s["production_valid"] for s in samples),
        "strict_error_stages": dict(Counter(s["strict"]["error_stage"] for s in samples if not s["strict"]["passed"])),
        "legal_action_names": dict(Counter(s["action"]["name"] for s in samples if s["production_valid"])),
        "raw_lexical_action_names_not_parse_validation": dict(Counter(name for s in samples for name in __import__('re').findall(r'"name"\s*:\s*"([^"\\]*)"', s["raw_text"]))),
        "final_candidate_sets": dict(Counter("+".join(c["final_action_names"]) for c in calls)),
        "n_to_legal_to_unique_histogram": dict(Counter(f'{c["request"]["n"]}->{c["policy_valid_samples"]}->{c["distinct_legal_action_count"]}->{c["final_candidate_count"]}' for c in calls)),
        "root_branch": {scope: {"calls": len(part), "raw_samples": sum(c["request"]["n"] for c in part),
                                 "strict_valid": sum(c["strict_valid_samples"] for c in part),
                                 "candidate_sets": dict(Counter("+".join(c["final_action_names"]) for c in part))}
                        for scope in ("root", "branch") if (part := [c for c in calls if c["scope"] == scope])},
        "all_seed_formulas_match": all(c["seed_formula_matches"] for c in calls),
        "seed_n_temperature_cap_values": sorted({(c["request"]["n"], c["request"]["temperature"], c["request"]["max_new_tokens"]) for c in calls}),
        "all_parser_dedup_prior_checks_match": all(c["parser_dedup_and_prior_match"] for c in calls),
        "calls_with_exact_generation_state_id": sum(c["exact_generation_state_id"] is not None for c in calls),
        "finish_reason_counts": dict(Counter(s["finish_reason"] or "not_captured" for s in samples)),
        "all_choice_token_usage_unknown": all(s["choice_actual_tokens"] is None for s in samples),
        "identical_request_repeat_groups": repeats,
        "commits": commits,
    }
    # Remove transient response choice duplication; raw strings remain in samples and the immutable HTTP journal.
    for call in calls:
        call.pop("http_choices", None)
    write_json(out / "summary.json", summary)
    write_json(out / "seed-groups.json", seed_groups)
    write_lines(out / "calls.jsonl", calls)
    write_lines(out / "samples.jsonl", samples)
    columns = ["call_index", "scope", "planning_commit_number", "absolute_action_depth", "search_depth_from_current_root", "branch_id", "exact_generation_state_id", "generation_event_line", "proposal_event_line", "candidate_event_line", "policy_valid_samples", "strict_valid_samples", "distinct_legal_action_count", "final_candidate_count", "raw_unique_texts"]
    with (out / "call-funnel.csv").open("w") as handle:
        writer = csv.DictWriter(handle, columns + ["n", "seed", "temperature", "final_action_names"])
        writer.writeheader()
        for call in calls:
            writer.writerow({**{key: call[key] for key in columns},
                             **{key: call["request"][key] for key in ("n", "seed", "temperature")},
                             "final_action_names": "+".join(call["final_action_names"])})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.repo, args.artifacts, args.output = (p.resolve() for p in (args.repo, args.artifacts, args.output))
    sys.path.insert(0, str(args.repo))
    spec = importlib.util.spec_from_file_location("p14_strict_validator", args.repo / ".dev/tools/p14_semif_pretrial.py")
    strict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strict)
    from jevgo.core import Candidate, Action
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for condition, relative, http_relative in (
        ("9b-4096-B3", "a/4096-B3", "a/4096-B3-http"),
        ("semif-env20-sim8", "c/loops/horizon20-output256-seed0-sim8", None),
        ("semif-env20-sim2", "c/loops/horizon20-output256-seed0-sim2", None),
    ):
        summaries.append(audit(condition, relative, http_relative, args, strict, Candidate, Action))
    write_json(args.output / "summary.json", summaries)
    sources = []
    paths = [args.repo / p for p in ("jevgo/research.py", "jevgo/core.py", "jevgo/models/semif.py", "jevgo/models/autotrust.py", ".dev/tools/p14_semif_pretrial.py")]
    paths += [args.artifacts / rel / f for rel in ("a/4096-B3", "c/loops/horizon20-output256-seed0-sim8", "c/loops/horizon20-output256-seed0-sim2") for f in ("events.jsonl", "config.json")]
    paths += [args.artifacts / "a/4096-B3-http/http.jsonl", Path(__file__).resolve()]
    for path in paths:
        row = {"path": str(path), "sha256": digest(path.read_bytes()), "bytes": path.stat().st_size}
        if path.is_relative_to(args.repo) and str(path.relative_to(args.repo)).startswith("jevgo/"):
            historical = subprocess.run(["git", "show", "955df143e7e2db059cb870e8d4fed9e9741621a2:" + str(path.relative_to(args.repo))], cwd=args.repo, capture_output=True, check=True).stdout
            row.update(runtime_955_sha256=digest(historical), unchanged_since_runtime_955=historical == path.read_bytes())
        sources.append(row)
    write_json(args.output / "sources.json", sources)
    print(json.dumps([{k: x[k] for k in ("condition", "logical_generation_calls", "raw_samples", "strict_valid", "legal_action_names", "final_candidate_sets", "all_parser_dedup_prior_checks_match")} for x in summaries], indent=2))


if __name__ == "__main__":
    main()
