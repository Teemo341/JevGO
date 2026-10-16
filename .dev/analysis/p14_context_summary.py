"""Summarize existing P14a evidence without making model/environment requests."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".dev/tools"))
from p14_context import journal_accounting
from jevgo.evaluation import audit_run

ROLES = ("generation", "candidate", "evidence", "stop")


def read(path):
    return json.loads(path.read_text())


def http_summary(directory):
    path = directory / "http.jsonl"
    entries = [json.loads(line) for line in path.open()] if path.exists() else []
    finished = [row for row in entries if row["event"] == "request_finished"]
    sent = [row for row in finished if row["network_sent"]]
    roles = {role: {"requests": 0, "known_input_tokens": 0, "known_output_tokens": 0,
                    "unknown_input_requests": 0, "unknown_output_requests": 0,
                    "repair_requests": 0, "finish_reasons": {}} for role in ROLES}
    for row in sent:
        role = roles[row["role"]]
        role["requests"] += 1
        role["repair_requests"] += bool(row.get("repair"))
        try:
            body = json.loads(row["raw_response"])
        except (ValueError, KeyError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        raw = body.get("usage", {})
        if not isinstance(raw, dict):
            raw = {}
        for remote, label in (("prompt_tokens", "input"), ("completion_tokens", "output")):
            value = raw.get(remote)
            if type(value) is int and value >= 0:
                role[f"known_{label}_tokens"] += value
            else:
                role[f"unknown_{label}_requests"] += 1
        for choice in body.get("choices", []):
            reason = choice.get("finish_reason") or "unknown"
            role["finish_reasons"][reason] = role["finish_reasons"].get(reason, 0) + 1
    peaks = {}
    for role in ROLES:
        group = [row for row in finished if row["role"] == role]
        if group:
            peak = max(group, key=lambda row: row["total_window_tokens"])
            peaks[role] = {"request_number": peak["request_number"],
                           "input_tokens": peak["offline_input_tokens"],
                           "output_reserve": peak["requested_output_tokens_per_sequence"],
                           "total_window_tokens": peak["total_window_tokens"],
                           "remaining_tokens": peak["remaining_tokens"],
                           "network_sent": peak["network_sent"]}
    return {"roles": roles, "peaks_including_rejected_attempts": peaks,
            "http_requests": len(sent), "local_rejected_or_budget_blocked": len(finished)-len(sent),
            "known_input_tokens": sum(v["known_input_tokens"] for v in roles.values()),
            "known_output_tokens": sum(v["known_output_tokens"] for v in roles.values()),
            "usage_complete": all(v["unknown_input_requests"] == v["unknown_output_requests"] == 0
                                  for v in roles.values())}


def reconcile_totals(data, ledger, result, audit=None):
    observed = (data["http_requests"], data["known_input_tokens"], data["known_output_tokens"])
    for expected in (ledger["used"], result["usage"], *([audit["usage_from_events"]] if audit else [])):
        assert observed == tuple(expected[k] for k in ("model_requests", "input_tokens", "output_tokens")), "Raw HTTP totals disagree with current ledger/result/audit"
    if "usage_complete" in result:
        assert data["usage_complete"] == result["usage_complete"], "Result cost completeness disagrees with raw HTTP"
    if audit is not None:
        assert data["usage_complete"] == audit["usage_complete"], "Cost completeness disagrees"
        assert audit["termination_reason"] == result["termination_reason"], "Stale termination audit"
        assert audit["raw_error"] == result["error"], "Stale error audit"
        assert audit["committed_steps"] == len(result["final_state"]["actions"]), "Stale action audit"


def validate_reserves(directory, generation_reserve):
    path = directory / "http.jsonl"
    for line in path.open() if path.exists() else []:
        row = json.loads(line)
        if row["event"] != "request_preflight":
            continue
        expected = generation_reserve if row["role"] == "generation" else 1
        assert row["requested_output_tokens_per_sequence"] == expected, "Unexpected role output reserve"


def summarize(root):
    conditions, protocols, missing = [], [], []
    for window, memory in ((4096, 4), (8192, 32)):
        protocol = root / f"{window}-protocol"
        if (protocol / "result.json").exists():
            result = read(protocol / "result.json")
            config = read(protocol / "config.json")
            requests = read(protocol / "requests.json")
            assert config["model"]["max_model_len"] == window
            assert requests[0]["max_new_tokens"] == 128
            journal_directory = root / f"{window}-protocol-http"
            data = http_summary(journal_directory)
            ledger = read(protocol / "ledger.json")
            reconcile_totals(data, ledger, result)
            journal_path = journal_directory / "http.jsonl"
            journal = [json.loads(line) for line in journal_path.open()] if journal_path.exists() else []
            journal_audit = journal_accounting(journal, ledger)
            validate_reserves(journal_directory, 128)
            protocols.append({"window": window, "generation_output_reserve": 128,
                              "ok": result["ok"] and result["usage_complete"] and data["usage_complete"] and journal_audit["ok"],
                              "journal_accounting": journal_audit, "cost": data})
        else:
            missing.append(f"{window}-protocol")
        for method in ("B0", "B3"):
            name = f"{window}-{method}"
            if not (root / f"{name}-audit.json").exists():
                missing.append(name)
                continue
            audit = read(root / f"{name}-audit.json")
            result = read(root / name / "result.json")
            ledger = read(root / name / "ledger.json")
            config = read(root / name / "config.json")
            assert config["metadata"]["model_config"]["max_model_len"] == window
            assert result["config"]["method"] == method and result["config"]["options"]["memory_records"] == memory
            assert result["config"]["options"]["max_new_tokens"] == 256
            data = http_summary(root / f"{name}-http")
            reconcile_totals(data, ledger, result, audit)
            fresh_audit = audit_run(root / name)
            assert fresh_audit["cost_complete"] == audit["usage_complete"]
            assert all(math.isclose(fresh_audit["usage"][key], audit["usage_from_events"][key], rel_tol=1e-10, abs_tol=1e-8)
                       for key in fresh_audit["usage"])
            journal_path = root / f"{name}-http/http.jsonl"
            journal = [json.loads(line) for line in journal_path.open()] if journal_path.exists() else []
            fresh_journal = journal_accounting(journal, ledger)
            assert fresh_journal == audit["journal_accounting"], "Stale journal audit"
            validate_reserves(root / f"{name}-http", 256)
            peak = max(data["peaks_including_rejected_attempts"].values(), key=lambda row: row["total_window_tokens"], default=None)
            window_runtime = root / f"runtime-{window}"
            ready = read(window_runtime / "ready.json") if (window_runtime / "ready.json").exists() else None
            cleanup = read(window_runtime / "cleanup.json") if (window_runtime / "cleanup.json").exists() else None
            launch = read(window_runtime / "launch.json")
            command_exit = read(window_runtime / f"{method}-command/exit.json")
            provenance = read(root / f"{name}-http/provenance.json")
            conditions.append({"condition": name, "window": window, "memory_records": memory,
                "inference_commit": provenance["commit"],
                "run_exit_code": command_exit["rc"],
                "run_command_elapsed_seconds": command_exit["elapsed_seconds"],
                "method": method, "generation_output_reserve": 256, "judge_output_reserve": 1,
                "committed_steps": audit["committed_steps"], "objective_success": audit["objective_success"],
                "termination_reason": audit["termination_reason"], "raw_error": audit["raw_error"],
                "peak_request": peak, "cost": data, "usage": audit["usage_from_events"],
                "event_ledger_result_reconcile": audit["event_ledger_result_reconcile"],
                "journal_accounting_ok": audit["journal_accounting"]["ok"],
                "offline_service_prompt_counts_match": audit["offline_service_prompt_counts_match"],
                "original_role_frontier_9_actions_19_memories": audit["successful_frontier_9_actions_19_memories"],
                "context_condition_resolved": audit["context_condition_resolved"],
                "technical_closed_loop_pass": audit["technical_closed_loop_pass"],
                "actions": result["final_state"]["actions"],
                "sampled_window_gpu_peak_mib": cleanup["sampled_peak_memory_used_mib"] if cleanup else None,
                "service_ready_elapsed_seconds": ready["elapsed_seconds"] if ready else None,
                "gpu_peak_scope": "entire window service startup/protocol/B0/B3, not condition-specific; includes preallocated pool",
                "launch": launch})
    return {"conditions": conditions, "protocols": protocols, "missing": missing,
            "analysis_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "loop_totals": {key: sum(row["cost"][key] for row in conditions) for key in
                            ("http_requests", "known_input_tokens", "known_output_tokens")},
            "protocol_totals": {key: sum(row["cost"][key] for row in protocols) for key in
                                ("http_requests", "known_input_tokens", "known_output_tokens")},
            "model_requests_made_by_analysis": 0, "research_effectiveness": "pending",
            "historical_failures_preserved": True,
            "limitations": ["Single development task/seed, not an effectiveness comparison.",
                            "GPU peaks are 5-second nvidia-smi observations including vLLM pools, not exact or minimum VRAM.",
                            "Memory4 changes the condition and cannot resolve the historical memory32 failure.",
                            "Unknown costs remain unknown; token subtotals are not substituted for missing usage."]}


def markdown(summary):
    def known_total(cost, kind):
        value = cost[f"known_{kind}_tokens"]
        return str(value) if cost["usage_complete"] else f">={value} (unknown remainder)"
    lines = ["# P14a 9B pretrial results", "", "Development evidence only; no effectiveness claim.", "",
             "| Window / memory | Method | Commit steps | Objective success | Stop | Requests | Env / replay steps | Input / output tokens | Peak prompt + reserve | Remaining | Original-role 9 actions / 19 memories | Closed loop |",
             "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in summary["conditions"]:
        cost, peak = row["cost"], row["peak_request"]
        peak_text = f"{peak['input_tokens']} + {peak['output_reserve']}" if peak else "not observed"
        remaining = peak['remaining_tokens'] if peak else "not observed"
        usage = row.get('usage', {})
        env_cost = f"{usage.get('env_steps', 'unknown')} / {usage.get('replay_steps', 'unknown')}"
        lines.append(f"| {row['window']} / {row['memory_records']} | {row['method']} | {row['committed_steps']} | {row['objective_success']} | {row['termination_reason']} | {cost['http_requests']} | {env_cost} | {known_total(cost, 'input')} / {known_total(cost, 'output')} | {peak_text} | {remaining} | {row['original_role_frontier_9_actions_19_memories']} | {row['technical_closed_loop_pass']} |")
    lines += ["", "Per-role costs use actual raw service usage; generation n=4 remains one HTTP request. Repair requests are included in generation.", "", "| Condition | Role | Requests | Known input tokens | Known output tokens | Repairs | Unknown input / output requests |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["conditions"]:
        for name, role in row["cost"]["roles"].items():
            lines.append(f"| {row['condition']} | {name} | {role['requests']} | {role['known_input_tokens']} | {role['known_output_tokens']} | {role['repair_requests']} | {role['unknown_input_requests']} / {role['unknown_output_requests']} |")
    lines += ["", "Protocol costs are separate (generation reserve 128; loop generation reserve 256):", ""]
    for row in summary["protocols"]:
        cost = row["cost"]
        lines.append(f"- Window {row['window']}: ok={row['ok']}, {cost['http_requests']} requests, {known_total(cost, 'input')} input / {known_total(cost, 'output')} output tokens.")
    lines += ["", "Window-wide GPU samples include startup and both methods, including preallocated KV/runtime pools; these are not minimum VRAM requirements or exact allocator peaks.", ""]
    for window in (4096, 8192):
        row = next((row for row in summary["conditions"] if row["window"] == window), None)
        if row: lines.append(f"- Window {window}: sampled memory used peak {row['sampled_window_gpu_peak_mib']} MiB; service ready after {row.get('service_ready_elapsed_seconds')} seconds (loading/compilation/warmup, separate from API usage); see launch/cleanup and full samples.")
    if summary.get('analysis_commit'):
        lines += ["", f"Analysis commit: {summary['analysis_commit']}. Inference commits: {', '.join(sorted({row['inference_commit'] for row in summary['conditions']}))}."]
    lines += ["", "Failures and limitations:", ""]
    for row in summary["conditions"]:
        if row["raw_error"]: lines.append(f"- {row['condition']}: {row['raw_error']}")
    lines += [f"- {text}" for text in summary["limitations"]]
    if summary["missing"]: lines.append("- Still missing: " + ", ".join(summary["missing"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(result))


if __name__ == "__main__":
    main()
