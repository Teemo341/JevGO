"""P14a development-only 9B context preparation, admission and request audit.

No model weights are loaded by prepare/offline. Runtime calls the unchanged
factory/runner with a development backend subclass; rejected payloads are never
truncated or sent. Offline lengths never replace service usage.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx
import yaml

from jevgo.core import BudgetExceeded, BudgetLedger, GenerationRequest, JudgeRequest, ModelFailure, RunConfig
from jevgo.models.autotrust import AutoTrustBackend, load_model_config

REPAIR_SUFFIX = "\nThe previous proposals were invalid. Return only the required JSON object."
ARMS = ((4096, 4), (8192, 32))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_json(path: Path, value: object) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def provenance(assets: Path) -> dict:
    config, model_dir = load_model_config("jev_9b", assets)
    return {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "cwd": str(Path.cwd()), "source_root": str(ROOT), "python": sys.executable,
        "assets_root": str(assets.resolve()), "model_config": config,
        "agent_configuration": {"model": "gpt-6-astra", "reasoning_effort": "ultra"},
        "device_environment": {key: os.environ.get(key) for key in
                               ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")},
        "sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                   (ROOT / "jevgo/models/autotrust.py", ROOT / "jevgo/research.py", Path(__file__))},
        "tokenizer_sha256": {name: digest(model_dir / name) for name in
                             ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
                             if (model_dir / name).is_file()},
    }


def extract_public(text: str) -> tuple[dict | None, int | None, int | None]:
    """Decode exactly the public-state object, retaining any repair/item suffix."""
    marker = "Public state:\n"
    if marker not in text:
        return None, None, None
    start = text.index(marker) + len(marker)
    value, count = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Public state must be an object")
    return value, start, start + count


def frontier(text: str) -> dict:
    state, _, _ = extract_public(text)
    return {"branch_actions": len(state.get("actions", [])) if state else None,
            "memory_entries": len(state.get("memory", [])) if state else None}


class Lengths:
    def __init__(self, assets: Path):
        from transformers import AutoTokenizer
        _, directory = load_model_config("jev_9b", assets)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)

    def measure(self, path: str, payload: dict, window: int) -> dict:
        if any(key in payload for key in ("truncate_prompt_tokens", "truncation")):
            raise ValueError("Prompt truncation is forbidden")
        if path == "/v1/chat/completions":
            rendered = self.tokenizer.apply_chat_template(
                payload["messages"], tokenize=False, add_generation_prompt=True,
                **payload.get("chat_template_kwargs", {}))
            raw = payload["messages"][-1]["content"]
            template = "checkpoint_chat_template/add_generation_prompt/enable_thinking=false"
        elif path == "/v1/completions":
            if payload.get("add_special_tokens") is not False:
                raise ValueError("Expected the 9B bare-v1 no-special-token completion")
            rendered = raw = payload["prompt"]
            template = "bare-v1/add_special_tokens=false"
        else:
            raise ValueError(f"Unsupported 9B endpoint: {path}")
        ids = self.tokenizer.encode(rendered, add_special_tokens=False, truncation=False)
        reserved = payload["max_tokens"]
        return {"offline_input_tokens": len(ids), "requested_output_tokens_per_sequence": reserved,
                "requested_output_tokens_all_sequences": reserved * payload.get("n", 1),
                "total_window_tokens": len(ids) + reserved, "max_model_len": window,
                "remaining_tokens": window - len(ids) - reserved,
                "fits": len(ids) + reserved <= window, "template": template,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                **frontier(raw)}


class CaptureBackend(AutoTrustBackend):
    """Use production payload builders but intercept before HTTP or usage charging."""
    def __init__(self, assets: Path):
        super().__init__("jev_9b", assets_root=assets, client=object(), max_model_len=262144)
        self.payloads = []

    def _request(self, path, payload, parse, call, ledger, phase, **kwargs):
        self.payloads.append({"endpoint": path, "payload": deepcopy(payload)})
        if path == "/v1/chat/completions":
            return tuple("capture-only" for _ in range(payload["n"]))
        count = len(payload["allowed_token_ids"])
        return [1 / count] * count, {}

    def expand(self, event: dict) -> list[dict]:
        self.payloads = []
        data = event["payload"]["request"]
        if event["event_type"] == "generation_request":
            self.generate(GenerationRequest.model_validate(data), ledger=BudgetLedger())
        else:
            self.score(JudgeRequest.model_validate(data), ledger=BudgetLedger())
        return self.payloads


def historical_sources(history: Path) -> list[dict]:
    result = []
    for method in ("B0", "B3"):
        path = history / f"jev_9b-{method}" / "events.jsonl"
        source_hash = digest(path)
        active = None
        for line, text in enumerate(path.open(), 1):
            event = json.loads(text)
            if event["event_type"] in ("generation_request", "judge_request"):
                active = {"method": method, "source": str(path), "source_sha256": source_hash,
                          "line": line, "event": event, "following_usage": []}
                result.append(active)
            elif active and event["event_type"] == "usage":
                entry = event["payload"]
                if entry["source"].startswith("autotrust."):
                    active["following_usage"].append(entry)
    return result


def make_config(historical: dict, window: int, memory: int, method: str, url: str) -> dict:
    result = {key: deepcopy(historical[key]) for key in ("episode", "run")}
    result["run"]["method"] = method
    result["run"]["options"].update(memory_records=memory, max_new_tokens=256)
    result["model"] = {"backend": "jev_9b", "base_url": url, "max_model_len": window}
    validate_config(result)
    return result


def validate_config(config: dict) -> None:
    run = RunConfig.model_validate(config["run"])
    if ((config["model"]["max_model_len"], run.options["memory_records"]) not in ARMS
            or config["model"]["backend"] != "jev_9b" or run.method not in ("B0", "B3")
            or run.options.get("max_new_tokens") != 256 or run.learning_mode != "none"
            or run.budget.env_steps != 2000 or run.budget.model_requests != 3000
            or run.budget.input_tokens is not None or run.budget.output_tokens is not None
            or (run.candidate_count, run.simulations, run.max_depth, run.rollout_depth, run.max_steps)
            != (4, 8, 4, 2, 20) or run.seed != 0
            or config["episode"]["seed"] != 0
            or config["episode"]["task"] != "MiniGrid-Empty-5x5-v0"):
        raise ValueError("Configuration differs from the declared P14a smoke conditions")


def prepare(args) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    sources = historical_sources(args.history)
    write_json(args.output / "historical-requests.json", sources)
    historical = {method: json.loads((args.history / f"jev_9b-{method}" / "config.json").read_text())
                  for method in ("B0", "B3")}
    for window, memory in ARMS:
        for method in ("B0", "B3"):
            config = make_config(historical[method], window, memory, method, args.base_url)
            name = f"9b-{window}-memory{memory}-{method}.yaml"
            (args.output / name).write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(args.output / "preparation.json", {
        **provenance(args.assets_root), "history_root": str(args.history),
        "original_failures_preserved": True, "gpu_inference_requests": 0,
        "historical_logical_requests": len(sources), "runtime_capacity": "not_tested",
        "conditions": [{"window": w, "memory": m, "generation_output": 256, "judge_output": 1}
                       for w, m in ARMS],
        "historical_config_sha256": {m: digest(args.history / f"jev_9b-{m}" / "config.json")
                                     for m in historical},
    })


def variants(source: dict):
    """Historical requests are exact; modified-memory samples are explicitly synthetic.

    Retaining the last records is NOT an emulation of JEV relevance selection.
    The +2 evidence sample includes transient hypothesis/observation records.
    """
    original = source["event"]
    yield "historical_exact", original
    event = deepcopy(original)
    request = event["payload"]["request"]
    key = "prompt" if event["event_type"] == "generation_request" else "context"
    state, start, end = extract_public(request[key])
    if state is not None:
        maximum = 6 if request.get("role") == "evidence" else 4
        state["memory"] = state.get("memory", [])[-maximum:]
        request[key] = request[key][:start] + json.dumps(state, ensure_ascii=False) + request[key][end:]
        if request.get("role") == "evidence":
            request["items"] = request["items"][-4:]
            request["options"] = {}  # Policy-generated requests have no explicit item IDs.
        yield "counterfactual_suffix_memory4", event
    if original["event_type"] == "generation_request" and original["payload"].get("attempt", 0) == 0:
        for name, item in (("historical_repair_probe", original), ("memory4_repair_probe", event)):
            item = deepcopy(item)
            item["payload"]["request"]["prompt"] += REPAIR_SUFFIX
            item["payload"]["request"]["seed"] += 1
            item["payload"]["attempt"] = 1
            yield name, item


def offline(args) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    lengths, backend = Lengths(args.assets_root), CaptureBackend(args.assets_root)
    sources = json.loads(args.sources.read_text())
    summary = defaultdict(lambda: {"count": 0, "exceeds": 0, "peak_total": 0,
                                   "peak_input": 0, "min_remaining": None})
    records = args.output / "lengths.jsonl"
    for source in sources:
        for variant, event in variants(source):
            role = event["payload"]["request"].get("role", "generation")
            for item_index, captured in enumerate(backend.expand(event)):
                for window, memory in ARMS:
                    measured = lengths.measure(captured["endpoint"], captured["payload"], window)
                    historical_responses = [entry for entry in source["following_usage"]
                                            if entry["source"] == "autotrust.response"]
                    actual = (historical_responses[item_index] if variant == "historical_exact"
                              and item_index < len(historical_responses) else None)
                    row = {"source_event_id": event["event_id"], "source_method": source["method"],
                           "source_line": source["line"], "variant": variant, "role": role,
                           "item_index": item_index, "endpoint": captured["endpoint"], **measured,
                           "source_request_actual_usage": actual,
                           "synthetic_request_actual_usage": None}
                    append_json(records, row)
                    key = f"{variant}/{role}/{window}"
                    item = summary[key]
                    item["count"] += 1
                    item["exceeds"] += not measured["fits"]
                    item["peak_total"] = max(item["peak_total"], measured["total_window_tokens"])
                    item["peak_input"] = max(item["peak_input"], measured["offline_input_tokens"])
                    remaining = measured["remaining_tokens"]
                    item["min_remaining"] = remaining if item["min_remaining"] is None else min(item["min_remaining"], remaining)
    write_json(args.output / "summary.json", {
        **provenance(args.assets_root), "source_sha256": digest(args.sources), "groups": dict(summary),
        "http_inference_requests": 0, "gpu_inference_requests": 0,
        "runtime_capacity": "not_tested", "actual_usage_is_not_offline_length": True,
        "limitations": ["Counterfactual suffix-memory samples are length probes, not JEV selection or new trajectories.",
                        "Actual loops may retain different/longer records and actions; runtime guard checks every complete request.",
                        "Historical unknown usage stays unknown; offline counts do not repair historical accounting.",
                        "8192 admission requires a loaded service with matching max_model_len and protocol success."],
    })


class RecordingClient:
    def __init__(self, client):
        self.client = client
        self.response = None

    def post(self, *args, **kwargs):
        self.response = None
        response = self.client.post(*args, **kwargs)
        self.response = response
        return response

    def close(self):
        self.client.close()


class GuardedBackend(AutoTrustBackend):
    def __init__(self, *args, journal: Path, lengths: Lengths, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal, self.lengths = journal, lengths
        self._client = RecordingClient(self._client)
        self.logical = None
        self.number = 0

    def generate(self, request, **kwargs):
        self.logical = {"role": "generation", "repair": request.prompt.endswith(REPAIR_SUFFIX)}
        return super().generate(request, **kwargs)

    def score(self, request, **kwargs):
        self.logical = {"role": request.role.value, "repair": False}
        return super().score(request, **kwargs)

    def _request(self, path, payload, parse, call, ledger, phase, **kwargs):
        self.number += 1
        row = {"request_number": self.number, "endpoint": path, "phase": str(phase),
               **self.logical, "request": payload, "network_sent": False,
               "actual_usage": None, "actual_usage_complete": False}
        row.update(self.lengths.measure(path, payload, self.max_model_len))
        append_json(self.journal, {"event": "request_preflight", **row})
        if not row["fits"]:
            row.update(error="local_context_rejection: complete prompt plus output reserve exceeds window")
            append_json(self.journal, {"event": "request_finished", **row})
            raise ModelFailure(row["error"])
        self._client.response = None
        started = time.monotonic()
        entries_before = len(ledger.entries)
        try:
            result = super()._request(path, payload, parse, call, ledger, phase, **kwargs)
            return result
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            row["budget_block"] = isinstance(error, BudgetExceeded)
            raise
        finally:
            entries = list(ledger.entries)[entries_before:]
            row["network_sent"] = any(e.source == "autotrust.request" for e in entries)
            row["elapsed_seconds"] = time.monotonic() - started
            row["ledger_entries"] = [e.model_dump(mode="json") for e in entries]
            response = self._client.response
            if response is not None:
                row.update(status_code=response.status_code, raw_response=response.text)
                try:
                    body = response.json()
                    raw = body.get("usage") if isinstance(body, dict) else None
                    row["actual_usage"] = raw
                    row["actual_usage_complete"] = isinstance(raw, dict) and all(
                        type(raw.get(k)) is int and raw[k] >= 0 for k in ("prompt_tokens", "completion_tokens"))
                    if row["actual_usage_complete"]:
                        # vLLM reports the shared prompt once, even when n > 1;
                        # completion_tokens aggregates the returned sequences.
                        row["offline_expected_reported_prompt_tokens"] = row["offline_input_tokens"]
                        row["offline_service_prompt_match"] = raw["prompt_tokens"] == row["offline_input_tokens"]
                except (ValueError, TypeError):
                    pass
            append_json(self.journal, {"event": "request_finished", **row})


def service_admission(url: str, window: int) -> dict:
    base = url.removesuffix("/").removesuffix("/v1")
    response = httpx.get(base + "/v1/models", timeout=30)
    response.raise_for_status()
    body = response.json()
    cards = [item for item in body["data"] if item["id"] == "autotrust/JEV-9B"]
    if len(cards) != 1 or cards[0].get("max_model_len") != window:
        raise ValueError(f"Service/client window mismatch or unknown window: {body}")
    return {"client_max_model_len": window, "server_models": body,
            "window_matches": True, "capacity_inference_tested": False}


def execute(args) -> None:
    from jevgo import factory
    config = yaml.safe_load(args.config.read_text())
    validate_config(config)
    args.audit_output.mkdir(parents=True, exist_ok=False)
    write_json(args.audit_output / "provenance.json", {**provenance(args.assets_root),
               "config_sha256": digest(args.config), "mode": args.command})
    write_json(args.audit_output / "config.json", config)
    lengths = Lengths(args.assets_root)
    try:
        admission = service_admission(config["model"]["base_url"], config["model"]["max_model_len"])
        write_json(args.audit_output / "service-admission.json", admission)
        def build(model_config, *, assets_root, episode=None):
            options = dict(model_config)
            model = options.pop("backend")
            return GuardedBackend(model, assets_root=assets_root, lengths=lengths,
                                  journal=args.audit_output / "http.jsonl", **options)
        with patch.object(factory, "build_backend", build):
            if args.command == "protocol":
                result = factory.check_model(config["model"], assets_root=args.assets_root, output_dir=args.output)
                ok = result["ok"] and result["usage_complete"]
                stop = {"termination_reason": "protocol_complete" if ok else "protocol_failed",
                        "raw_error": result["error"]}
            else:
                result = factory.run_from_dict(config, assets_root=args.assets_root, output_dir=args.output)
                ok = result.error is None or result.termination_reason.value == "budget_exhausted"
                stop = {"termination_reason": result.termination_reason.value, "raw_error": result.error}
        write_json(args.audit_output / "execution.json", {"ok": ok, "output": str(args.output), **stop})
    except Exception as error:
        write_json(args.audit_output / "execution.json", {"ok": False, "error": f"{type(error).__name__}: {error}"})
        raise
    if not ok:
        raise SystemExit(1)


def journal_accounting(journal: list[dict], ledger: dict) -> dict:
    """Independent pairing and raw-response-to-ledger accounting, including failures."""
    preflights = [r for r in journal if r["event"] == "request_preflight"]
    finished = [r for r in journal if r["event"] == "request_finished"]
    identifiers = [r["request_number"] for r in preflights]
    pairs = (identifiers == list(range(1, len(preflights) + 1))
             and identifiers == [r["request_number"] for r in finished])
    fields = ("request_number", "request", "endpoint", "phase", "role", "repair", "fits",
              "offline_input_tokens", "requested_output_tokens_per_sequence", "max_model_len")
    pairs = pairs and all(all(a.get(k) == b.get(k) for k in fields) for a, b in zip(preflights, finished))
    requests = [r for r in finished if r["network_sent"]]
    counts_match = len(requests) == ledger["used"]["model_requests"]
    totals = {"input_tokens": 0, "output_tokens": 0}
    unknown = []
    raw_matches_record = True
    for row in requests:
        try:
            body = json.loads(row["raw_response"])
            raw = body.get("usage") if isinstance(body, dict) else None
        except (ValueError, KeyError):
            raw = None
        raw_matches_record &= raw == row["actual_usage"]
        for remote, local in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
            value = raw.get(remote) if isinstance(raw, dict) else None
            if type(value) is int and value >= 0:
                totals[local] += value
            else:
                unknown.append({"request_number": row["request_number"], "field": local})
    # Known subtotals may reconcile even with unknown costs; completeness stays separate.
    tokens_match = all(totals[k] == ledger["used"][k] for k in totals)
    journal_entries = [entry for row in finished for entry in row.get("ledger_entries", [])]
    recorded_entries = [entry for entry in ledger["entries"] if entry["source"] in
                        ("autotrust.request", "autotrust.response")]
    entries_match = journal_entries == recorded_entries
    return {"preflight_finished_pairs_match": pairs, "http_ledger_request_counts_match": counts_match,
            "raw_usage_matches_journal": raw_matches_record, "known_raw_token_totals_match_ledger": tokens_match,
            "request_response_entries_match_ledger": entries_match, "known_raw_token_totals": totals,
            "unknown_raw_usage": unknown,
            "ok": pairs and counts_match and raw_matches_record and tokens_match and entries_match}


def frontier_met(rows: list[dict], method: str) -> bool:
    required_role = "generation" if method == "B0" else "evidence"
    return any(row["role"] == required_role and not row.get("error") and row.get("status_code") == 200
               and row.get("network_sent") and (row.get("branch_actions") or 0) >= 9
               and (row.get("memory_entries") or 0) >= 19 for row in rows)


def budget_acceptance(production_audit: dict) -> bool:
    return all(production_audit.get(key) is True for key in
               ("within_budget", "declared_over_budget_consistent", "cost_complete"))


def audit(args) -> None:
    from jevgo.evaluation import audit_run
    result = json.loads((args.run / "result.json").read_text())
    ledger = json.loads((args.run / "ledger.json").read_text())
    journal = [json.loads(line) for line in (args.audit / "http.jsonl").open()]
    rows = [row for row in journal if row["event"] == "request_finished"]
    events = [json.loads(line) for line in (args.run / "events.jsonl").open()]
    sums = {key: sum(e["payload"]["usage"][key] for e in events if e["event_type"] == "usage")
            for key in ledger["used"]}
    try:
        production_audit = audit_run(args.run)
        production_audit.pop("result")  # Full validated result already lives in the run directory.
        reconciles = True
    except (ValueError, TypeError, KeyError) as error:
        production_audit = {"error": f"{type(error).__name__}: {error}"}
        reconciles = False
    journal_audit = journal_accounting(journal, ledger)
    sent = [row for row in rows if row["network_sent"]]
    crossed = frontier_met(rows, result["config"]["method"])
    objective_success = result["final_state"]["success"] is True
    memory = result["config"]["options"]["memory_records"]
    unknown = [row["request_number"] for row in sent if not row["actual_usage_complete"]]
    errors = [row for row in rows if row.get("error") and not row.get("budget_block")]
    complete = not unknown and not journal_audit["unknown_raw_usage"] and all(
        e.get("metadata", {}).get("usage_complete", True) for e in ledger["entries"])
    prompt_matches = bool(sent) and all(row.get("offline_service_prompt_match") is True for row in sent)
    commits = sum(e["event_type"] == "transition" and e["payload"]["phase"] == "commit" for e in events)
    finite_stop = result["termination_reason"] in ("success", "environment_truncated", "max_steps", "budget_exhausted", "environment_terminated")
    technical_pass = (bool(commits) and finite_stop and not errors and complete and reconciles
                      and prompt_matches and journal_audit["ok"] and budget_acceptance(production_audit))
    write_json(args.output, {
        "run": str(args.run), "termination_reason": result["termination_reason"], "raw_error": result["error"],
        "objective_success": objective_success, "committed_steps": commits,
        "usage_from_events": sums, "event_ledger_result_reconcile": reconciles,
        "production_audit": production_audit, "journal_accounting": journal_audit,
        "usage_complete": complete, "unknown_usage_request_numbers": unknown,
        "http_requests_sent": len(sent), "journal_request_attempts": len(rows),
        "offline_service_prompt_counts_match": prompt_matches,
        "local_rejections": [r["request_number"] for r in rows if not r["network_sent"]],
        "error_request_numbers": [r["request_number"] for r in errors],
        "peak_offline_input_tokens": max((r["offline_input_tokens"] for r in rows), default=None),
        "peak_total_window_tokens": max((r["total_window_tokens"] for r in rows), default=None),
        "peak_actual_aggregate_input_tokens": max((r["actual_usage"]["prompt_tokens"] for r in sent if r["actual_usage_complete"]), default=None),
        "successful_frontier_9_actions_19_memories": crossed,
        "memory32_frontier_requirement_met": crossed or objective_success,
        "technical_closed_loop_pass": technical_pass,
        "context_condition_resolved": technical_pass and (memory != 32 or crossed or objective_success),
        "historical_failure_overwritten": False, "research_effectiveness": "pending",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=Path(os.environ.get("JEVGO_ASSETS_ROOT", ROOT)))
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--history", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--base-url", default="http://127.0.0.1:8019")
    off = commands.add_parser("offline")
    off.add_argument("--sources", type=Path, required=True)
    off.add_argument("--output", type=Path, required=True)
    for name in ("protocol", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--audit-output", type=Path, required=True)
    check = commands.add_parser("audit")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--audit", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "offline": offline, "protocol": execute, "run": execute, "audit": audit}[args.command](args)


if __name__ == "__main__":
    main()
