"""P14h finite, development-only capacity evidence; never starts a GPU service.

prepare is CPU-only. run sends only the selected, predeclared condition. It does
not retry, truncate prompts, infer missing usage, or manage server processes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from p14_context import CaptureBackend, Lengths, digest, write_json
from jevgo.models.autotrust import load_model_config
from jevgo.models.serve_autotrust import build_launch

SCHEMA = "p14h-capacity-v1"
SHAPES = {"single": (1, 1), "http4": (4, 1), "n4": (1, 4)}
METRIC = "vllm:num_requests_running"
RUNTIME_FIELDS = ("hardware", "driver", "kernel", "runtime", "dtype", "tensor_parallel_size",
                  "max_num_seqs", "kv_cache_dtype", "prefix_caching", "max_num_batched_tokens",
                  "chunked_prefill", "service_command", "exclusive_service")


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def capture_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def complete_ids(lengths, endpoint, payload):
    if endpoint == "/v1/chat/completions":
        text = lengths.tokenizer.apply_chat_template(payload["messages"], tokenize=False,
            add_generation_prompt=True, **payload.get("chat_template_kwargs", {}))
    else:
        text = payload["prompt"]
    return lengths.tokenizer.encode(text, add_special_tokens=False, truncation=False)


def sized_payload(capture, lengths, window, decode, n, nonce, input_target=None):
    """Find and check an exact full-template length. No truncation or guessing."""
    event = {"event_type": "generation_request", "payload": {"request": {
        "prompt": f"Capacity probe {nonce}. Continue the sequence with x tokens.\n",
        "system": "This is a synthetic token capacity test.", "n": n,
        "max_new_tokens": decode, "temperature": 0.7, "seed": 1414}}}
    item = capture.expand(event)[0]
    payload = item["payload"]
    base = payload["messages"][-1]["content"]
    target = input_target if input_target is not None else window - decode
    count = lengths.measure(item["endpoint"], payload, window)["offline_input_tokens"]
    repetitions = target - count
    if repetitions < 0:
        raise ValueError("Window is shorter than complete template and output reserve")
    # ' x' is stable in the local Qwen tokenizer. Every candidate is re-tokenized;
    # incompatible tokenizers fail explicitly instead of changing the target.
    for _ in range(12):
        payload["messages"][-1]["content"] = base + " x" * repetitions
        measured = lengths.measure(item["endpoint"], payload, window)
        error = target - measured["offline_input_tokens"]
        if error == 0:
            payload.update(stream=True, stream_options={"include_usage": True},
                           return_token_ids=True, ignore_eos=True, min_tokens=decode)
            return {**item, "measurement": measured}
        repetitions += error
        if repetitions < 0:
            break
    raise ValueError("Cannot construct exact template length with bounded padding search")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare(args):
    if args.windows not in ([262144], [262144, 32768]):
        raise ValueError("Native-first plan permits 262144 and at most fallback 32768")
    if not 1 <= args.decode <= 1024:
        raise ValueError("decode must be between 1 and 1024")
    if len(set(args.shapes)) != len(args.shapes):
        raise ValueError("Duplicate condition shapes are forbidden")
    if not 1 <= args.tp <= 4 or args.max_num_seqs < 4 or args.batched_tokens < 32:
        raise ValueError("Invalid finite runtime launch limits")
    if not 0 < args.gpu_memory_utilization < 1 or not 1 <= args.diagnostic_input <= 32736:
        raise ValueError("Invalid memory utilization or diagnostic input length")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg, model_dir = load_model_config("jev_9b", args.assets_root)
    native = json.loads((model_dir / "config.json").read_text())
    if native.get("max_position_embeddings") != 262144 or native.get("dtype") != "bfloat16":
        raise ValueError("Expected original BF16 9B checkpoint declaring 262144")
    lengths = Lengths(args.assets_root)
    capture = CaptureBackend(args.assets_root)
    run_id = uuid.uuid4().hex
    conditions = []
    for window in args.windows:
        launch, environment = build_launch("jev_9b", assets_root=args.assets_root,
            max_model_len=window, tensor_parallel_size=args.tp, port=args.port)
        launch += ["--dtype", "bfloat16", "--kv-cache-dtype", "auto", "--max-num-seqs", str(args.max_num_seqs),
                   "--max-num-batched-tokens", str(args.batched_tokens),
                   "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                   "--no-enable-prefix-caching", "--enable-chunked-prefill"]
        write_json(args.output / f"launch-{window}.json", {"argv": launch,
            "environment": environment, "evidence_status": "unexecuted_launch_plan",
            "window": window, "tensor_parallel_size": args.tp, "dtype": "bfloat16",
            "kv_cache_dtype": "auto", "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.batched_tokens, "prefix_caching": False,
            "chunked_prefill": True, "gpu_memory_utilization": args.gpu_memory_utilization})
        for shape in args.shapes:
            http_count, n = SHAPES[shape]
            cid = f"w{window}-{shape}"
            requests = []
            for i in range(http_count):
                item = sized_payload(capture, lengths, window, args.decode, n,
                                     f"{run_id}-{cid}-{i}",
                                     input_target=None if shape == "single" else args.diagnostic_input)
                path = args.output / f"{cid}-{i}.json"
                write_json(path, item)
                requests.append({"file": path.name, "sha256": digest(path),
                    "input_tokens": item["measurement"]["offline_input_tokens"],
                    "n": n, "output_target_per_sequence": args.decode})
            conditions.append({"id": cid, "window": window, "shape": shape,
                "http_requests": http_count, "sequences": http_count * n,
                "requests": requests, "max_attempts": 1,
                "admission": "primary" if window == 262144 else "native_resource_failure_required",
                "cache_policy": "unique_user_prefix; launch disables prefix caching; verify runtime",
                "purpose": "synthetic_capacity_only_not_long_context_quality"})
    files = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "model.safetensors.index.json")
    manifest = {"schema": SCHEMA, "run_id": run_id, "learning_mode": "none",
        "model": cfg, "native_declaration": native, "assets_root": str(args.assets_root),
        "model_files_sha256": {name: digest(model_dir / name) for name in files
                               if (model_dir / name).is_file()},
        "tool_sha256": digest(Path(__file__)), "base_tool_sha256": digest(Path(__file__).with_name("p14_context.py")),
        "python": sys.executable, "cpu_platform": platform.platform(),
        "tokenizer_runtime": {name: package_version(name) for name in ("transformers", "tokenizers")},
        "conditions": conditions, "max_model_requests": sum(c["http_requests"] for c in conditions),
        "input_tokens_planned": sum(r["input_tokens"] for c in conditions for r in c["requests"]),
        "output_tokens_upper_bound": sum(r["n"] * args.decode for c in conditions for r in c["requests"]),
        "execution_policy": "one attempt per condition; no automatic restart/retry/fallback; root owns GPU",
        "actual_model_requests": 0}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("max_model_requests", "input_tokens_planned",
                                                   "output_tokens_upper_bound", "actual_model_requests")}))
    return 0


def load_condition(path, condition_id):
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown manifest schema")
    condition = next(c for c in manifest["conditions"] if c["id"] == condition_id)
    requests = []
    for spec in condition["requests"]:
        source = path.parent / spec["file"]
        if digest(source) != spec["sha256"]:
            raise ValueError(f"Request hash changed: {source}")
        item = json.loads(source.read_text())
        if item["measurement"]["total_window_tokens"] > condition["window"]:
            raise ValueError("Request does not fit its predeclared window")
        requests.append(item)
    return manifest, condition, requests


def usage_value(usage, key):
    value = usage.get(key) if isinstance(usage, dict) else None
    return value if type(value) is int and value >= 0 else None


def response_summary(chunks, request, error=None):
    payload = request["payload"]
    n = payload.get("n", 1)
    counts = {i: None for i in range(n)}
    finish = {}
    prompt_ids = None
    usage = None
    for chunk in chunks:
        if chunk.get("usage") is not None:
            usage = chunk["usage"]
        if chunk.get("prompt_token_ids") is not None:
            ids = chunk["prompt_token_ids"]
            if isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids):
                prompt_ids = ids
            else:
                error = error or "invalid_prompt_token_ids"
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            error = error or "invalid_choices"
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                error = error or "invalid_choice"
                continue
            i = choice.get("index")
            if type(i) is not int or i not in counts:
                error = error or "unexpected_choice_index"
                continue
            ids = choice.get("token_ids")
            if isinstance(ids, list) and all(type(x) is int and x >= 0 for x in ids):
                counts[i] = (counts[i] or 0) + len(ids)
            elif ids is not None:
                error = error or "invalid_completion_token_ids"
            if choice.get("finish_reason") is not None:
                if isinstance(choice["finish_reason"], str):
                    finish[i] = choice["finish_reason"]
                else:
                    error = error or "invalid_finish_reason"
    input_tokens = usage_value(usage, "prompt_tokens")
    output_tokens = usage_value(usage, "completion_tokens")
    target = payload["max_tokens"]
    # n=1 total usage is also an actual count; never divide n>1 totals to
    # fabricate per-choice output lengths. Missing token IDs remain unknown.
    if n == 1 and counts[0] is None and output_tokens is not None:
        counts[0] = output_tokens
    target_met = all(value == target for value in counts.values())
    prompt_match = (input_tokens == request["measurement"]["offline_input_tokens"]
                    if input_tokens is not None else None)
    ids_match = (ids_digest(prompt_ids) == request["measurement"]["token_ids_sha256"]
                 if prompt_ids is not None else None)
    count_match = (sum(counts.values()) == output_tokens
                   if output_tokens is not None and all(v is not None for v in counts.values()) else None)
    return {"usage_raw": usage, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "usage_complete": input_tokens is not None and output_tokens is not None,
            "unknown_usage": [key for key, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)) if value is None],
            "actual_decode_per_sequence": counts, "decode_target": target,
            "decode_target_met": target_met, "finish_reasons": finish,
            "offline_service_prompt_match": prompt_match, "service_prompt_ids_match": ids_match,
            "sequence_counts_match_usage": count_match, "error": error,
            "prefill_decode_pass": (error is None and prompt_match is True and ids_match is True
                                      and target_met and count_match is True and len(finish) == n)}


async def request_once(client, base_url, item, path, timeout):
    start = time.monotonic()
    capture_json(path, {"event": "start", "monotonic": start, "request": item})
    chunks, raw_lines, first, error, status = [], [], None, None, None
    try:
        async with asyncio.timeout(timeout), client.stream("POST", base_url + item["endpoint"], json=item["payload"]) as response:
            status = response.status_code
            capture_json(path, {"event": "headers", "status": status,
                "headers": dict(response.headers), "monotonic": time.monotonic()})
            async for line in response.aiter_lines():
                stamp = time.monotonic()
                raw_lines.append(line)
                capture_json(path, {"event": "raw_line", "monotonic": stamp, "line": line})
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line[5:].strip())
                    if not isinstance(chunk, dict):
                        raise ValueError("SSE data is not an object")
                    chunks.append(chunk)
                    if chunk.get("error"):
                        error = f"service_error: {chunk['error']}"
                    choices = chunk.get("choices", [])
                    if not isinstance(choices, list) or any(not isinstance(c, dict) for c in choices):
                        raise ValueError("SSE choices must be a list of objects")
                    if first is None and any(c.get("token_ids") or
                            (isinstance(c.get("delta"), dict) and c["delta"].get("content"))
                            for c in choices):
                        first = stamp
                except (ValueError, TypeError) as exc:
                    error = f"invalid_sse: {exc}"
            # Error responses commonly use ordinary JSON rather than SSE. The
            # body can contain real partial usage; preserve it instead of
            # replacing known billed work with unknowns.
            if not chunks:
                try:
                    body = json.loads("\n".join(raw_lines))
                    if isinstance(body, dict):
                        chunks.append(body)
                        if body.get("error"):
                            error = f"service_error: {body['error']}"
                except (ValueError, TypeError):
                    pass  # Raw bytes/lines remain in the journal.
            if status != 200:
                error = f"HTTP {status}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.monotonic()
    summary = {**response_summary(chunks, item, error), "network_attempted": True,
               "http_status": status, "start_monotonic": start, "end_monotonic": end,
               "http_wall_seconds": end - start, "client_ttft_seconds": first - start if first else None,
               "client_after_first_token_seconds": end - first if first else None,
               "timing_scope": "client wall incl queue/network; TTFT is not pure prefill",
               "raw_path": str(path)}
    capture_json(path, {"event": "finished", "summary": summary})
    return summary


def metrics_values(raw, name):
    pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)(?:\s.*)?$")
    values = []
    for line in raw.splitlines():
        match = pattern.match(line)
        if match:
            values.append(float(match.group(1)))
    return values


def sample_metrics(client, base_url, output):
    stamp = time.monotonic()
    try:
        response = client.get(base_url + "/metrics")
        row = {"monotonic": stamp, "status": response.status_code, "raw": response.text}
    except Exception as exc:
        row = {"monotonic": stamp, "status": None, "error": f"{type(exc).__name__}: {exc}", "raw": ""}
    capture_json(output, row)
    return row


def overlap_max(rows):
    events = [(r["start_monotonic"], 1) for r in rows] + [(r["end_monotonic"], -1) for r in rows]
    current = peak = 0
    for _, change in sorted(events):
        current += change
        peak = max(peak, current)
    return peak


def concurrency_summary(rows, samples, condition, metadata):
    if not rows:
        return {"status": "not_tested", "http_overlap_peak": 0}
    start, end = min(r["start_monotonic"] for r in rows), max(r["end_monotonic"] for r in rows)
    # Do not add DP engines (could double-count replicated metrics); a single
    # engine must independently reach the target. TP ranks aren't summed.
    counts = [max(metrics_values(s["raw"], METRIC), default=0) for s in samples
              if start <= s["monotonic"] <= end and s["status"] == 200]
    peak = max(counts, default=None)
    semantics = metadata.get("active_sequence_evidence", {})
    qualified = (metadata.get("exclusive_service") is True and
                 semantics.get("metric") == METRIC and
                 semantics.get("unit") == "scheduler_sequences" and
                 bool(semantics.get("source_sha256")) and
                 metadata.get("runtime_source_hashes_verified") is True)
    overlap = overlap_max(rows)
    coincident = [max(metrics_values(s["raw"], METRIC), default=0) for s in samples
                  if s["status"] == 200 and sum(
                      r["start_monotonic"] <= s["monotonic"] < r["end_monotonic"]
                      for r in rows) >= condition["http_requests"]]
    coincident_peak = max(coincident, default=None)
    met = (qualified and coincident_peak is not None
           and coincident_peak >= condition["sequences"])
    status = "observed_active_sequences" if met else "active_sequence_target_unproven"
    return {"status": status, "http_overlap_peak": overlap, "target_http": condition["http_requests"],
            "target_sequences": condition["sequences"], "scheduler_running_peak": peak,
            "scheduler_running_peak_during_target_http_overlap": coincident_peak,
            "metric_semantics_qualified": qualified, "active_sequences_target_met": bool(met),
            "limitation": "sampled scheduler state; queue overlap/completion alone is not active concurrency"}


def save_evidence(output, sources):
    hashes = {}
    for i, source in enumerate(sources):
        destination = output / f"evidence-{i}-{source.name}"
        shutil.copyfile(source, destination)
        hashes[destination.name] = digest(destination)
    return hashes


def claim_attempt(manifest_path, condition_id, output):
    directory = manifest_path.parent / "attempts"
    directory.mkdir(exist_ok=True)
    # Exclusive create prevents accidentally paying for the same condition twice.
    # Copying the manifest elsewhere is a new orchestrated attempt and must be
    # reported explicitly by the parent, not hidden as a retry.
    with (directory / f"{condition_id}.json").open("x") as stream:
        json.dump({"manifest_sha256": digest(manifest_path), "output": str(output),
                   "condition": condition_id, "claimed_unix": time.time()}, stream)


def run(args):
    start = time.monotonic()
    manifest, condition, items = load_condition(args.manifest, args.condition)
    metadata = json.loads(args.runtime_metadata.read_text())
    source_hashes = metadata.get("active_sequence_evidence", {}).get("source_sha256", {})
    metadata["runtime_source_hashes_verified"] = bool(source_hashes) and all(
        Path(path).is_file() and digest(Path(path)) == expected for path, expected in source_hashes.items())
    if condition["admission"] != "primary" and not args.native_failure:
        raise ValueError("Fallback requires --native-failure raw evidence; no automatic fallback")
    if args.timeout <= 0 or args.timeout > 1200 or not 0.05 <= args.metrics_interval <= 5:
        raise ValueError("Require bounded timeout <=1200 and metrics interval in [0.05,5]")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "runtime-metadata.json", metadata)
    evidence = save_evidence(args.output, [args.manifest, args.runtime_metadata, *args.evidence,
                                          *([args.native_failure] if args.native_failure else [])])
    lengths = Lengths(args.assets_root)
    for item in items:
        actual = lengths.measure(item["endpoint"], item["payload"], condition["window"])
        if actual != item["measurement"]:
            raise ValueError("Runtime tokenizer/template differs from prepared complete payload")
    base_url = args.base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    claim_attempt(args.manifest, args.condition, args.output)
    rows, samples = [], []
    admission_error = None
    service = {"status": "unobserved"}
    with httpx.Client(timeout=args.timeout, trust_env=False) as client:
        try:
            response = client.get(base_url + "/v1/models")
            write_json(args.output / "models-response.json", {"status": response.status_code, "raw": response.text})
            response.raise_for_status()
            matching = [m for m in response.json().get("data", []) if m.get("id") == manifest["model"]["repository"]]
            if len(matching) != 1 or matching[0].get("max_model_len") != condition["window"]:
                raise ValueError("service window/model mismatch or unknown")
            service = {"status": "passed", "observed": matching[0]}
        except Exception as exc:
            admission_error = f"{type(exc).__name__}: {exc}"
            service = {"status": "failed", "error": admission_error}
        if not admission_error:
            # Dedicated short-timeout metrics client avoids delaying model requests.
            with httpx.Client(timeout=2, trust_env=False) as metrics_client:
                samples.append(sample_metrics(metrics_client, base_url, args.output / "metrics.jsonl"))
                if max(metrics_values(samples[0]["raw"], METRIC), default=0) > 0:
                    metadata["exclusive_service"] = False
                stop = threading.Event()
                def poll():
                    while not stop.wait(args.metrics_interval):
                        samples.append(sample_metrics(metrics_client, base_url, args.output / "metrics.jsonl"))
                worker = threading.Thread(target=poll, daemon=True)
                worker.start()
                try:
                    async def execute_group():
                        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as async_client:
                            return await asyncio.gather(*[
                                request_once(async_client, base_url, item,
                                    args.output / f"request-{i}.jsonl", args.timeout)
                                for i, item in enumerate(items)])
                    rows = asyncio.run(execute_group())
                finally:
                    stop.set()
                    worker.join()
                samples.append(sample_metrics(metrics_client, base_url, args.output / "metrics.jsonl"))
    concurrency = concurrency_summary(rows, samples, condition, metadata)
    usage_complete = bool(rows) and all(r["usage_complete"] for r in rows)
    input_known = sum(r["input_tokens"] or 0 for r in rows)
    output_known = sum(r["output_tokens"] or 0 for r in rows)
    prefill_pass = bool(rows) and all(r["prefill_decode_pass"] for r in rows)
    wall = time.monotonic() - start
    metrics_costs = {}
    if len(samples) >= 2 and metadata.get("exclusive_service") is True:
        for name in ("request_prefill_time_seconds", "request_decode_time_seconds", "request_queue_time_seconds"):
            before = metrics_values(samples[0]["raw"], "vllm:" + name + "_sum")
            after = metrics_values(samples[-1]["raw"], "vllm:" + name + "_sum")
            before_count = metrics_values(samples[0]["raw"], "vllm:" + name + "_count")
            after_count = metrics_values(samples[-1]["raw"], "vllm:" + name + "_count")
            count_delta = sum(after_count) - sum(before_count) if before_count and after_count else None
            metrics_costs[name] = {"count_delta": count_delta,
                "sum_delta": (sum(after) - sum(before) if before and after and
                              count_delta is not None and count_delta > 0 and sum(after) >= sum(before) else None)}
    report = {"schema": SCHEMA, "condition": condition, "manifest_sha256": digest(args.manifest),
        "four_layers": {"native_declaration": {"status": "declared_only", "window": manifest["native_declaration"]["max_position_embeddings"]},
            "service_window": service,
            "real_prefill_decode": {"status": "passed" if prefill_pass else "failed" if rows else "not_tested"},
            "concurrency": concurrency}, "requests": rows,
        "metadata_unknown": [key for key in RUNTIME_FIELDS if metadata.get(key) is None],
        "cost": {"model_requests": len(rows), "known_input_tokens_lower_bound": input_known,
            "known_output_tokens_lower_bound": output_known, "usage_complete": usage_complete,
            "input_tokens": input_known if usage_complete else None,
            "output_tokens": output_known if usage_complete else None,
            "client_probe_e2e_seconds": wall, "http_wall_seconds_sum": sum(r["http_wall_seconds"] for r in rows),
            "throughput_output_tokens_per_client_probe_second": output_known / wall if usage_complete else None,
            "server_instrumented_metrics": metrics_costs,
            "scope": "probe E2E excludes service/model startup; HTTP sum overlaps; metric sums are server instrumentation"},
        "admission_error": admission_error, "evidence_sha256": evidence,
        "capacity_pass": (prefill_pass and concurrency.get("active_sequences_target_met", False)
                          and all(metadata.get(key) is not None for key in RUNTIME_FIELDS)),
        "limits": ["Synthetic capacity does not establish long-context quality.",
                   "No retry; server/GPU lifecycle and startup failure evidence belong to parent orchestration.",
                   "Original unknown usage stays unknown; offline counts never replace service usage."]}
    write_json(args.output / "runtime-metadata.json", metadata)
    write_json(args.output / "report.json", report)
    write_json(args.output / "files-sha256.json", {p.name: digest(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({"condition": condition["id"], "capacity_pass": report["capacity_pass"], "cost": report["cost"]}))
    return 0 if report["capacity_pass"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=Path(os.environ.get("JEVGO_ASSETS_ROOT", ROOT)))
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="CPU only, immutable finite native-first manifest")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--windows", type=int, nargs="+", default=[262144, 32768])
    prep.add_argument("--shapes", choices=SHAPES, nargs="+", default=list(SHAPES))
    prep.add_argument("--decode", type=int, default=32)
    prep.add_argument("--tp", type=int, default=2)
    prep.add_argument("--port", type=int, default=8019)
    prep.add_argument("--batched-tokens", type=int, default=2048)
    prep.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    prep.add_argument("--max-num-seqs", type=int, default=8)
    prep.add_argument("--diagnostic-input", type=int, default=8192)
    execute = commands.add_parser("run", help="one finite condition against an already-owned service")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--condition", required=True)
    execute.add_argument("--base-url", required=True)
    execute.add_argument("--runtime-metadata", type=Path, required=True)
    execute.add_argument("--evidence", type=Path, action="append", default=[])
    execute.add_argument("--native-failure", type=Path)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--timeout", type=float, default=1200)
    execute.add_argument("--metrics-interval", type=float, default=0.1)
    args = parser.parse_args(argv)
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
