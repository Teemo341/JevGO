# P14h native window and concurrency evidence

Development-only EVAL tool, supporting AR-0/2/5/7. It uses the original BF16 9B
checkpoint and production `CaptureBackend` request builder plus `Lengths` full
chat-template tokenizer. It never starts a service, loads weights, touches GPUs,
changes published defaults, or trains. GPU allocation, native startup/failure,
server logs, hardware telemetry and process cleanup belong to the parent task.

`prepare` requires local assets only. Every request is predeclared, hashed and
measured without truncation. Default native-first conditions are:

| Service window | Condition | HTTP requests | n per HTTP | Exact input per HTTP | Actual decode target per sequence |
| --- | --- | --- | --- | --- | --- |
| 262144 | w262144-single | 1 | 1 | 262112 | 32 |
| 262144 | w262144-http4 | 4 | 1 | 8192 | 32 |
| 262144 | w262144-n4 | 1 | 4 | 8192 | 32 |
| 32768 | w32768-single | 1 | 1 | 32736 | 32 |
| 32768 | w32768-http4 | 4 | 1 | 8192 | 32 |
| 32768 | w32768-n4 | 1 | 4 | 8192 | 32 |

The 32768 conditions are available only if the parent records a native resource
failure, supplied with `--native-failure`. The tool does not automatically select
a fallback or retry. Each condition can be claimed once per manifest directory;
a second run fails before model HTTP. If infrastructure failure justifies a new
attempt, preserve the first evidence and explicitly account for the new manifest
copy/attempt. Never erase `attempts/` to conceal a retry.

Maximum across all prepared conditions: 12 HTTP requests, 376768 prompt tokens
(shared n=4 prompt charged once if the service reports that convention), 576
output tokens. Native stage alone: 6 HTTP, 303072 input, 288 output. These are
predeclared bounds, not incurred costs. Failure usage remains unknown unless the
service actually returns it. Four concurrent HTTP completions do not prove four
active sequences; single HTTP n=4 is a different condition.

## CPU preparation

Run from the task worktree, with a fresh output directory:

```bash
export JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO
export JEVGO_PY=/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python
"$JEVGO_PY" -B .dev/tools/p14_capacity.py prepare \
  --output .dev/artifacts/P14h/prepared \
  --tp 2 --max-num-seqs 8 --gpu-memory-utilization .85 \
  --decode 32 --diagnostic-input 8192
```

Use `--windows 262144` to omit the fallback altogether. Every generated
`launch-WINDOW.json` is an **unexecuted plan** from production `build_launch`,
with BF16, TP2, KV auto, max_num_seqs=8, max_num_batched_tokens=2048, prefix cache
disabled and chunked prefill enabled. The parent chooses GPUs and starts the
recorded argv after the required GPU checks. The native declaration in config
is independent of whether that runtime starts or executes the native request.
Unique per-request user prefixes and disabled prefix caching control the main
cache condition; actual cache flags and usage details must still be captured.
Synthetic padding and forced min_tokens/ignore_eos are capacity probes, not
production generation policy or evidence of long-context reasoning quality.

## One bounded condition on the already owned service

Prepare `runtime.json` from actual launch/runtime and hardware evidence. It must
contain `hardware`, `driver`, `kernel`, `runtime`, `dtype`,
`tensor_parallel_size`, `max_num_seqs`, `kv_cache_dtype`, `prefix_caching`,
`max_num_batched_tokens`, `chunked_prefill`, `service_command`, and
`exclusive_service`. Unknown values can be null and are reported; they prevent
full capacity acceptance. Record startup elapsed time and measured GPU sampling
scope/peak in this metadata or attached evidence. A reserved KV pool peak is not
minimum required VRAM. The tool preserves the complete metadata without forcing
its shape beyond the named keys.

For concurrency acceptance add `active_sequence_evidence` with
`metric: "vllm:num_requests_running"`, `unit: "scheduler_sequences"`, and
`source_sha256` identifying the reviewed runtime source. On local vLLM 0.31,
`v1/engine/async_llm.py` fans out n children, `parallel_sampling.py` makes each
child n=1, scheduler `get_request_counts` counts `len(self.running)`, and
`metrics/loggers.py` copies that count into the gauge. Save those exact source
hashes/snippets as evidence and revalidate after changing runtime/installations.
No claim is made from a unit string alone without this auditable source evidence.
An exclusive owned service is required; pre-existing running requests invalidate
that assumption. TP/DP gauge values are never summed to fabricate concurrency.

```bash
"$JEVGO_PY" -B .dev/tools/p14_capacity.py run \
  --manifest .dev/artifacts/P14h/prepared/manifest.json \
  --condition w262144-single --base-url http://127.0.0.1:8019 \
  --runtime-metadata .dev/artifacts/P14h/runtime.json \
  --evidence .dev/artifacts/P14h/metric-semantics.json \
  --evidence .dev/artifacts/P14h/service.stdout.txt \
  --output .dev/artifacts/P14h/native-single --timeout 1200
```

Then the parent may invoke `w262144-http4` and `w262144-n4` with fresh outputs;
these prove at most the 8192-input diagnostic workload, not native-length times
four. For a documented resource fallback, start exactly one 32768 service and
supply `--native-failure /absolute/path/to/raw-native-failure.json` for its chosen
conditions. All commands work on local or remote services; transferred manifests
retain request hashes and the runtime recount must match. Do not copy weights or
use this tool to manage AutoDL lifecycle.

Each streaming request has an absolute asyncio deadline (at most 1200 seconds),
including prefill and all response chunks. Timeout closes the client stream;
server-side cancellation/completion must be checked by the parent before any new
condition. The client does not kill server processes. Model HTTP is attempted
once, with raw response lines/headers, error body, exact request, times and usage
saved even when parsing or the service fails. GET /models and /metrics are
telemetry, not model inference costs.

## Interpretation and artifacts

`report.json` separates four layers: checkpoint declaration, observed /models
service window, real exact-prompt plus decode execution, and observed scheduler
concurrency. Actual returned prompt token IDs must match the complete offline
template hash; the service prompt usage must equal the offline count. Per-choice
decode IDs are counted and reconciled to actual completion usage; for n=1 only,
completion usage is a fallback actual count. An n=4 aggregate total is never
divided into fabricated per-sequence lengths. EOS-short output fails the declared
target even when HTTP is 200. All error/unknown usage is preserved as null plus
known lower bounds, without charging offline token estimates as actual usage.

`request-N.jsonl` holds complete SSE/raw evidence; `metrics.jsonl` holds full
sampled telemetry. A sampled running gauge must reach the target during request
intervals on the exclusive service. If polling misses a short decode, the target
is unproven; queued successful returns and HTTP overlap alone do not pass it.
The gauge demonstrates scheduler running membership, not a kernel execution
trace. Missing evidence is not proof of a model architecture limit.

`client_probe_e2e_seconds` covers this client invocation's preparation/admission,
HTTP and telemetry, but excludes model/service startup. `http_wall_seconds_sum`
can overlap in time and must not be called end-to-end elapsed time. Client TTFT
includes queue/network and is not pure prefill. Server prefill/decode/queue
histogram sum deltas are separate instrumentation; missing or stale metrics remain
unknown. No combined startup+probe lifecycle time is invented. Hash inventories
cover requests, manifest, metadata, supplied evidence and all output files.

Exit 0 means the tested condition passed exact prefill/decode, active sequence
proof and required metadata; exit 2 means failed/unproven. An ordinary exception
before HTTP is a local preparation/admission failure, not GPU inference.
CPU mocks and tokenizer preparation do not validate real model capacity. SemIf,
other checkpoints and final generation/repair/candidate/evidence/stop workloads
still need separate frozen-condition validation after their shapes stabilize.
