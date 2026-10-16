# P14a 9B context pretrial

Development-only EVAL tooling, covering AR-2 requests/repair, AR-5 transient memory
and AR-7 accounting. All model parameters remain frozen. This tool does not
change published defaults, MCTS, admissible actions or objective success.

`prepare` and `offline` never load weights or send HTTP requests. `protocol` and
`run` execute real inference only after the parent task has allocated a GPU and
started the matching service. AutoDL preparation, GPU allocation, short private
TMPDIR, service stdout/stderr, argv/PID/exit status, sampled GPU peaks and cleanup
belong to that parent orchestration; this script never manages a machine's power
or terminates processes. A sampled GPU peak is not an allocator-exact peak.

## Local preparation (already generated artifacts must not be overwritten)

Run from the P14a worktree with the shared absolute interpreter:

```bash
export JEVGO_ASSETS_ROOT=/mnt/nas-tianwen/home/ssy/JevGO
export JEVGO_PY=/mnt/nas-tianwen/home/ssy/JevGO/.venv/bin/python
export P14A_OUT=/mnt/nas-tianwen/home/ssy/JevGO/.dev/artifacts/P14-20261017-pretrial/a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$JEVGO_PY" -B .dev/tools/p14_context.py prepare \
  --history "$JEVGO_ASSETS_ROOT/.dev/artifacts/gpu-20261014-w1d2iqko" \
  --base-url http://127.0.0.1:8019 --output "$P14A_OUT"
"$JEVGO_PY" -B .dev/tools/p14_context.py offline \
  --sources "$P14A_OUT/historical-requests.json" --output "$P14A_OUT/offline"
"$JEVGO_PY" -B -m pytest -q .dev/tests/autoresearch/test_p14_context.py --basetemp=.dev/tmp/p14-context-tests
```

The four YAML files derive from the actual historical B0/B3 smoke configurations.
Only the declared model URL/window, method and memory setting are set explicitly.
Each keeps MiniGrid-Empty-5x5-v0 seed 0, no training, generation reserve 256,
candidate_count 4, simulations 8, depth 4, rollout 2, max_steps 20, the original
commit reserve, 3000 cumulative requests and 2000 environment steps. There is no
input/output token hard cap. 9B judge reserve stays 1 in the production backend.

`historical-requests.json` preserves every recorded logical generation/judge
request and its following usage entries, source line/ID and source hashes.
`offline/lengths.jsonl` expands all individual HTTP payloads through the production
AutoTrust builder and counts the checkpoint's real tokenizer/chat template.
Generation includes system framing and the disabled-thinking generation prefix;
9B judge uses the bare-v1 completion with no extra special tokens. Reserve is
per sequence; `n=4` does not require four prompt copies in a single context.
The installed vLLM reports shared prompt tokens once and completion tokens across
all sequences. Offline lengths never fill in historical unknown usage.

The exact historical samples are kept unchanged. Separate counterfactual
suffix-memory samples retain the last four memories, or six before evidence
selection (four retained plus a newly appended hypothesis/observation), and four
evidence items. These are length probes, **not** a JEV memory-selection simulation
or a bound on future trajectory lengths. Additional repair probes append the
exact production repair suffix and increment the seed; they are unexecuted probes.
Expansion includes the unexecuted later items of an interrupted logical judge
call. Thus expanded item counts are not historical HTTP request counts. Each
`source_request_actual_usage` refers only to the matching item's saved response,
or is null when no response was recorded; the source bundle retains the complete
logical request's ledger entries. Repeated rows for different candidate windows
must not be summed as new or historical costs.

## Remote commands after the parent prepares the assigned AutoDL

The parent transfers only `historical-requests.json` and the four generated YAML
files to the same project-relative artifact path. The tool is tracked source.
No historical artifact directory, Python environment or weights need copying.
Set the remote roots to the actual prepared paths, for example:

```bash
cd /root/JevGO
export JEVGO_ASSETS_ROOT=/root/JevGO
export JEVGO_PY=/root/JevGO/.venv/bin/python
export P14A_OUT=/root/JevGO/.dev/artifacts/P14-20261017-pretrial/a
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$JEVGO_PY" -B .dev/tools/p14_context.py offline \
  --sources "$P14A_OUT/historical-requests.json" --output "$P14A_OUT/offline-remote"
```

After recording live allocation and selecting the assigned GPU, the parent
starts one window at a time. The service command (first add `--dry-run`) is:

```bash
"$JEVGO_PY" -B -m jevgo.models.serve_autotrust \
  --model jev_9b --assets-root "$JEVGO_ASSETS_ROOT" \
  --host 127.0.0.1 --port 8019 --max-model-len 4096
```

The parent supplies the selected `CUDA_VISIBLE_DEVICES` and short owned `TMPDIR`,
captures the service's startup/capacity logs, and waits for readiness. Run the
protocol followed by the two loops:

```bash
"$JEVGO_PY" -B .dev/tools/p14_context.py protocol \
  --config "$P14A_OUT/9b-4096-memory4-B0.yaml" \
  --output "$P14A_OUT/4096-protocol" --audit-output "$P14A_OUT/4096-protocol-http"
"$JEVGO_PY" -B .dev/tools/p14_context.py run \
  --config "$P14A_OUT/9b-4096-memory4-B0.yaml" \
  --output "$P14A_OUT/4096-B0" --audit-output "$P14A_OUT/4096-B0-http"
"$JEVGO_PY" -B .dev/tools/p14_context.py run \
  --config "$P14A_OUT/9b-4096-memory4-B3.yaml" \
  --output "$P14A_OUT/4096-B3" --audit-output "$P14A_OUT/4096-B3-http"
"$JEVGO_PY" -B .dev/tools/p14_context.py audit \
  --run "$P14A_OUT/4096-B0" --audit "$P14A_OUT/4096-B0-http" --output "$P14A_OUT/4096-B0-audit.json"
"$JEVGO_PY" -B .dev/tools/p14_context.py audit \
  --run "$P14A_OUT/4096-B3" --audit "$P14A_OUT/4096-B3-http" --output "$P14A_OUT/4096-B3-audit.json"
```

After stopping only the owned 4096 service, repeat with service window 8192 and
replace configuration `4096-memory4` with `8192-memory32`, and output labels
`4096` with `8192`. Every output directory must be new. Keep failed attempts and
give retries distinct directories. Runtime roots are parameterized; pass global
`--assets-root /actual/path` before the subcommand when not using the environment.

`protocol` uses the unchanged production four-request check (generation reserve
128 plus candidate/evidence/stop judges). This is short protocol evidence;
closed-loop generation always reserves 256. Startup + short protocol do not
prove full-length capacity or effectiveness. The subsequent long loop is needed.

## Interpretation

Before inference the tool reads `/v1/models` and requires the base model's actual
`max_model_len` to equal the client YAML. Missing capacity evidence fails closed.
The development backend checks every complete payload immediately before the
HTTP request. It never truncates. Local window rejection is recorded with no
HTTP/model-request charge; the already-started logical generation/judge call may
still count. HTTP failures retain response bodies and unknown usage. Clean budget
admission stops are distinguished from model/context errors.

`http.jsonl` pairs complete request preflight and completion records, preserving
raw HTTP responses, actual usage, offline counts, reserve, public action/memory
counts, phase and role. `audit` runs the production event/ledger/result audit,
then checks journal pairing, HTTP-attempt counts, exact per-request ledger entries
and known raw-response token totals. Unknown usage prevents technical acceptance.
Every successful request must match the offline prompt count, detecting silent
truncation or service/template differences. A mismatch remains a failure.

For memory32, the original boundary must be covered by a successful **generation**
request in B0 or **evidence** request in B3 with both at least 9 branch actions and
19 memory entries, or objective success must already have occurred. A budget stop
before that boundary is inconclusive. All conditions preserve original failures;
memory4 cannot erase memory32's failure. A clean budget_exhausted stop returns
exit zero while preserving its original error text and termination reason, so
the parent can continue the independent arm. CLI exit zero does not establish
task success, context resolution or effectiveness. Technical acceptance also
requires complete costs within the declared limits and consistent over-budget
metadata.
