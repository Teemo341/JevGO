"""Finite P14h capacity evidence: actual lengths, usage, queue vs sequences."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

PATH = Path(__file__).resolve().parents[2] / "tools/p14_capacity.py"
spec = importlib.util.spec_from_file_location("p14_capacity", PATH)
p14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p14)


def request(n=1, target=2):
    return {"endpoint": "/v1/chat/completions", "payload": {"model": "test", "n": n, "max_tokens": target},
            "measurement": {"offline_input_tokens": 3, "token_ids_sha256": p14.ids_digest([1, 2, 3])}}


def chunks(n=1, actual=2, include_usage=True, include_ids=True):
    result = [{"prompt_token_ids": [1, 2, 3], "choices": [{"index": i,
        **({"token_ids": list(range(actual))} if include_ids else {}), "finish_reason": "length"} for i in range(n)]}]
    if include_usage:
        result.append({"usage": {"prompt_tokens": 3, "completion_tokens": actual * n}, "choices": []})
    return result


def test_actual_decode_not_max_tokens_and_no_fabricated_n4_per_sequence_counts():
    short = p14.response_summary(chunks(actual=1), request())
    assert not short["decode_target_met"] and not short["prefill_decode_pass"]
    assert short["actual_decode_per_sequence"] == {0: 1}
    n4 = p14.response_summary(chunks(n=4, include_ids=False), request(n=4))
    assert n4["output_tokens"] == 8
    assert n4["actual_decode_per_sequence"] == {0: None, 1: None, 2: None, 3: None}
    assert not n4["prefill_decode_pass"]
    assert p14.response_summary(chunks(n=4), request(n=4))["prefill_decode_pass"]


def test_missing_usage_is_unknown_even_with_actual_prompt_and_decode_ids():
    result = p14.response_summary(chunks(include_usage=False), request())
    assert result["input_tokens"] is None and result["output_tokens"] is None
    assert result["unknown_usage"] == ["input_tokens", "output_tokens"]
    assert result["actual_decode_per_sequence"] == {0: 2}
    assert not result["prefill_decode_pass"]


def test_prompt_service_truncation_and_wrong_ids_fail():
    data = chunks()
    data[0]["prompt_token_ids"] = [3, 2, 1]
    assert not p14.response_summary(data, request())["prefill_decode_pass"]
    data = chunks()
    data[-1]["usage"]["prompt_tokens"] = 2
    assert not p14.response_summary(data, request())["prefill_decode_pass"]


def test_queued_http_overlap_cannot_prove_concurrent_active_sequences():
    rows = [{"start_monotonic": 1, "end_monotonic": 9} for _ in range(4)]
    samples = [{"monotonic": 2, "status": 200, "raw": 'vllm:num_requests_running{engine="0"} 1\n'}]
    condition = {"http_requests": 4, "sequences": 4}
    metadata = {"exclusive_service": True, "runtime_source_hashes_verified": True, "active_sequence_evidence": {
        "metric": p14.METRIC, "unit": "scheduler_sequences", "source_sha256": {"scheduler": "proof"}}}
    result = p14.concurrency_summary(rows, samples, condition, metadata)
    assert result["http_overlap_peak"] == 4 and not result["active_sequences_target_met"]
    samples[0]["raw"] = 'vllm:num_requests_running{engine="0"} 4\n'
    assert p14.concurrency_summary(rows, samples, condition, metadata)["active_sequences_target_met"]
    assert not p14.concurrency_summary(rows, samples, condition, {})["active_sequences_target_met"]


def test_n4_is_one_http_and_needs_four_actual_active_sequences():
    rows = [{"start_monotonic": 1, "end_monotonic": 9}]
    result = p14.concurrency_summary(rows, [], {"http_requests": 1, "sequences": 4}, {})
    assert result["http_overlap_peak"] == 1
    assert result["target_sequences"] == 4 and not result["active_sequences_target_met"]


def test_duplicate_tp_or_dp_gauges_are_never_summed_to_fake_concurrency():
    rows = [{"start_monotonic": 1, "end_monotonic": 9}]
    samples = [{"monotonic": 2, "status": 200, "raw":
        'vllm:num_requests_running{engine="0"} 2\nvllm:num_requests_running{engine="1"} 2\n'}]
    result = p14.concurrency_summary(rows, samples, {"http_requests": 1, "sequences": 4}, {})
    assert result["scheduler_running_peak"] == 2


def test_stream_raw_body_and_usage_preserved_for_success_and_failure(tmp_path):
    async def run(status):
        body = "\n".join("data: " + json.dumps(c) for c in chunks()) + "\ndata: [DONE]\n"
        transport = httpx.MockTransport(lambda req: httpx.Response(status,
            text=body if status == 200 else '{"error":"OOM"}'))
        async with httpx.AsyncClient(transport=transport) as client:
            return await p14.request_once(client, "http://localhost", request(), tmp_path / f"{status}.jsonl", 2)
    good = asyncio.run(run(200))
    assert good["prefill_decode_pass"] and good["network_attempted"]
    bad = asyncio.run(run(500))
    assert bad["input_tokens"] is None and bad["output_tokens"] is None
    assert bad["error"] == "HTTP 500"
    assert 'OOM' in (tmp_path / "500.jsonl").read_text()


def test_total_request_deadline_catches_stream_that_never_finishes(tmp_path):
    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(1)
            yield b"data: [DONE]\n"
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda req: httpx.Response(200, stream=Slow()))) as client:
            return await p14.request_once(client, "http://localhost", request(), tmp_path / "timeout.jsonl", .01)
    result = asyncio.run(run())
    assert result["error"].startswith("TimeoutError:")
    assert result["http_wall_seconds"] < .5
    assert result["unknown_usage"] == ["input_tokens", "output_tokens"]


def test_immutable_predeclared_requests_reject_hash_changes(tmp_path):
    source = tmp_path / "request.json"
    source.write_text(json.dumps(request()))
    manifest = {"schema": p14.SCHEMA, "conditions": [{"id": "native", "window": 8,
        "requests": [{"file": source.name, "sha256": p14.digest(source)}]}]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    source.write_text("{}")
    with pytest.raises(ValueError, match="hash changed"):
        p14.load_condition(path, "native")


def test_same_manifest_condition_cannot_be_silently_retried(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{}')
    p14.claim_attempt(manifest, "w262144-single", tmp_path / "first")
    with pytest.raises(FileExistsError):
        p14.claim_attempt(manifest, "w262144-single", tmp_path / "second")
