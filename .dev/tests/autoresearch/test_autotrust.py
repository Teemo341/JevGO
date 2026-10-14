"""CPU protocol fixtures only; real AutoTrust checkpoint/GPU inference is pending."""
import json
import math
from pathlib import Path
import shutil
import tempfile

import httpx
import pytest
import yaml

from jevgo.core import (
    BudgetExceeded, BudgetLedger, BudgetLimits, GenerationBackend, GenerationRequest,
    JudgeBackend, JudgeRequest, ModelFailure, UnsupportedCapability, Usage, UsagePhase,
)
from jevgo.models.autotrust import AutoTrustBackend
from jevgo.models.serve_autotrust import build_launch, main


@pytest.fixture
def assets():
    scratch = Path(__file__).resolve().parents[2] / "tmp"
    scratch.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="p03-", dir=scratch))
    (root / "configs").mkdir()
    backbones = {}
    for key, name in (("jev_9b", "JEV-9B"), ("jev_27b", "JEV-27B"), ("jev_27b_vl", "JEV-27B-VL")):
        directory = root / "models" / name
        adapter = directory / "adapter_vllm"
        adapter.mkdir(parents=True)
        cfg = {"repository": f"autotrust/{name}", "local_directory": f"models/{name}",
               "revision": "fixture-revision", "base_model": "fixture-base"}
        if key == "jev_27b_vl":
            cfg["max_num_seqs"] = 8
        backbones[key] = cfg
        head = {"verbalizer_ids": list(range(100, 124)),
                "bias": [0.2, -0.1] + [0.0] * 6 + [0.3, -0.3] + [0.0] * 14,
                "slots": {"num_slots": 24, "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]},
                          "verbalizers": ["false", "true"] + list("012345ABCDEFGHIJKLMNOP"),
                          "template_version": "bare-v1"}}
        (adapter / "decision_head.json").write_text(json.dumps(head))
        (directory / "calibration.json").write_text(json.dumps({"per_kind": {"noul": 2, "choice": 0.5, "score": 1}}))
        (adapter / "adapter_config.json").write_text('{"r":32}')
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
            (directory / name).write_text("{}")
        (directory / "model.safetensors.index.json").write_text('{"weight_map":{"x":"model-1.safetensors"}}')
        (directory / "model-1.safetensors").write_bytes(b"CPU fixture, not a checkpoint")
        (adapter / "adapter_model.safetensors").write_bytes(b"CPU fixture")
        (directory / "serve_decide.py").write_text("raise AssertionError('must not start GPU service')\n")
    (root / "configs/jevgo.yaml").write_text(yaml.safe_dump({"backbones": backbones}))
    yield root
    shutil.rmtree(root)


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def decide_body(payload, probs=None, **overrides):
    options = payload.get("options", ["false", "true"] if payload["kind"] == "noul" else list("012345"))
    probs = probs if probs is not None else [1 / len(options)] * len(options)
    index = max(range(len(probs)), key=probs.__getitem__)
    return {"kind": payload["kind"], "effective_kind": payload["kind"], "options": options,
            "probabilities": probs, "choice_index": index, "choice": options[index],
            "num_model_requests": 1, "usage": {"prompt_tokens": 7, "completion_tokens": 1},
            "protocol": "fixture-bare-v1", "adaptation": "native", **overrides}


def completion_body(ids, logits=None, **overrides):
    logits = logits if logits is not None else [0] * len(ids)
    return {"model": "jev-decision", "choices": [{"index": 0, "logprobs": {
        "top_logprobs": [{f"token_id:{token}": value for token, value in zip(ids, logits)}]}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 1}, **overrides}


@pytest.mark.parametrize("model", ["jev_9b", "jev_27b", "jev_27b_vl"])
def test_generation_base_route_order_and_usage(assets, model):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert body["model"] != "jev-decision"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json={"model": body["model"], "choices": [
            {"index": 1, "message": {"content": "second"}}, {"index": 0, "message": {"content": "first"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 9}})
    ledger = BudgetLedger()
    with client_for(handler) as client:
        backend = AutoTrustBackend(model, assets_root=assets, client=client, base_url="http://local/v1")
        assert isinstance(backend, GenerationBackend) and isinstance(backend, JudgeBackend)
        result = backend.generate(GenerationRequest(prompt="task", system="rules", n=2, seed=31), ledger=ledger)
    assert result.texts == ("first", "second")
    assert seen[0]["seed"] == 31 and seen[0]["messages"][0]["role"] == "system"
    assert result.identity.adapter is None and result.identity.mode == "frozen"
    assert result.identity.revision == "fixture-revision"
    assert result.usage == ledger.used
    assert result.usage.generation_calls == result.usage.model_requests == 1
    assert result.usage.input_tokens == 10 and result.usage.output_tokens == 9


def test_9b_bias_temperature_and_option_identity(assets):
    def handler(request):
        assert request.url.path == "/v1/completions"
        body = json.loads(request.content)
        assert body["model"] == "jev-decision"
        assert body["allowed_token_ids"] == [108, 109]
        assert body["logprobs"] == 2 and body["max_tokens"] == 1
        assert body["top_k"] == 0 and body["top_p"] == 1
        assert body["add_special_tokens"] is False and body["return_tokens_as_token_ids"] is True
        assert 'A) [beta] "same\\ntext"' in body["prompt"]
        return httpx.Response(200, json=completion_body([108, 109], [-2, -1]))
    with client_for(handler) as client:
        backend = AutoTrustBackend("jev_9b", assets_root=assets, client=client)
        result = backend.score(JudgeRequest(role="candidate", context="goal", items=("same\ntext", "same\ntext"),
                                             options={"option_ids": ["beta", "alpha"]}), ledger=BudgetLedger())
    expected_first = 1 / (1 + math.exp(((-1 - 0.3) - (-2 + 0.3)) / 0.5))
    assert result.scores == pytest.approx((expected_first, 1 - expected_first))
    assert result.metadata["option_ids"] == ("beta", "alpha")
    assert result.metadata["decisions"][0]["option_token_ids"] == (108, 109)
    assert result.identity.adapter == "jev-decision"


@pytest.mark.parametrize("role,kind,word", [("evidence", "noul", "usefulness"),
    ("state_value", "score", "future progress"), ("stop", "noul", "stopping"), ("candidate", "noul", "reasonable")])
def test_roles_are_independent_item_opinions(assets, role, kind, word):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["kind"] == kind and word in body["question"]
        assert body["thinking"] == "off" and body["strategy"] == "single"
        probs = [0, 0, 0, 0.5, 0.5, 0] if kind == "score" else [0.25, 0.75]
        return httpx.Response(200, json=decide_body(body, probs))
    items = ("first",) if role == "candidate" else ("first", "second")
    ledger = BudgetLedger()
    with client_for(handler) as client:
        backend = AutoTrustBackend("jev_27b", assets_root=assets, client=client)
        result = backend.score(JudgeRequest(role=role, context="context", items=items), ledger=ledger)
    assert result.scores == pytest.approx([0.7 if kind == "score" else 0.75] * len(items))
    assert result.metadata["role"] == role and result.metadata["objective_success"] is False
    assert len(seen) == len(items)
    assert result.usage.judge_calls == 1 and result.usage.model_requests == len(items)
    assert result.usage.input_tokens == 7 * len(items) and result.usage == ledger.used


def test_27b_candidate_choice_uses_reported_internal_cost(assets):
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/decide" and body["kind"] == "choice"
        return httpx.Response(200, json=decide_body(body, [0.2, 0.8], num_model_requests=3,
                                                   usage={"prompt_tokens": 45, "completion_tokens": 12}))
    ledger = BudgetLedger()
    with client_for(handler) as client:
        result = AutoTrustBackend("jev_27b", assets_root=assets, client=client).score(
            JudgeRequest(role="candidate", context="goal", items=("a", "b")), ledger=ledger)
    assert result.scores == (0.2, 0.8)
    assert result.usage.model_requests == 3 and result.usage.input_tokens == 45 and result.usage.output_tokens == 12
    assert result.usage == ledger.used


def test_vl_encodes_images_for_both_protocols(assets):
    path = assets / "scene.png"
    path.write_bytes(b"fixture-image")
    seen = []
    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if request.url.path == "/v1/decide":
            assert body["state"][1]["image"].startswith("data:image/png;base64,")
            return httpx.Response(200, json=decide_body(body))
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "image"}}]})
    with client_for(handler) as client:
        backend = AutoTrustBackend("jev_27b_vl", assets_root=assets, client=client)
        backend.score(JudgeRequest(role="stop", context="goal", items=("answer",), image_paths=(str(path),)), ledger=BudgetLedger())
        backend.generate(GenerationRequest(prompt="describe", image_paths=(str(path),)), ledger=BudgetLedger())
    assert len(seen) == 2


@pytest.mark.parametrize("model", ["jev_9b", "jev_27b"])
def test_text_model_rejects_images_before_any_charge(assets, model):
    with client_for(lambda _: pytest.fail("Must not send request")) as client:
        backend = AutoTrustBackend(model, assets_root=assets, client=client)
        ledger = BudgetLedger()
        for request, method in [(GenerationRequest(prompt="x", image_paths=("missing.png",)), backend.generate),
                                (JudgeRequest(role="evidence", context="x", items=("a",), image_paths=("missing.png",)), backend.score)]:
            with pytest.raises(UnsupportedCapability):
                method(request, ledger=ledger)
        assert ledger.used == Usage()


@pytest.mark.parametrize("corruption", ["nan", "missing", "order", "count", "mass", "wrong_kind", "index", "thinking"])
def test_invalid_typed_responses_keep_known_cost(assets, corruption):
    def handler(request):
        body = decide_body(json.loads(request.content), [0.3, 0.7])
        if corruption == "nan":
            body["probabilities"][0] = float("nan")
        elif corruption == "missing":
            body.pop("probabilities")
        elif corruption == "order":
            body["options"].reverse()
        elif corruption == "count":
            body.pop("num_model_requests")
        elif corruption == "mass":
            body["probabilities"] = [0.8, 0.9]
        elif corruption == "wrong_kind":
            body["kind"] = "score"
        elif corruption == "index":
            body["choice_index"] = 0
        elif corruption == "thinking":
            body["thinking"] = {"used": True}
        return httpx.Response(200, content=json.dumps(body), headers={"content-type": "application/json"})
    ledger = BudgetLedger()
    with client_for(handler) as client:
        with pytest.raises(ModelFailure):
            AutoTrustBackend("jev_27b", assets_root=assets, client=client).score(
                JudgeRequest(role="candidate", context="x", items=("a", "b")), ledger=ledger)
    assert ledger.used.model_requests == 1 and ledger.used.judge_calls == 1
    assert ledger.used.input_tokens == 7 and ledger.used.output_tokens == 1
    assert ledger.entries[-1].failed
    if corruption == "count":
        assert "model_requests" in ledger.entries[-1].metadata["unknown_usage"]


@pytest.mark.parametrize("invalid", ["missing_slot", "nan", "infinity"])
def test_9b_invalid_logit_is_not_a_zero_probability(assets, invalid):
    def handler(request):
        body = completion_body([108, 109])
        values = body["choices"][0]["logprobs"]["top_logprobs"][0]
        if invalid == "missing_slot":
            del values["token_id:109"]
        else:
            values["token_id:109"] = float("nan") if invalid == "nan" else float("inf")
        return httpx.Response(200, content=json.dumps(body))
    ledger = BudgetLedger()
    with client_for(handler) as client:
        with pytest.raises(ModelFailure):
            AutoTrustBackend("jev_9b", assets_root=assets, client=client).score(
                JudgeRequest(role="candidate", context="x", items=("a", "b")), ledger=ledger)
    assert ledger.used.input_tokens == 11 and ledger.entries[-1].failed


@pytest.mark.parametrize("failure", ["timeout", "http", "malformed"])
def test_failure_attempt_unknown_usage_no_retry(assets, failure):
    attempts = []
    def handler(request):
        attempts.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("fixture timeout", request=request)
        if failure == "malformed":
            return httpx.Response(200, text="not JSON")
        return httpx.Response(500, json={"error": "fixture", "usage": {"prompt_tokens": 17}})
    ledger = BudgetLedger()
    with client_for(handler) as client:
        with pytest.raises(ModelFailure):
            AutoTrustBackend("jev_27b", assets_root=assets, client=client).score(
                JudgeRequest(role="evidence", context="x", items=("a", "b")), ledger=ledger)
    assert len(attempts) == 1 and ledger.used.model_requests == 1 and ledger.used.judge_calls == 1
    assert ledger.used.input_tokens == (17 if failure == "http" else 0)
    assert ledger.entries[-1].metadata["usage_complete"] is False
    assert "output_tokens" in ledger.entries[-1].metadata["unknown_usage"]
    assert ledger.entries[-1].metadata["model_requests_is_lower_bound"] is True


def test_partial_multi_item_failure_keeps_previous_cost(assets):
    seen = []
    def handler(request):
        seen.append(request)
        if len(seen) == 2:
            return httpx.Response(503, json={"error": "busy", "num_model_requests": 2,
                                            "usage": {"prompt_tokens": 20, "completion_tokens": 3}})
        return httpx.Response(200, json=decide_body(json.loads(request.content)))
    ledger = BudgetLedger()
    with client_for(handler) as client:
        with pytest.raises(ModelFailure):
            AutoTrustBackend("jev_27b", assets_root=assets, client=client).score(
                JudgeRequest(role="evidence", context="x", items=("a", "b", "c")), ledger=ledger)
    assert len(seen) == 2 and ledger.used.model_requests == 3
    assert ledger.used.input_tokens == 27 and ledger.used.output_tokens == 4


def test_missing_usage_success_is_explicitly_unknown(assets):
    def handler(request):
        body = decide_body(json.loads(request.content))
        del body["usage"]
        return httpx.Response(200, json=body)
    with client_for(handler) as client:
        result = AutoTrustBackend("jev_27b", assets_root=assets, client=client).score(
            JudgeRequest(role="stop", context="x", items=("a",)), ledger=BudgetLedger())
    assert result.metadata["usage_complete"] is False
    assert result.metadata["unknown_usage"] == ("input_tokens", "output_tokens")


@pytest.mark.parametrize("resource,limit", [("model_requests", 0), ("judge_calls", 0),
                                             ("input_tokens", 10), ("output_tokens", 0)])
def test_budget_rejection_prevents_http_and_charges(assets, resource, limit):
    ledger = BudgetLedger(BudgetLimits(**{resource: limit}))
    with client_for(lambda _: pytest.fail("Must not send")) as client:
        with pytest.raises(BudgetExceeded):
            AutoTrustBackend("jev_27b", assets_root=assets, client=client).score(
                JudgeRequest(role="stop", context="x", items=("a",)), ledger=ledger)
    assert ledger.used == Usage()


def test_commit_reserve_and_explicit_token_bound(assets):
    ledger = BudgetLedger(BudgetLimits(input_tokens=30), commit_reserve=Usage(input_tokens=20))
    with client_for(lambda request: httpx.Response(200, json=completion_body([100, 101]))) as client:
        backend = AutoTrustBackend("jev_9b", assets_root=assets, client=client)
        request = JudgeRequest(role="stop", context="x", items=("a",), options={"input_tokens_upper_bound": 12})
        with pytest.raises(BudgetExceeded):
            backend.score(request, ledger=ledger)
        result = backend.score(request, ledger=ledger, phase=UsagePhase.COMMIT)
    assert result.usage.input_tokens == 11
    assert all(entry.phase == UsagePhase.COMMIT for entry in ledger.entries)
    assert ledger.commit_reserve.input_tokens == 20


@pytest.mark.parametrize("ids", [["x", "x"], ["x"], ["x", ""], [0, "b"]])
def test_invalid_option_ids_fail_before_request(assets, ids):
    ledger = BudgetLedger()
    with client_for(lambda _: pytest.fail("Must not send")) as client:
        with pytest.raises(ValueError):
            AutoTrustBackend("jev_9b", assets_root=assets, client=client).score(
                JudgeRequest(role="candidate", context="x", items=("a", "b"), options={"option_ids": ids}), ledger=ledger)
    assert ledger.used == Usage()


@pytest.mark.parametrize("model", ["jev_9b", "jev_27b", "jev_27b_vl"])
def test_launch_is_local_offline_and_lora_flags(assets, model):
    command, env = build_launch(model, assets_root=assets, executable="/local/python", port=8123, max_model_len=8192)
    assert command[0] == "/local/python"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "8123"
    assert command[command.index("--logprobs-mode") + 1] == "processed_logprobs"
    assert command[command.index("--max-lora-rank") + 1] == "32"
    assert command[command.index("--lora-modules") + 1].startswith("jev-decision=" + str(assets))
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["JEV_DECIDE_THINKING"] == "off"
    if model == "jev_9b":
        assert command[1:3] == ["-m", "vllm.entrypoints.openai.api_server"]
    else:
        assert command[1].endswith("serve_decide.py") and "--trust-request-chat-template" in command
    if model == "jev_27b_vl":
        assert int(command[command.index("--max-num-seqs") + 1]) <= 8


def test_launcher_rejects_vl_excess_concurrency_and_missing_shards(assets):
    config_path = assets / "configs/jevgo.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["backbones"]["jev_27b_vl"]["max_num_seqs"] = 9
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="max_num_seqs"):
        build_launch("jev_27b_vl", assets_root=assets)
    (assets / "models/JEV-9B/model-1.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="Local serving files missing"):
        build_launch("jev_9b", assets_root=assets)


def test_dry_run_does_not_exec_or_import_vllm(assets, capsys, monkeypatch):
    monkeypatch.setattr("jevgo.models.serve_autotrust.os.execvpe", lambda *args: pytest.fail("dry run cannot exec"))
    assert main(["--model", "jev_27b_vl", "--assets-root", str(assets), "--dry-run"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "GPU pending" in data["validation"]


def test_injected_client_is_not_closed(assets):
    with client_for(lambda _: pytest.fail("Must not send")) as client:
        backend = AutoTrustBackend("jev_9b", assets_root=assets, client=client)
        backend.close()
        assert not client.is_closed
        with pytest.raises(ModelFailure, match="closed"):
            backend.generate(GenerationRequest(prompt="x"), ledger=BudgetLedger())


def test_measured_overage_stops_later_work(assets):
    ledger = BudgetLedger(BudgetLimits(wall_seconds=1))
    ledger.record(Usage(wall_seconds=2), source="prior operation")
    with client_for(lambda _: pytest.fail("Must stop after a measured overage")) as client:
        backend = AutoTrustBackend("jev_9b", assets_root=assets, client=client)
        with pytest.raises(BudgetExceeded):
            backend.generate(GenerationRequest(prompt="x"), ledger=ledger)
    assert ledger.used.generation_calls == 0


@pytest.mark.parametrize("role,ids,logits,expected", [
    ("stop", [100, 101], [-1, -2], 1 / (1 + math.exp(((-1 + 0.2) - (-2 - 0.1)) / 2))),
    ("state_value", list(range(102, 108)), [0, 0, 0, 0, 0, 0], 0.5),
])
def test_9b_noul_and_score_use_their_own_slots(assets, role, ids, logits, expected):
    def handler(request):
        body = json.loads(request.content)
        assert body["allowed_token_ids"] == ids
        return httpx.Response(200, json=completion_body(ids, logits))
    with client_for(handler) as client:
        result = AutoTrustBackend("jev_9b", assets_root=assets, client=client).score(
            JudgeRequest(role=role, context="goal", items=("item",)), ledger=BudgetLedger())
    assert result.scores == pytest.approx((expected,))


@pytest.mark.parametrize("corruption", ["length", "duplicate", "missing_index", "wrong_model", "nontext"])
def test_invalid_generation_keeps_token_usage(assets, corruption):
    def handler(request):
        body = {"model": "autotrust/JEV-9B", "choices": [
            {"index": 0, "message": {"content": "one"}}, {"index": 1, "message": {"content": "two"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        if corruption == "length":
            body["choices"].pop()
        elif corruption == "duplicate":
            body["choices"][1]["index"] = 0
        elif corruption == "missing_index":
            body["choices"][1]["index"] = 2
        elif corruption == "wrong_model":
            body["model"] = "jev-decision"
        else:
            body["choices"][1]["message"]["content"] = None
        return httpx.Response(200, json=body)
    ledger = BudgetLedger()
    with client_for(handler) as client:
        with pytest.raises(ModelFailure):
            AutoTrustBackend("jev_9b", assets_root=assets, client=client).generate(
                GenerationRequest(prompt="x", n=2), ledger=ledger)
    assert ledger.used.generation_calls == ledger.used.model_requests == 1
    assert ledger.used.input_tokens == 3 and ledger.used.output_tokens == 4
    assert ledger.entries[-1].failed


@pytest.mark.parametrize("count", [17, 257])
def test_native_choice_limits_are_explicit(assets, count):
    model = "jev_9b" if count == 17 else "jev_27b"
    with client_for(lambda _: pytest.fail("No fallback or splitting")) as client:
        ledger = BudgetLedger()
        with pytest.raises(ValueError, match="Too many"):
            AutoTrustBackend(model, assets_root=assets, client=client).score(
                JudgeRequest(role="candidate", context="x", items=tuple(str(i) for i in range(count))), ledger=ledger)
    assert ledger.used == Usage()


def test_string_option_ids_are_not_silently_split(assets):
    with client_for(lambda _: pytest.fail("No request")) as client:
        with pytest.raises(ValueError, match="ordered array"):
            AutoTrustBackend("jev_9b", assets_root=assets, client=client).score(
                JudgeRequest(role="candidate", context="x", items=("a", "b"), options={"option_ids": "xy"}),
                ledger=BudgetLedger())


def test_invalid_local_head_fails_without_network(assets):
    path = assets / "models/JEV-9B/adapter_vllm/decision_head.json"
    head = json.loads(path.read_text())
    head["verbalizer_ids"].pop()
    path.write_text(json.dumps(head))
    with pytest.raises(ValueError, match="24 slots"):
        AutoTrustBackend("jev_9b", assets_root=assets)
