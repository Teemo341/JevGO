"""Frozen AutoTrust HTTP backends; local assets only, no implicit retries.

``AutoTrustBackend('jev_27b', assets_root=..., base_url='http://127.0.0.1:8000')``
implements both core model protocols. The URL may optionally end in /v1.
Generation uses the base model; judging uses jev-decision with thinking off.

Request options: ``input_tokens_upper_bound`` is an optional bound PER internal
model request (otherwise max_model_len is conservatively reserved). Generation
also accepts ``enable_thinking``; judging accepts unique aligned ``option_ids``.
No request option can override routes, adapters, decision kinds or model names.
Scores are model opinions, never objective success or calibrated task returns.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import time
from typing import Any

import httpx
import yaml

from jevgo.core import (
    BudgetExceeded, BudgetLedger, GenerationRequest, GenerationResponse, JudgeRequest, JudgeResponse,
    JudgeRole, ModelFailure, ModelIdentity, UnsupportedCapability, Usage, UsagePhase,
)

MODELS = ("jev_9b", "jev_27b", "jev_27b_vl")
ADAPTER = "jev-decision"


def load_model_config(model: str, assets_root: str | Path) -> tuple[dict, Path]:
    """Resolve one explicitly selected model from the existing local project config."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    root = Path(assets_root).expanduser().resolve()
    with (root / "configs/jevgo.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)["backbones"][model]
    directory = (root / cfg["local_directory"]).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Local model directory missing: {directory}")
    return cfg, directory


def _finite(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric value")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    json.dumps(data, allow_nan=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


@dataclass
class _Call:
    usage: Usage
    requests: list[dict] = field(default_factory=list)


class AutoTrustBackend:
    """Synchronous backend with injectable httpx-compatible client.

    The backend closes only its own client. All costs are charged here exactly
    once; returned usage is a report, not an instruction for the caller to charge.
    max_model_len must match the service setting, for conservative token bounds.
    """
    def __init__(self, model: str, *, assets_root: str | Path,
                 base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0,
                 client: Any = None, max_model_len: int = 4096) -> None:
        self.config, self.model_dir = load_model_config(model, assets_root)
        self.model = model
        self.max_model_len = _positive_int(max_model_len, "max_model_len")
        if _finite(timeout) <= 0:
            raise ValueError("timeout must be positive")
        url = httpx.URL(base_url)
        if url.scheme not in ("http", "https") or not url.host or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) service URL")
        self.base_url = str(url).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.timeout = timeout
        self.served_model = self.config["repository"]
        self._head = _read_json(self.model_dir / "adapter_vllm/decision_head.json")
        self._temps = _read_json(self.model_dir / "calibration.json")["per_kind"]
        self._validate_head()
        fingerprints = {name: hashlib.sha256((self.model_dir / name).read_bytes()).hexdigest()
                        for name in ("adapter_vllm/decision_head.json", "calibration.json")}
        self.identity = ModelIdentity(
            backend="autotrust", model=self.served_model, revision=self.config.get("revision"),
            metadata={"model_key": model, "base_model": self.config["base_model"],
                      "model_directory": str(self.model_dir), "asset_sha256": fingerprints,
                      "generation_adapter": None, "judge_adapter": ADAPTER},
        )
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._closed = False

    def _validate_head(self) -> None:
        head = self._head
        ids, bias, slots = head["verbalizer_ids"], head["bias"], head["slots"]
        if slots.get("template_version") != "bare-v1":
            raise ValueError("Unsupported AutoTrust decision template")
        expected = {"noul": ["false", "true"], "score": list("012345"),
                    "choice": list("ABCDEFGHIJKLMNOP")}
        if not len(ids) == len(bias) == len(slots["verbalizers"]) == slots["num_slots"] == 24:
            raise ValueError("Decision head must contain all 24 slots")
        if any(type(token) is not int or token < 0 for token in ids) or len(set(ids)) != len(ids):
            raise ValueError("Decision token IDs must be unique nonnegative integers")
        for value in bias:
            _finite(value)
        for kind, labels in expected.items():
            start, end = slots["ranges"][kind]
            if (type(start) is not int or type(end) is not int or start < 0
                    or end > len(ids) or slots["verbalizers"][start:end] != labels):
                raise ValueError(f"Invalid decision slots for {kind}")
            if _finite(self._temps[kind]) <= 0:
                raise ValueError(f"Invalid calibration temperature for {kind}")

    def close(self) -> None:
        if not self._closed and self._owns_client:
            self._client.close()
        self._closed = True

    def _options(self, options: Mapping, allowed: set[str]) -> int:
        if self._closed:
            raise ModelFailure("AutoTrust backend is closed")
        extra = set(options) - allowed - {"input_tokens_upper_bound"}
        if extra:
            raise ValueError(f"Unsupported AutoTrust options: {sorted(extra)}")
        bound = _positive_int(options.get("input_tokens_upper_bound", self.max_model_len),
                              "input_tokens_upper_bound")
        if bound > self.max_model_len:
            raise ValueError("input_tokens_upper_bound exceeds max_model_len")
        return bound

    def _images(self, image_paths: tuple[str, ...]) -> list[str]:
        if image_paths and self.model != "jev_27b_vl":
            raise UnsupportedCapability(f"{self.model} uses a text-only local checkpoint")
        if len(image_paths) > 8:
            raise ValueError("The VL service supports at most eight images per prompt")
        images = []
        for name in image_paths:
            path = Path(name)
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                raise ValueError(f"Unsupported image format: {path}")
            images.append(f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii"))
        return images

    @staticmethod
    def _accounting(data: Any, typed: bool) -> tuple[Usage, dict]:
        body = data if isinstance(data, dict) else {}
        raw = body.get("usage")
        raw = raw if isinstance(raw, dict) else {}
        counts, unknown = {}, []
        for remote, local in (("prompt_tokens", "input_tokens"), ("completion_tokens", "output_tokens")):
            value = raw.get(remote)
            if type(value) is int and value >= 0:
                counts[local] = value
            else:
                unknown.append(local)
        count = body.get("num_model_requests") if typed else 1
        if type(count) is not int or count < 1:
            count = 1  # One HTTP attempt is known; hidden server work is unknown.
            unknown.append("model_requests")
        total = raw.get("total_tokens")
        return Usage(model_requests=count - 1, **counts), {
            "reported_total_tokens": total if type(total) is int and total >= 0 else None,
            "num_model_requests": count, "unknown_usage": unknown,
            "usage_complete": not unknown,
            "model_requests_is_lower_bound": "model_requests" in unknown,
        }

    @staticmethod
    def _ensure_available(ledger: BudgetLedger, usage: Usage, phase: UsagePhase) -> None:
        # A prior measured overage (including wall time) must stop later work.
        if ledger.over_budget:
            raise BudgetExceeded(ledger.over_budget, UsagePhase(phase))
        ledger.ensure_available(usage, phase=phase)

    def _request(self, path: str, payload: dict, parse: Callable, call: _Call,
                 ledger: BudgetLedger, phase: UsagePhase, *, request_bound: int,
                 input_bound: int, output_bound: int) -> Any:
        self._ensure_available(ledger, Usage(model_requests=request_bound,
                                             input_tokens=input_bound * request_bound,
                                             output_tokens=output_bound), phase)
        fixed = Usage(model_requests=1)
        ledger.charge(fixed, source="autotrust.request", phase=phase,
                      metadata={"endpoint": path, "attempt": len(call.requests) + 1})
        call.usage = call.usage + fixed
        started, data, failed = time.monotonic(), None, True
        status = None
        try:
            response = self._client.post(self.base_url + path, json=payload, timeout=self.timeout)
            status = response.status_code
            data = response.json()
            response.raise_for_status()
            if not isinstance(data, dict):
                raise ValueError("Expected response object")
            json.dumps(data, allow_nan=False)
            if "error" in data:
                raise ValueError("Service returned an error object")
            value = parse(data)
            failed = False
            return value
        except Exception as error:
            if isinstance(error, ModelFailure):
                raise
            raise ModelFailure(f"AutoTrust {path} failed: {type(error).__name__}: {error}") from error
        finally:
            measured, metadata = self._accounting(data, path == "/v1/decide")
            measured = measured.model_copy(update={"wall_seconds": time.monotonic() - started})
            metadata.update(endpoint=path, status_code=status, failed=failed,
                            input_tokens_upper_bound=input_bound, model_requests_upper_bound=request_bound)
            ledger.record(measured, source="autotrust.response", phase=phase,
                          failed=failed, metadata=metadata)
            call.usage = call.usage + measured
            call.requests.append(metadata)

    @staticmethod
    def _metadata(call: _Call) -> dict:
        return {"requests": call.requests, "num_model_requests": call.usage.model_requests,
                "usage_complete": all(record["usage_complete"] for record in call.requests),
                "unknown_usage": sorted({key for record in call.requests for key in record["unknown_usage"]})}

    def generate(self, request: GenerationRequest, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> GenerationResponse:
        input_bound = self._options(request.options, {"enable_thinking"})
        images = self._images(request.image_paths)
        if request.max_new_tokens > self.max_model_len:
            raise ValueError("max_new_tokens exceeds max_model_len")
        thinking = request.options.get("enable_thinking", False)
        if type(thinking) is not bool:
            raise ValueError("enable_thinking must be a bool")
        content: Any = request.prompt
        if images:
            content = [{"type": "text", "text": request.prompt}] + [
                {"type": "image_url", "image_url": {"url": image}} for image in images]
        messages = ([{"role": "system", "content": request.system}] if request.system else [])
        messages.append({"role": "user", "content": content})
        payload = {"model": self.served_model, "messages": messages, "n": request.n,
                   "max_tokens": request.max_new_tokens, "temperature": request.temperature,
                   "seed": request.seed, "chat_template_kwargs": {"enable_thinking": thinking}}
        bound = Usage(generation_calls=1, model_requests=1, input_tokens=input_bound * request.n,
                      output_tokens=request.n * request.max_new_tokens)
        self._ensure_available(ledger, bound, phase)
        fixed = Usage(generation_calls=1)
        ledger.charge(fixed, source="autotrust.generate", phase=phase)
        call = _Call(fixed)

        def parse(data: dict) -> tuple[str, ...]:
            if data.get("model", self.served_model) != self.served_model:
                raise ValueError("Generation response model does not match base model")
            choices = data["choices"]
            if not isinstance(choices, list) or len(choices) != request.n:
                raise ValueError("Generation response must have exactly n choices")
            by_index = {}
            for choice in choices:
                index, text = choice["index"], choice["message"]["content"]
                if type(index) is not int or index in by_index or not isinstance(text, str):
                    raise ValueError("Invalid generation index or content")
                by_index[index] = text
            if set(by_index) != set(range(request.n)):
                raise ValueError("Generation choice indices do not align")
            return tuple(by_index[i] for i in range(request.n))

        texts = self._request("/v1/chat/completions", payload, parse, call, ledger, phase,
                              request_bound=1, input_bound=input_bound * request.n,
                              output_bound=request.n * request.max_new_tokens)
        return GenerationResponse(texts=texts, identity=self.identity, usage=call.usage,
                                  metadata={**self._metadata(call), "adapter": None,
                                            "enable_thinking": thinking})

    def _decision(self, kind: str, state: Any, question: str, options: list[str],
                  call: _Call, ledger: BudgetLedger, phase: UsagePhase,
                  input_bound: int) -> tuple[list[float], dict]:
        if self.model != "jev_9b":
            payload = {"kind": kind, "state": state, "question": question,
                       "thinking": "off", "strategy": "single"}
            if kind == "choice":
                payload["options"] = options

            def parse(data: dict) -> tuple[list[float], dict]:
                _positive_int(data["num_model_requests"], "num_model_requests")
                if data["kind"] != kind or data.get("effective_kind", kind) != kind:
                    raise ValueError("Decision kind mismatch")
                if data["options"] != options:
                    raise ValueError("Decision options changed order or identity")
                if data.get("model", self.served_model) != self.served_model:
                    raise ValueError("Decision response model mismatch")
                probs = data["probabilities"]
                if not isinstance(probs, list) or len(probs) != len(options):
                    raise ValueError("Decision probabilities must cover all options")
                probs = [_finite(value) for value in probs]
                if any(value < 0 or value > 1 for value in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-5):
                    raise ValueError("Invalid probability distribution")
                if "choice_index" in data:
                    index = data["choice_index"]
                    if (type(index) is not int or not 0 <= index < len(options)
                            or probs[index] != max(probs) or data.get("choice", options[index]) != options[index]):
                        raise ValueError("Decision choice does not match its distribution")
                if data.get("thinking", {}).get("used", False):
                    raise ValueError("Unexpected thinking despite thinking=off")
                return probs, {key: data[key] for key in ("protocol", "adaptation") if key in data}

            # The bundled server can retry a readout internally. Reserve both paths.
            request_bound = 2 * math.ceil(len(options) / 128)
            return self._request("/v1/decide", payload, parse, call, ledger, phase,
                                 request_bound=request_bound, input_bound=input_bound,
                                 output_bound=request_bound)
        start, end = self._head["slots"]["ranges"][kind]
        if len(options) > end - start:
            raise ValueError("Choice exceeds the local decision head capacity")
        ids = self._head["verbalizer_ids"][start:start + len(options)]
        lines = [f"{'ABCDEFGHIJKLMNOP'[i]}) {value}" for i, value in enumerate(options)] if kind == "choice" else options
        prompt = f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
        payload = {"model": ADAPTER, "prompt": prompt, "max_tokens": 1, "temperature": 1.0,
                   "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0,
                   "logprobs": len(options), "allowed_token_ids": ids,
                   "add_special_tokens": False, "return_tokens_as_token_ids": True}

        def parse(data: dict) -> tuple[list[float], dict]:
            if data.get("model", ADAPTER) != ADAPTER or len(data["choices"]) != 1:
                raise ValueError("Unexpected decision model or number of completions")
            if type(data["choices"][0].get("index")) is not int or data["choices"][0]["index"] != 0:
                raise ValueError("Invalid decision completion index")
            rows = data["choices"][0]["logprobs"]["top_logprobs"]
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
                raise ValueError("Expected exactly one option-token readout")
            logits = [_finite(rows[0][f"token_id:{token}"]) for token in ids]
            values = [(value + self._head["bias"][start + i]) / self._temps[kind]
                      for i, value in enumerate(logits)]
            values = [_finite(value) for value in values]
            exponentials = [math.exp(value - max(values)) for value in values]
            total = sum(exponentials)
            return [value / total for value in exponentials], {
                "option_token_ids": ids, "raw_logprobs": logits,
                "bias": self._head["bias"][start:start + len(options)],
                "temperature": self._temps[kind], "protocol": "bare-v1",
            }

        return self._request("/v1/completions", payload, parse, call, ledger, phase,
                             request_bound=1, input_bound=input_bound, output_bound=1)

    def score(self, request: JudgeRequest, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.SEARCH) -> JudgeResponse:
        input_bound = self._options(request.options, {"option_ids"})
        images = self._images(request.image_paths)
        raw_ids = request.options.get("option_ids", [f"item-{i}" for i in range(len(request.items))])
        if not isinstance(raw_ids, (list, tuple)):
            raise ValueError("option_ids must be an ordered array")
        ids = list(raw_ids)
        if (len(ids) != len(request.items) or any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("option_ids must be unique nonempty strings aligned with items")
        is_choice = request.role == JudgeRole.CANDIDATE and len(request.items) > 1
        if is_choice and len(request.items) > (16 if self.model == "jev_9b" else 256):
            raise ValueError("Too many candidates for the selected AutoTrust model")
        if is_choice:
            kind, question = "choice", "Which candidate is the most reasonable next action for the stated goal?"
            scale = "relative_candidate_preference"
        elif request.role == JudgeRole.STATE_VALUE:
            kind, question = "score", ("Rate the remaining potential for useful future progress from this state, "
                                       "0 (none) to 5 (high). Exclude rewards already observed; this is a heuristic.")
            scale = "expected_ordinal_rating_0_to_5_divided_by_5"
        else:
            kind = "noul"
            question = {
                JudgeRole.CANDIDATE: "Is this candidate a reasonable next action for the stated goal?",
                JudgeRole.EVIDENCE: ("Is this evidence useful for resolving the research question, including useful "
                                     "contradictions or failures? Assess usefulness, not whether it confirms a belief."),
                JudgeRole.STOP: ("Does this item support stopping and submitting the available result given the "
                                 "stated criteria and evidence? This is advisory, not an objective success signal."),
            }[request.role]
            scale = "model_probability_true_for_role_question"
        count = 1 if is_choice else len(request.items)
        per_request = 1 if self.model == "jev_9b" else 2 * math.ceil((len(ids) if is_choice else 6) / 128)
        self._ensure_available(ledger, Usage(judge_calls=1, model_requests=count * per_request,
                                             input_tokens=count * per_request * input_bound,
                                             output_tokens=count * per_request), phase)
        fixed = Usage(judge_calls=1)
        ledger.charge(fixed, source="autotrust.score", phase=phase)
        call, scores, decisions = _Call(fixed), [], []
        groups = [list(range(len(ids)))] if is_choice else [[i] for i in range(len(ids))]
        for group in groups:
            state: Any = request.context
            if is_choice:
                options = [f"[{ids[i]}] {json.dumps(request.items[i], ensure_ascii=False)}" for i in group]
            else:
                i = group[0]
                state += f"\nItem [{ids[i]}]:\n{request.items[i]}"
                options = [str(i) for i in range(6)] if kind == "score" else ["false", "true"]
            if images:
                state = [state] + [{"image": image} for image in images]
            probs, detail = self._decision(kind, state, question, options, call, ledger, phase, input_bound)
            if is_choice:
                scores.extend(probs)
            else:
                scores.append(sum(i * value for i, value in enumerate(probs)) / 5 if kind == "score" else probs[1])
            decisions.append({"kind": kind, "item_ids": [ids[i] for i in group],
                              "options": options, "probabilities": probs, **detail})
        identity = self.identity.model_copy(update={"adapter": ADAPTER})
        return JudgeResponse(scores=tuple(scores), identity=identity, usage=call.usage,
                             metadata={**self._metadata(call), "role": request.role.value,
                                       "option_ids": ids, "score_scale": scale, "question": question,
                                       "objective_success": False, "thinking": "off",
                                       "decisions": decisions})
