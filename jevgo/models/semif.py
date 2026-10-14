"""Frozen SemIf direct scoring and generation through one local model instance.

Production construction uses the configured local Qwen3.5-4B checkpoint on CUDA.
CPU tests must inject a small ``model`` and ``tokenizer`` together; this module
never downloads weights or silently loads a large checkpoint onto CPU.

Multi-candidate requests use one direct categorical readout (2--16 options).
Single candidates and all evidence/state_value/stop items use independent binary
readouts. These conditional option scores are uncalibrated model opinions.
A direct readout generates no text tokens: its output_tokens cost is exactly zero.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import yaml
from semif_phase1 import core as semif_core, direct

from jevgo.core import (
    BudgetExceeded, BudgetLedger, GenerationRequest, GenerationResponse,
    JudgeRequest, JudgeResponse, JudgeRole, ModelFailure, ModelIdentity,
    UnsupportedCapability, Usage, UsagePhase,
)


@dataclass
class _Call:
    usage: Usage
    requests: list[dict] = field(default_factory=list)


class _GenerationProgress:
    """Track emitted tokens even when generate raises before returning sequences."""
    def __init__(self, prompt_ids: list[int]) -> None:
        self.prompt_ids = prompt_ids
        self.seen_prompt = False
        self.output_tokens = 0
        self.complete = False

    def put(self, value: Any) -> None:
        tokens = value.detach().cpu().reshape(-1).tolist()
        if not self.seen_prompt:
            if tokens != self.prompt_ids:
                raise ValueError("Generation streamer did not receive the expected prompt")
            self.seen_prompt = True
        else:
            self.output_tokens += len(tokens)

    def end(self) -> None:
        pass


class SemIfBackend:
    """GenerationBackend and JudgeBackend backed by the same frozen Torch model.

    ``model_path`` must resolve to the path in assets_root/configs/jevgo.yaml;
    pretrained loading is local-only and cannot change the configured identity. ``model_metadata`` describes injected model
    provenance (source/revision), otherwise it is explicitly marked as injected.
    Request options are limited to generation ``enable_thinking`` and judging
    ``option_ids``. Input token counts are exact; overlong prompts are rejected.
    """
    def __init__(self, *, assets_root: str | Path, model_path: str | Path | None = None,
                 device: str = "cuda", dtype: str = "bfloat16", max_model_len: int = 4096,
                 model: Any = None, tokenizer: Any = None,
                 model_metadata: dict | None = None) -> None:
        if type(max_model_len) is not int or max_model_len < 1:
            raise ValueError("max_model_len must be a positive integer")
        if (model is None) != (tokenizer is None):
            raise ValueError("Inject model and tokenizer together")
        root = Path(assets_root).expanduser().resolve()
        with (root / "configs/jevgo.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)["backbones"]["semif"]
        configured_source = (root / config["local_directory"]).resolve()
        source = (root / (config["local_directory"] if model_path is None else model_path)).resolve()
        injected = model is not None
        metadata = dict(model_metadata or {})
        if not injected:
            if source != configured_source:
                raise ValueError("model_path must resolve to the configured SemIf checkpoint directory")
            if not source.is_dir() or not (source / "config.json").is_file():
                raise FileNotFoundError(f"Local SemIf checkpoint missing: {source}")
            if device not in ("auto", "cuda"):
                raise UnsupportedCapability("Pretrained SemIf loading requires CUDA; inject tiny models for CPU tests")
            target = semif_core.resolve_device(device)
            if target.type != "cuda":
                raise UnsupportedCapability("Pretrained SemIf loading requires CUDA")
            model, tokenizer, metadata = semif_core.load_causal_model(
                str(source), config["model_revision"], device="cuda", dtype=dtype,
            )
        self.model, self.tokenizer = model, tokenizer
        self.max_model_len = max_model_len
        self._closed = False
        self._freeze()
        self.device = next(self.model.parameters()).device
        if self.device.type not in ("cpu", "cuda"):
            raise UnsupportedCapability("SemIfBackend supports CPU fixtures and CUDA inference")
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "get_text_config"):
            model_config = model_config.get_text_config()
        capacity = getattr(model_config, "max_position_embeddings", None)
        if isinstance(capacity, int) and max_model_len > capacity:
            raise ValueError("max_model_len exceeds the model's context capacity")
        # The pinned loader returns torch.__version__ as a str subclass.
        # Normalize dependency version strings at this boundary; keep strict
        # JSON validation for all other metadata and non-string values.
        for key in ("torch_version", "transformers_version"):
            if isinstance(metadata.get(key), str):
                metadata[key] = str(metadata[key])
        self._model_metadata = {
            **metadata, "injected_model": injected,
            "source": metadata.get("source", f"injected:{type(model).__name__}" if injected else str(source)),
            "semif_revision": config["revision"], "scorer": "semif_phase1.direct",
        }
        self.identity = ModelIdentity(
            backend="semif", model=self._model_metadata["source"] if injected else config["base_model"],
            revision=metadata.get("revision") if injected else config["model_revision"],
            metadata={**self._model_metadata, "mode": "frozen", "max_model_len": max_model_len,
                      "shared_generation_and_judge_instance": True, "calibrated": False},
        )

    def _freeze(self) -> None:
        self.model.eval()
        self.model.requires_grad_(False)

    def close(self) -> None:
        # Keep ownership with the caller for injected models. No global CUDA cache
        # clearing: other backends may share this process and its device.
        self._closed = True

    def _ready(self, request: GenerationRequest | JudgeRequest, allowed: set[str]) -> None:
        if self._closed:
            raise ModelFailure("SemIf backend is closed")
        if request.image_paths:
            raise UnsupportedCapability("SemIf direct scoring uses the local text tower only")
        extra = set(request.options) - allowed
        if extra:
            raise ValueError(f"Unsupported SemIf request options: {sorted(extra)}")
        self._freeze()

    @staticmethod
    def _ensure(ledger: BudgetLedger, usage: Usage, phase: UsagePhase) -> None:
        if ledger.over_budget:
            raise BudgetExceeded(ledger.over_budget, UsagePhase(phase))
        ledger.ensure_available(usage, phase=phase)

    def _invoke(self, operation: Callable, *, call: _Call, ledger: BudgetLedger,
                phase: UsagePhase, input_tokens: int, output_bound: int,
                source: str, progress: _GenerationProgress | None = None) -> Any:
        self._ensure(ledger, Usage(model_requests=1, input_tokens=input_tokens,
                                    output_tokens=output_bound), phase)
        fixed = Usage(model_requests=1, input_tokens=input_tokens)
        ledger.charge(fixed, source=source, phase=phase,
                      metadata={"attempt": len(call.requests) + 1})
        call.usage = call.usage + fixed
        started, failed = time.monotonic(), True
        try:
            result = operation()
            failed = False
            return result
        except Exception as error:
            if isinstance(error, ModelFailure):
                raise
            raise ModelFailure(f"{source} failed: {type(error).__name__}: {error}") from error
        finally:
            unknown = ["output_tokens"] if progress is not None and not progress.complete else []
            measured = Usage(output_tokens=progress.output_tokens if progress else 0,
                             wall_seconds=time.monotonic() - started)
            detail = {"source": source, "failed": failed, "input_tokens": input_tokens,
                      "output_tokens": measured.output_tokens, "unknown_usage": unknown,
                      "usage_complete": not unknown, "output_tokens_is_lower_bound": bool(unknown)}
            ledger.record(measured, source=source + ".result", phase=phase,
                          failed=failed, metadata=detail)
            call.usage = call.usage + measured
            call.requests.append(detail)

    @staticmethod
    def _usage_metadata(call: _Call) -> dict:
        return {"requests": call.requests, "num_model_requests": call.usage.model_requests,
                "usage_complete": all(item["usage_complete"] for item in call.requests),
                "unknown_usage": sorted({field for item in call.requests for field in item["unknown_usage"]})}

    def generate(self, request: GenerationRequest, *, ledger: BudgetLedger,
                 phase: UsagePhase = UsagePhase.SEARCH) -> GenerationResponse:
        import torch

        self._ready(request, {"enable_thinking"})
        thinking = request.options.get("enable_thinking", False)
        if type(thinking) is not bool:
            raise ValueError("enable_thinking must be a bool")
        messages = ([{"role": "system", "content": request.system}] if request.system else [])
        messages.append({"role": "user", "content": request.prompt})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking,
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) + request.max_new_tokens > self.max_model_len:
            raise ValueError("Generation prompt plus max_new_tokens exceeds max_model_len; no truncation allowed")
        self._ensure(ledger, Usage(generation_calls=1, model_requests=request.n,
                                   input_tokens=len(ids) * request.n,
                                   output_tokens=request.n * request.max_new_tokens), phase)
        fixed = Usage(generation_calls=1)
        ledger.charge(fixed, source="semif.generate", phase=phase)
        call, texts = _Call(fixed), []
        for index in range(request.n):
            progress = _GenerationProgress(ids)

            def operation() -> str:
                # One request per sample also supports n>1 with greedy decoding.
                inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
                mask = torch.ones_like(inputs)
                kwargs = {"max_new_tokens": request.max_new_tokens, "num_return_sequences": 1,
                          "do_sample": request.temperature > 0, "use_cache": True,
                          "return_dict_in_generate": True, "output_scores": False,
                          "streamer": progress}
                if request.temperature > 0:
                    kwargs["temperature"] = request.temperature
                if getattr(self.tokenizer, "pad_token_id", None) is not None:
                    kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                device_ids = [self.device.index or 0] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=device_ids), torch.inference_mode():
                    torch.random.default_generator.manual_seed(request.seed + index)
                    if device_ids:
                        torch.cuda.default_generators[device_ids[0]].manual_seed(request.seed + index)
                    output = self.model.generate(input_ids=inputs, attention_mask=mask, **kwargs)
                sequences = output.sequences
                if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
                    raise ValueError("Expected one generated sequence")
                tokens = sequences[0].detach().cpu().tolist()
                if len(tokens) < len(ids):
                    raise ValueError("Generated sequence is shorter than its prompt")
                actual = len(tokens) - len(ids)
                streamed = progress.output_tokens
                progress.output_tokens, progress.complete = actual, True
                if tokens[:len(ids)] != ids or actual > request.max_new_tokens:
                    raise ValueError("Generated sequence changed its prompt or exceeded the output limit")
                if progress.seen_prompt and streamed != actual:
                    raise ValueError("Streamed and returned output token counts differ")
                text = self.tokenizer.decode(tokens[len(ids):], skip_special_tokens=True)
                if not isinstance(text, str):
                    raise ValueError("Tokenizer did not return generation text")
                return text

            texts.append(self._invoke(operation, call=call, ledger=ledger, phase=phase,
                                      input_tokens=len(ids), output_bound=request.max_new_tokens,
                                      source="semif.model.generate", progress=progress))
        return GenerationResponse(texts=tuple(texts), identity=self.identity, usage=call.usage,
                                  metadata={**self._usage_metadata(call), "enable_thinking": thinking,
                                            "sample_seeds": [request.seed + i for i in range(request.n)],
                                            "scoring_path": None})

    @staticmethod
    def _validate_score(result: dict, row: dict, input_tokens: int) -> list[float]:
        expected = [option["id"] for option in row["options"]]
        if result["id"] != row["id"] or result["option_ids"] != expected:
            raise ValueError("SemIf changed option IDs or their order")
        if result["input_tokens"] != input_tokens:
            raise ValueError("SemIf input token count changed after preflight")
        probabilities, logits = result["probabilities"], result["option_logits"]
        if len(probabilities) != len(expected) or len(logits) != len(expected):
            raise ValueError("SemIf must return every option slot")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in [*probabilities, *logits]):
            raise ValueError("SemIf scores and logits must be finite")
        if any(value < 0 or value > 1 for value in probabilities) or not math.isclose(sum(probabilities), 1, abs_tol=1e-6):
            raise ValueError("Invalid conditional option distribution")
        return probabilities

    def score(self, request: JudgeRequest, *, ledger: BudgetLedger,
              phase: UsagePhase = UsagePhase.SEARCH) -> JudgeResponse:
        self._ready(request, {"option_ids"})
        raw_ids = request.options.get("option_ids", [f"item-{i}" for i in range(len(request.items))])
        if not isinstance(raw_ids, (tuple, list)):
            raise ValueError("option_ids must be an ordered array")
        ids = list(raw_ids)
        if (len(ids) != len(request.items) or any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("option_ids must be unique nonempty strings aligned with items")
        categorical = request.role == JudgeRole.CANDIDATE and len(request.items) > 1
        if categorical and len(request.items) > 16:
            raise ValueError("SemIf direct candidate scoring supports at most 16 options")
        questions = {
            JudgeRole.CANDIDATE: "Which candidate is the most reasonable next action for the stated goal?",
            JudgeRole.EVIDENCE: ("Is this evidence useful for resolving the question, including useful contradictions "
                                 "and failed experiments? Evaluate usefulness, not agreement with a belief."),
            JudgeRole.STATE_VALUE: ("Does this state have useful remaining potential for future progress? "
                                    "Exclude already observed rewards; give a heuristic opinion."),
            JudgeRole.STOP: ("Does this item support stopping and submitting the available result given the stated "
                             "criteria and evidence? This is advisory, not an objective success signal."),
        }
        question = questions[request.role]
        if request.role == JudgeRole.CANDIDATE and not categorical:
            question = "Is this candidate a reasonable next action for the stated goal?"
        rows = []
        if categorical:
            rows.append({"id": "candidate", "state": {"context": request.context}, "question": question,
                         "options": [{"id": identifier, "description": item}
                                     for identifier, item in zip(ids, request.items)]})
        else:
            for index, (identifier, item) in enumerate(zip(ids, request.items)):
                rows.append({"id": f"{request.role.value}:{index}",
                             "state": {"context": request.context, "item": {"id": identifier, "text": item}},
                             "question": question, "options": [{"id": "no", "description": "No"},
                                                                    {"id": "yes", "description": "Yes"}]})
        # Validate all lengths and exact single-token answer boundaries before any model call.
        prepared = [(row, len(direct.encode_prompt(self.tokenizer, row, self.max_model_len)[0])) for row in rows]
        self._ensure(ledger, Usage(judge_calls=1, model_requests=len(rows),
                                   input_tokens=sum(count for _, count in prepared)), phase)
        fixed = Usage(judge_calls=1)
        ledger.charge(fixed, source="semif.score", phase=phase)
        call, scores, decisions = _Call(fixed), [], []
        for row, input_tokens in prepared:
            def operation() -> tuple[list[float], dict]:
                result = direct.score(self.model, self.tokenizer, row, self._model_metadata,
                                      max_tokens=self.max_model_len)
                probabilities = self._validate_score(result, row, input_tokens)
                return probabilities, {
                    "option_ids": result["option_ids"], "option_logits": result["option_logits"],
                    "probabilities": probabilities, "prompt_sha256": result["prompt_sha256"],
                    "prompt_version": result["prompt_version"], "readout": result["readout"],
                    "probability_status": result["probability_status"],
                }

            probabilities, detail = self._invoke(operation, call=call, ledger=ledger, phase=phase,
                                                 input_tokens=input_tokens, output_bound=0,
                                                 source="semif.direct")
            scores.extend(probabilities if categorical else [probabilities[1]])
            detail["item_ids"] = ids if categorical else [ids[len(decisions)]]
            decisions.append(detail)
        scale = "conditional_candidate_preference" if categorical else "conditional_yes_score_for_role_question"
        return JudgeResponse(scores=tuple(scores), identity=self.identity, usage=call.usage,
                             metadata={**self._usage_metadata(call), "role": request.role.value,
                                       "option_ids": ids, "score_scale": scale, "question": question,
                                       "state_value_is_binary_heuristic": request.role == JudgeRole.STATE_VALUE,
                                       "probability_status": "uncalibrated", "calibrated": False,
                                       "objective_success": False, "scorer": "semif_phase1.direct",
                                       "decisions": decisions})
