"""CPU protocol + installed direct scorer tests; real checkpoint inference pending."""
import json
import math
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
import yaml
from semif_phase1 import direct

from jevgo.core import (
    BudgetExceeded, BudgetLedger, BudgetLimits, GenerationBackend, GenerationRequest,
    JudgeBackend, JudgeRequest, ModelFailure, UnsupportedCapability, Usage, UsagePhase,
)
from jevgo.models.semif import SemIfBackend


class ByteTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.messages = []
        self.fail_decode = False

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is False and kwargs["add_generation_prompt"] is True
        self.messages.append(messages)
        return "\n".join(message["content"] for message in messages) + "\nASSISTANT: "

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text.encode("utf-8"))

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens and self.fail_decode:
            raise ValueError("fixture decode failure")
        return "".join(chr(token) for token in ids if not skip_special_tokens or token not in (0, 1))


class FixedModel(torch.nn.Module):
    """Deterministic injected model, exercising the installed Torch direct scorer."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.forward_calls = 0
        self.generate_calls = 0
        self.fail_forward_at = None
        self.fail_generation = None

    def forward(self, input_ids, attention_mask, use_cache, return_dict, logits_to_keep):
        assert torch.is_inference_mode_enabled()
        assert self.training is False and self.anchor.requires_grad is False
        assert use_cache is False and return_dict is True and logits_to_keep == 1
        assert input_ids.shape == attention_mask.shape
        self.forward_calls += 1
        if self.forward_calls == self.fail_forward_at:
            raise RuntimeError("fixture forward failure")
        logits = torch.zeros((1, 1, 260))
        logits[0, 0, ord("A")] = 2
        logits[0, 0, ord("B")] = -1
        return SimpleNamespace(logits=logits)

    def generate(self, input_ids, attention_mask, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not self.training and not self.anchor.requires_grad
        assert kwargs["use_cache"] is True and kwargs["num_return_sequences"] == 1
        self.generate_calls += 1
        if self.fail_generation == "before":
            raise RuntimeError("fixture generate failure")
        kwargs["streamer"].put(input_ids)
        generated = torch.randint(65, 91, (2,)) if kwargs["do_sample"] else torch.tensor([65, 66])
        for value in generated:
            kwargs["streamer"].put(value.reshape(1))
        if self.fail_generation == "after_two":
            raise RuntimeError("fixture partial generation failure")
        kwargs["streamer"].end()
        return SimpleNamespace(sequences=torch.cat((input_ids, generated.reshape(1, -1)), dim=1))


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def assets():
    scratch = Path(__file__).resolve().parents[2] / "tmp"
    scratch.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="p04-", dir=scratch))
    (root / "configs").mkdir()
    directory = root / "models/Qwen3.5-4B"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    (root / "configs/jevgo.yaml").write_text(yaml.safe_dump({"backbones": {"semif": {
        "repository": "https://example.invalid/scorer", "revision": "scorer-revision",
        "model_revision": "model-revision", "base_model": "Qwen/Qwen3.5-4B",
        "local_directory": "models/Qwen3.5-4B",
    }}}))
    yield root
    shutil.rmtree(root)


@pytest.fixture
def backend(assets):
    return SemIfBackend(assets_root=assets, model=FixedModel(), tokenizer=ByteTokenizer())


def test_installed_direct_scorer_frozen_parameter_and_ids(backend):
    assert isinstance(backend, GenerationBackend) and isinstance(backend, JudgeBackend)
    assert backend.model.training is False
    assert all(not parameter.requires_grad for parameter in backend.model.parameters())
    before = backend.model.anchor.detach().clone()
    ledger = BudgetLedger()
    result = backend.score(JudgeRequest(role="candidate", context="goal", items=("action A", "action B"),
                                        options={"option_ids": ["second", "first"]}), ledger=ledger)
    assert backend.model.forward_calls == 1
    expected = 1 / (1 + math.exp(-3))
    assert result.scores == pytest.approx((expected, 1 - expected))
    assert result.metadata["option_ids"] == ("second", "first")
    detail = result.metadata["decisions"][0]
    assert detail["option_logits"] == (2, -1) and len(detail["prompt_sha256"]) == 64
    assert detail["prompt_version"] == direct.PROMPT_VERSION
    assert "uncalibrated" in detail["probability_status"]
    assert result.metadata["calibrated"] is False and result.metadata["objective_success"] is False
    assert result.identity.model == "injected:FixedModel" and result.identity.revision is None
    assert result.identity.metadata["semif_revision"] == "scorer-revision"
    assert torch.equal(backend.model.anchor, before)
    assert result.usage == ledger.used
    assert result.usage.judge_calls == result.usage.model_requests == 1
    assert result.usage.input_tokens > 0 and result.usage.output_tokens == 0


@pytest.mark.parametrize("count", [2, 16])
def test_candidate_boundaries(backend, count):
    result = backend.score(JudgeRequest(role="candidate", context="", items=tuple(str(i) for i in range(count))),
                           ledger=BudgetLedger())
    assert len(result.scores) == count and sum(result.scores) == pytest.approx(1)
    assert backend.model.forward_calls == 1


def test_more_than_16_candidates_rejected_before_model(backend):
    ledger = BudgetLedger()
    with pytest.raises(ValueError, match="16"):
        backend.score(JudgeRequest(role="candidate", context="x", items=tuple(str(i) for i in range(17))), ledger=ledger)
    assert backend.model.forward_calls == 0 and ledger.used == Usage()


@pytest.mark.parametrize("role,word", [("candidate", "reasonable"), ("evidence", "usefulness"),
                                       ("state_value", "future progress"), ("stop", "stopping")])
def test_single_item_roles_use_distinct_binary_semantics(backend, role, word):
    result = backend.score(JudgeRequest(role=role, context="goal", items=("item",)), ledger=BudgetLedger())
    assert result.scores == pytest.approx((1 / (1 + math.exp(3)),))
    assert result.metadata["role"] == role and word in result.metadata["question"]
    assert result.metadata["decisions"][0]["option_ids"] == ("no", "yes")
    assert result.metadata["state_value_is_binary_heuristic"] == (role == "state_value")
    assert result.metadata["probability_status"] == "uncalibrated"


def test_32_evidence_items_are_independent_and_all_billed(backend):
    ledger = BudgetLedger()
    items = tuple(f"evidence {index}" for index in range(32))
    result = backend.score(JudgeRequest(role="evidence", context="goal", items=items), ledger=ledger)
    assert len(result.scores) == len(result.metadata["decisions"]) == 32
    assert backend.model.forward_calls == result.usage.model_requests == 32
    assert result.usage.judge_calls == 1 and result.usage == ledger.used
    assert result.metadata["decisions"][31]["item_ids"] == ("item-31",)
    assert result.usage.output_tokens == 0 and result.metadata["usage_complete"] is True


def test_generation_reuses_frozen_model_and_preserves_rng(backend):
    ledger = BudgetLedger()
    original = torch.random.get_rng_state().clone()
    request = GenerationRequest(prompt="x", n=3, max_new_tokens=3, seed=70)
    result = backend.generate(request, ledger=ledger)
    assert torch.equal(torch.random.get_rng_state(), original)
    assert len(result.texts) == 3 and backend.model.generate_calls == 3
    assert result.usage.generation_calls == 1 and result.usage.model_requests == 3
    assert result.usage.output_tokens == 6 and result.usage == ledger.used
    assert result.metadata["sample_seeds"] == (70, 71, 72)
    repeated = backend.generate(request, ledger=BudgetLedger())
    assert repeated.texts == result.texts
    backend.score(JudgeRequest(role="stop", context="goal", items=("answer",)), ledger=BudgetLedger())
    assert backend.model.forward_calls == 1
    assert not backend.model.training and all(not p.requires_grad for p in backend.model.parameters())


def test_greedy_n_uses_same_model_without_beam_search(backend):
    result = backend.generate(GenerationRequest(prompt="x", n=2, max_new_tokens=3, temperature=0), ledger=BudgetLedger())
    assert result.texts == ("AB", "AB")
    assert backend.model.generate_calls == 2 and result.usage.output_tokens == 4


@pytest.mark.parametrize("role", ["candidate", "evidence"])
def test_long_judge_prompt_rejected_without_truncation(backend, role):
    ledger = BudgetLedger()
    with pytest.raises(ValueError, match="no truncation"):
        backend.score(JudgeRequest(role=role, context="x" * 5000, items=("a", "b")), ledger=ledger)
    assert backend.model.forward_calls == 0 and ledger.used == Usage()


def test_later_long_evidence_item_prevents_all_model_work(backend):
    ledger = BudgetLedger()
    with pytest.raises(ValueError, match="no truncation"):
        backend.score(JudgeRequest(role="evidence", context="goal", items=("short", "x" * 5000)), ledger=ledger)
    assert backend.model.forward_calls == 0 and ledger.used == Usage()


def test_generation_context_includes_output_limit(backend):
    ledger = BudgetLedger()
    with pytest.raises(ValueError, match="no truncation"):
        backend.generate(GenerationRequest(prompt="x", max_new_tokens=4096), ledger=ledger)
    assert backend.model.generate_calls == 0 and ledger.used == Usage()


@pytest.mark.parametrize("resource", ["judge_calls", "model_requests", "input_tokens"])
def test_score_budget_preflight(backend, resource):
    ledger = BudgetLedger(BudgetLimits(**{resource: 0}))
    with pytest.raises(BudgetExceeded):
        backend.score(JudgeRequest(role="candidate", context="goal", items=("a", "b")), ledger=ledger)
    assert backend.model.forward_calls == 0 and ledger.used == Usage()


def test_score_generates_no_output_tokens(backend):
    result = backend.score(JudgeRequest(role="candidate", context="goal", items=("a", "b")),
                           ledger=BudgetLedger(BudgetLimits(output_tokens=0)))
    assert result.usage.output_tokens == 0 and result.metadata["unknown_usage"] == ()


def test_generation_budget_reserves_all_samples(backend):
    ledger = BudgetLedger(BudgetLimits(model_requests=1))
    with pytest.raises(BudgetExceeded):
        backend.generate(GenerationRequest(prompt="x", max_new_tokens=3, n=2), ledger=ledger)
    assert backend.model.generate_calls == 0 and ledger.used == Usage()


def test_commit_reserve_and_phase(backend):
    ledger = BudgetLedger(BudgetLimits(model_requests=1), commit_reserve=Usage(model_requests=1))
    request = JudgeRequest(role="stop", context="goal", items=("answer",))
    with pytest.raises(BudgetExceeded):
        backend.score(request, ledger=ledger)
    backend.score(request, ledger=ledger, phase=UsagePhase.COMMIT)
    assert ledger.used.model_requests == 1 and all(entry.phase == UsagePhase.COMMIT for entry in ledger.entries)


def test_existing_measured_overage_blocks_next_request(backend):
    ledger = BudgetLedger(BudgetLimits(wall_seconds=1))
    ledger.record(Usage(wall_seconds=2), source="prior")
    with pytest.raises(BudgetExceeded):
        backend.generate(GenerationRequest(prompt="x", max_new_tokens=3), ledger=ledger)
    assert ledger.used.model_requests == 0


def test_failed_second_judge_keeps_all_attempts_and_known_inputs(backend):
    backend.model.fail_forward_at = 2
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure, match="fixture forward failure"):
        backend.score(JudgeRequest(role="evidence", context="goal", items=("a", "b", "c")), ledger=ledger)
    assert backend.model.forward_calls == ledger.used.model_requests == 2
    assert ledger.used.judge_calls == 1 and ledger.used.input_tokens > 0 and ledger.used.output_tokens == 0
    assert ledger.entries[-1].failed and ledger.entries[-1].metadata["usage_complete"] is True


@pytest.mark.parametrize("failure,known_outputs", [("before", 0), ("after_two", 2)])
def test_failed_generation_keeps_partial_tokens_as_lower_bound(backend, failure, known_outputs):
    backend.model.fail_generation = failure
    ledger = BudgetLedger()
    original = torch.random.get_rng_state().clone()
    with pytest.raises(ModelFailure):
        backend.generate(GenerationRequest(prompt="x", n=3, max_new_tokens=3), ledger=ledger)
    assert torch.equal(torch.random.get_rng_state(), original)
    assert backend.model.generate_calls == ledger.used.model_requests == 1
    assert ledger.used.generation_calls == 1 and ledger.used.input_tokens > 0
    assert ledger.used.output_tokens == known_outputs
    assert ledger.entries[-1].failed
    assert ledger.entries[-1].metadata["unknown_usage"] == ("output_tokens",)
    assert ledger.entries[-1].metadata["output_tokens_is_lower_bound"] is True


def test_decode_failure_retains_complete_generated_token_cost(backend):
    backend.tokenizer.fail_decode = True
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure, match="decode failure"):
        backend.generate(GenerationRequest(prompt="x", max_new_tokens=3), ledger=ledger)
    assert ledger.used.output_tokens == 2 and ledger.entries[-1].failed
    assert ledger.entries[-1].metadata["usage_complete"] is True


@pytest.mark.parametrize("corruption", ["nan", "missing_slot", "ids", "mass", "tokens"])
def test_invalid_direct_response_not_accepted_and_cost_retained(backend, monkeypatch, corruption):
    original = direct.score
    def score(*args, **kwargs):
        result = original(*args, **kwargs)
        if corruption == "nan":
            result["option_logits"][0] = float("nan")
        elif corruption == "missing_slot":
            result["probabilities"].pop()
        elif corruption == "ids":
            result["option_ids"].reverse()
        elif corruption == "mass":
            result["probabilities"] = [0.9, 0.9]
        else:
            result["input_tokens"] += 1
        return result
    monkeypatch.setattr(direct, "score", score)
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure):
        backend.score(JudgeRequest(role="candidate", context="goal", items=("a", "b")), ledger=ledger)
    assert ledger.used.model_requests == 1 and ledger.used.input_tokens > 0 and ledger.entries[-1].failed


@pytest.mark.parametrize("ids", [["same", "same"], ["a"], "ab", ["a", ""]])
def test_invalid_option_ids_rejected(backend, ids):
    with pytest.raises(ValueError):
        backend.score(JudgeRequest(role="candidate", context="goal", items=("a", "b"), options={"option_ids": ids}),
                      ledger=BudgetLedger())
    assert backend.model.forward_calls == 0


def test_images_rejected_before_model(backend):
    ledger = BudgetLedger()
    with pytest.raises(UnsupportedCapability):
        backend.generate(GenerationRequest(prompt="x", image_paths=("missing.png",)), ledger=ledger)
    with pytest.raises(UnsupportedCapability):
        backend.score(JudgeRequest(role="stop", context="goal", items=("a",), image_paths=("missing.png",)), ledger=ledger)
    assert ledger.used == Usage()


def test_only_local_cuda_pretrained_loading_and_config_revision(assets, monkeypatch):
    calls = []
    def load(source, revision, **kwargs):
        assert Path(source).is_dir()
        calls.append((source, revision, kwargs))
        return FixedModel(), ByteTokenizer(), {"source": source, "revision": revision}
    monkeypatch.setattr("jevgo.models.semif.semif_core.resolve_device", lambda device: torch.device("cuda:0"))
    monkeypatch.setattr("jevgo.models.semif.semif_core.load_causal_model", load)
    backend = SemIfBackend(assets_root=assets)
    assert calls == [(str(assets / "models/Qwen3.5-4B"), "model-revision", {"device": "cuda", "dtype": "bfloat16"})]
    assert backend.identity.model == "Qwen/Qwen3.5-4B" and backend.identity.revision == "model-revision"
    assert backend.identity.metadata["injected_model"] is False
    with pytest.raises(UnsupportedCapability, match="inject tiny"):
        SemIfBackend(assets_root=assets, device="cpu")
    with pytest.raises(FileNotFoundError):
        SemIfBackend(assets_root=assets, model_path="remote/model-that-is-not-local")
    assert len(calls) == 1


def test_injection_must_be_complete_and_close_disables_calls(assets, backend):
    with pytest.raises(ValueError, match="together"):
        SemIfBackend(assets_root=assets, model=FixedModel())
    backend.close()
    with pytest.raises(ModelFailure, match="closed"):
        backend.generate(GenerationRequest(prompt="x"), ledger=BudgetLedger())


def test_random_tiny_qwen_real_direct_and_generation_same_instance(assets):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=260, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=4096, pad_token_id=0, eos_token_id=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1]},
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        model = Qwen3_5ForCausalLM(config)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    backend = SemIfBackend(assets_root=assets, model=model, tokenizer=ByteTokenizer(),
                           model_metadata={"source": "random-tiny-qwen-fixture", "revision": "fixture-v1"})
    ledger = BudgetLedger()
    scores = backend.score(JudgeRequest(role="candidate", context="s", items=("a", "b")), ledger=ledger)
    generated = backend.generate(GenerationRequest(prompt="x", max_new_tokens=2, temperature=0), ledger=ledger)
    assert backend.model is model
    assert len(scores.scores) == 2 and all(math.isfinite(value) for value in scores.scores)
    assert len(generated.texts) == 1 and 1 <= generated.usage.output_tokens <= 2
    assert ledger.used.model_requests == 2 and ledger.used.judge_calls == ledger.used.generation_calls == 1
    assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    assert generated.identity.model == "random-tiny-qwen-fixture"


def test_zero_wall_time_prevents_all_model_calls(backend):
    ledger = BudgetLedger(BudgetLimits(wall_seconds=0))
    with pytest.raises(BudgetExceeded):
        backend.score(JudgeRequest(role="candidate", context="goal", items=("a", "b")), ledger=ledger)
    with pytest.raises(BudgetExceeded):
        backend.generate(GenerationRequest(prompt="x", max_new_tokens=3), ledger=ledger)
    assert backend.model.forward_calls == backend.model.generate_calls == 0
    assert ledger.used == Usage()
