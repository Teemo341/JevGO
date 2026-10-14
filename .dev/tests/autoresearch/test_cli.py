"""CPU fixture and mocked HTTP checks; no GPU services or pretrained weights."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import httpx
import pytest
import yaml

from jevgo import factory
from jevgo.__main__ import main
from jevgo.core import (
    BudgetExceeded, BudgetLedger, BudgetLimits, EpisodeSpec, GenerationRequest,
    JudgeRequest, ModelFailure, RunConfig, UnsupportedCapability, Usage,
)
from jevgo.models.deterministic import DeterministicBackend, FIXTURE_TASK

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def workspace():
    scratch = ROOT / ".dev/tmp"
    scratch.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="p07a-", dir=scratch))
    yield root
    shutil.rmtree(root)


def public_prompt(count=0, task=FIXTURE_TASK):
    return "fixture\nPublic state:\n" + json.dumps({"task": task,
        "actions": [{"name": "forward", "arguments": {}} for _ in range(count)],
        "observation": {"text": "visible state"}})


def config(method="B0"):
    payload = yaml.safe_load((ROOT / "configs/autoresearch-smoke.yaml").read_text())
    payload["run"]["method"] = method
    return payload


def test_fixture_script_only_reads_public_history_and_handles_repair():
    backend = DeterministicBackend()
    ledger = BudgetLedger()
    request = GenerationRequest(prompt=public_prompt(2) + "\nReturn only valid JSON.", n=4)
    result = backend.generate(request, ledger=ledger)
    assert [json.loads(text)["action"]["name"] for text in result.texts] == ["right", "left", "right", "forward"]
    judged = backend.score(JudgeRequest(role="candidate", context=public_prompt(2), items=result.texts), ledger=ledger)
    assert judged.scores == (1, 0, 1, 0)
    assert result.identity.backend == "deterministic_fixture"
    assert result.identity.metadata["benchmark_effectiveness_claim"] is False
    assert result.usage.input_tokens == result.usage.output_tokens == 0
    assert ledger.used.model_requests == 2
    assert ledger.used.generation_calls == ledger.used.judge_calls == 1


@pytest.mark.parametrize("role,expected", [("evidence", 1), ("stop", 0), ("state_value", 0)])
def test_fixture_non_candidate_roles_remain_opinions(role, expected):
    response = DeterministicBackend().score(JudgeRequest(role=role, context=public_prompt(), items=("a", "b")),
                                            ledger=BudgetLedger())
    assert response.scores == (expected, expected)
    assert response.metadata["objective_success"] is False
    assert response.metadata["role"] == role


def test_fixture_rejects_other_tasks_images_and_closed_calls():
    with pytest.raises(UnsupportedCapability):
        DeterministicBackend(task="ALFWorld")
    backend = DeterministicBackend()
    with pytest.raises(ModelFailure):
        backend.generate(GenerationRequest(prompt=public_prompt(task="other")), ledger=BudgetLedger())
    with pytest.raises(UnsupportedCapability):
        backend.generate(GenerationRequest(prompt=public_prompt(), image_paths=("image.png",)), ledger=BudgetLedger())
    backend.close()
    with pytest.raises(ModelFailure):
        backend.generate(GenerationRequest(prompt=public_prompt()), ledger=BudgetLedger())


def test_fixture_budget_and_failed_candidate_costs():
    backend = DeterministicBackend()
    ledger = BudgetLedger(BudgetLimits(wall_seconds=0))
    with pytest.raises(BudgetExceeded):
        backend.generate(GenerationRequest(prompt=public_prompt()), ledger=ledger)
    assert ledger.used == Usage()
    ledger = BudgetLedger()
    with pytest.raises(ModelFailure):
        backend.score(JudgeRequest(role="candidate", context=public_prompt(), items=("not JSON",)), ledger=ledger)
    assert ledger.used.judge_calls == ledger.used.model_requests == 1 and ledger.entries[-1].failed


def test_factory_rejects_fixture_mislabeling_and_bad_model_options(workspace):
    with pytest.raises(UnsupportedCapability):
        factory.build_backend({"backend": "deterministic"}, assets_root=workspace,
                              episode=EpisodeSpec(environment="scienceworld", task=FIXTURE_TASK))
    with pytest.raises(UnsupportedCapability):
        factory.build_backend({"backend": "deterministic"}, assets_root=workspace,
                              episode=EpisodeSpec(environment="minigrid", task=FIXTURE_TASK,
                                                  options={"env_id": "MiniGrid-Empty-8x8-v0"}))
    with pytest.raises(ValueError):
        factory.build_backend({"backend": "deterministic", "base_url": "http://example.invalid"}, assets_root=workspace)
    with pytest.raises(ValueError):
        factory.build_backend({"backend": "unknown"}, assets_root=workspace)


def test_environment_factory_is_lazy_and_discovery_subdir_does_not_create_run(workspace, monkeypatch):
    calls = []
    def adapter(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(factory.importlib, "import_module", lambda name: SimpleNamespace(
        DiscoveryWorldAdapter=adapter, ALFWorldAdapter=adapter, ScienceWorldAdapter=adapter, MiniGridAdapter=adapter))
    output = workspace / "new-run"
    factory.build_environment(EpisodeSpec(environment="discoveryworld", task="Tutorial"), RunConfig(max_steps=7),
                              assets_root=workspace, output_dir=output)
    assert calls[0]["output_dir"] == output / "environment"
    assert calls[0]["max_steps"] == 7 and not output.exists()


def test_factory_passes_metadata_ownership_and_new_output_to_runner(workspace, monkeypatch):
    import jevgo.runner as runner
    backend = DeterministicBackend()
    closed = []
    environment = SimpleNamespace(close=lambda: closed.append("environment"))
    monkeypatch.setattr(factory, "build_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(factory, "build_environment", lambda *args, **kwargs: environment)
    def run(episode, settings, env, generator, judge, *, output_dir, metadata):
        assert not output_dir.exists()
        assert generator is judge is backend and env is environment
        assert metadata["model_identity"]["backend"] == "deterministic_fixture"
        assert metadata["modality"] == "text" and metadata["validation_level"] == "cpu_fixture"
        return "runner-result"
    monkeypatch.setattr(runner, "run_episode", run)
    assert factory.run_from_dict(config(), assets_root=workspace, output_dir=workspace / "run") == "runner-result"
    assert not closed and not backend.closed


def test_factory_closes_environment_if_backend_construction_fails(workspace, monkeypatch):
    closed = []
    monkeypatch.setattr(factory, "build_environment", lambda *args, **kwargs: SimpleNamespace(close=lambda: closed.append(True)))
    def fail(*args, **kwargs):
        raise RuntimeError("construction failed")
    monkeypatch.setattr(factory, "build_backend", fail)
    with pytest.raises(RuntimeError):
        factory.run_from_dict(config(), assets_root=workspace, output_dir=workspace / "run")
    assert closed == [True] and not (workspace / "run").exists()


def test_existing_output_rejected_before_environment_or_model(workspace, monkeypatch):
    monkeypatch.setattr(factory, "build_environment", lambda *args, **kwargs: pytest.fail("must reject first"))
    with pytest.raises(FileExistsError):
        factory.run_from_dict(config(), assets_root=workspace, output_dir=workspace)
    with pytest.raises(FileExistsError):
        factory.check_model({"backend": "deterministic"}, assets_root=workspace, output_dir=workspace)


def test_model_check_fixture_records_all_protocols_and_costs(workspace):
    output = workspace / "model-check"
    result = factory.check_model({"backend": "deterministic"}, assets_root=workspace, output_dir=output)
    assert result["ok"] and result["completed_requests"] == 4
    assert result["validation_level"] == "cpu_fixture" and not result["benchmark_effectiveness_verified"]
    assert result["usage"]["generation_calls"] == 1 and result["usage"]["judge_calls"] == 3
    requests = json.loads((output / "requests.json").read_text())
    assert [request["role"] for request in requests[1:]] == ["candidate", "evidence", "stop"]
    assert len((output / "responses.jsonl").read_text().splitlines()) == 4
    assert all((output / name).is_file() for name in ("versions.json", "ledger.json", "result.json", "config.json"))


def test_model_check_constructor_failure_keeps_artifacts(workspace, monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("local checkpoint absent")
    monkeypatch.setattr(factory, "build_backend", fail)
    output = workspace / "missing"
    result = factory.check_model({"backend": "semif"}, assets_root=workspace, output_dir=output)
    assert not result["ok"] and result["completed_requests"] == 0
    assert result["validation_level"] == "real_model_attempt"
    assert (output / "responses.jsonl").read_text() == ""
    assert json.loads((output / "ledger.json").read_text())["used"]["model_requests"] == 0


def test_model_check_mock_http_records_real_backend_protocol_and_partial_failure(workspace, monkeypatch):
    from jevgo.models.autotrust import AutoTrustBackend
    config_dir = workspace / "configs"
    config_dir.mkdir()
    model = workspace / "models/JEV-27B"
    (model / "adapter_vllm").mkdir(parents=True)
    (config_dir / "jevgo.yaml").write_text(yaml.safe_dump({"backbones": {"jev_27b": {
        "repository": "autotrust/JEV-27B", "base_model": "fixture", "revision": "fixture",
        "local_directory": "models/JEV-27B"}}}))
    head = {"verbalizer_ids": list(range(24)), "bias": [0] * 24, "slots": {"num_slots": 24,
        "verbalizers": ["false", "true"] + list("012345ABCDEFGHIJKLMNOP"),
        "ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}, "template_version": "bare-v1"}}
    (model / "adapter_vllm/decision_head.json").write_text(json.dumps(head))
    (model / "calibration.json").write_text(json.dumps({"per_kind": {"noul": 1, "score": 1, "choice": 1}}))
    seen = []
    def handler(request):
        seen.append(request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "proposal"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3}})
        return httpx.Response(503, json={"error": "fixture failure", "usage": {"prompt_tokens": 4}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        backend = AutoTrustBackend("jev_27b", assets_root=workspace, client=client)
        monkeypatch.setattr(factory, "build_backend", lambda *args, **kwargs: backend)
        output = workspace / "http-check"
        result = factory.check_model({"backend": "jev_27b"}, assets_root=workspace, output_dir=output)
    assert len(seen) == 2 and result["completed_requests"] == 1 and not result["ok"]
    assert result["usage"]["model_requests"] == 2 and result["usage"]["input_tokens"] == 13
    assert result["usage"]["output_tokens"] == 3 and result["usage_complete"] is False
    assert len((output / "responses.jsonl").read_text().splitlines()) == 1
    assert backend._closed


def test_cli_config_overrides_and_invalid_yaml(workspace, monkeypatch, capsys):
    import jevgo.__main__ as cli
    captured = []
    def run(payload, **kwargs):
        captured.append((payload, kwargs))
        return SimpleNamespace(error=None, model_dump_json=lambda: '{"ok":true}')
    monkeypatch.setattr(cli, "run_from_dict", run)
    path = workspace / "config.yaml"
    path.write_text(yaml.safe_dump(config()))
    assert main(["run", "--config", str(path), "--output", str(workspace / "run"),
                 "--assets-root", str(workspace), "--method", "B2", "--model", "jev_27b", "--base-url", "http://local:8000"]) == 0
    payload = captured[0][0]
    assert payload["run"]["method"] == "B2"
    assert payload["model"] == {"backend": "jev_27b", "base_url": "http://local:8000"}
    path.write_text("- not\n- an object\n")
    assert main(["run", "--config", str(path), "--output", str(workspace / "run")]) == 2
    assert "YAML object" in capsys.readouterr().err


def test_model_check_cli_and_no_overwrite(workspace, capsys):
    args = ["model-check", "--model", "deterministic", "--assets-root", str(workspace), "--output", str(workspace / "check")]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["completed_requests"] == 4
    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize("method", ["B0", "B1", "B2", "B3"])
def test_real_minigrid_four_methods_through_module_cli(workspace, method):
    output = workspace / method
    command = [sys.executable, "-B", "-m", "jevgo", "run", "--config", str(ROOT / "configs/autoresearch-smoke.yaml"),
               "--output", str(output), "--assets-root", str(ROOT), "--method", method]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    result = json.loads(completed.stdout)
    assert result["termination_reason"] == "success" and result["final_state"]["success"] is True
    assert len(result["final_state"]["actions"]) == 5
    assert result["metadata"]["validation_level"] == "cpu_fixture"
    assert (output / "events.jsonl").is_file() and (output / "result.json").is_file()


def test_cli_malformed_yaml_returns_configuration_error(workspace, capsys):
    path = workspace / "broken.yaml"
    path.write_text("episode: [unterminated")
    assert main(["run", "--config", str(path), "--output", str(workspace / "run")]) == 2
    error = capsys.readouterr().err
    assert "Error" in error and "Traceback" not in error
    assert not (workspace / "run").exists()


def test_vl_label_requires_actual_environment_images(tmp_path):
    from jevgo.factory import run_from_dict
    from jevgo.core import UnsupportedCapability
    payload = {"episode": {"environment": "minigrid", "task": "MiniGrid-Empty-5x5-v0",
                           "options": {"modality": "vl"}},
               "model": {"backend": "jev_27b_vl"}}
    with pytest.raises(UnsupportedCapability, match="public VL images"):
        run_from_dict(payload, assets_root=tmp_path, output_dir=tmp_path / "must-not-run")
    assert not (tmp_path / "must-not-run").exists()
