"""Configuration assembly and reproducible model-protocol checks."""
from __future__ import annotations

from collections.abc import Mapping
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from typing import Any

from jevgo import __version__
from jevgo.core import (
    BudgetLedger, EpisodeSpec, GenerationRequest, JudgeRequest, RunConfig,
    UnsupportedCapability, UsagePhase,
)
from jevgo.models.deterministic import DeterministicBackend, FIXTURE_TASK

MODEL_KEYS = ("deterministic", "jev_9b", "jev_27b", "jev_27b_vl", "semif")
ENVIRONMENTS = {
    "minigrid": ("jevgo.envs.minigrid", "MiniGridAdapter"),
    "alfworld": ("jevgo.envs.alfworld", "ALFWorldAdapter"),
    "scienceworld": ("jevgo.envs.scienceworld", "ScienceWorldAdapter"),
    "discoveryworld": ("jevgo.envs.discoveryworld", "DiscoveryWorldAdapter"),
}


def _new_output(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    return output


def _model_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping):
        raise ValueError("model must be an object")
    config = dict(config)
    key = config.get("backend")
    if key not in MODEL_KEYS:
        raise ValueError(f"model.backend must be one of {MODEL_KEYS}")
    allowed = {"backend"}
    if key == "deterministic":
        allowed |= {"scripted_actions"}
    elif key == "semif":
        allowed |= {"model_path", "device", "dtype", "max_model_len"}
    else:
        allowed |= {"base_url", "timeout", "max_model_len"}
    if set(config) - allowed:
        raise ValueError(f"Unsupported {key} model fields: {sorted(set(config) - allowed)}")
    json.dumps(config, allow_nan=False)
    return config


def build_backend(model_config: Mapping, *, assets_root: str | Path,
                  episode: EpisodeSpec | None = None):
    config = _model_config(model_config)
    key = config.pop("backend")
    if key == "deterministic":
        if episode is not None and (episode.environment.lower() != "minigrid"
                or episode.task != FIXTURE_TASK or episode.options.get("env_id", episode.task) != FIXTURE_TASK):
            raise UnsupportedCapability("The deterministic backend supports only the declared MiniGrid CPU fixture")
        return DeterministicBackend(task=episode.task if episode else FIXTURE_TASK, **config)
    if key == "semif":
        from jevgo.models.semif import SemIfBackend
        return SemIfBackend(assets_root=assets_root, **config)
    from jevgo.models.autotrust import AutoTrustBackend
    return AutoTrustBackend(key, assets_root=assets_root, **config)


def build_environment(episode: EpisodeSpec, run_config: RunConfig, *, assets_root: str | Path,
                      output_dir: str | Path):
    name = episode.environment.lower()
    if name not in ENVIRONMENTS:
        raise ValueError(f"Unsupported environment: {episode.environment}")
    module_name, class_name = ENVIRONMENTS[name]
    adapter = getattr(importlib.import_module(module_name), class_name)
    if name == "minigrid":
        return adapter(env_id=episode.options.get("env_id", episode.task), max_steps=run_config.max_steps)
    kwargs = {"assets_root": assets_root, "max_steps": run_config.max_steps}
    if name == "discoveryworld":
        kwargs["output_dir"] = Path(output_dir) / "environment"
    return adapter(**kwargs)


def run_from_dict(payload: Mapping, *, assets_root: str | Path, output_dir: str | Path):
    if not isinstance(payload, Mapping) or set(payload) - {"episode", "run", "model"}:
        raise ValueError("Run configuration requires only episode, run and model objects")
    episode = EpisodeSpec.model_validate(payload["episode"])
    config = RunConfig.model_validate(payload.get("run", {}))
    model_config = _model_config(payload["model"])
    modality = episode.options.get("modality", "text")
    if modality not in ("text", "vl"):
        raise ValueError("Episode modality must be text or vl")
    if modality == "vl" and model_config["backend"] != "jev_27b_vl":
        raise UnsupportedCapability("VL episodes require the configured VL model")
    output = _new_output(output_dir)
    from jevgo.runner import run_episode
    environment = build_environment(episode, config, assets_root=assets_root, output_dir=output)
    try:
        backend = build_backend(model_config, assets_root=assets_root, episode=episode)
    except BaseException:
        environment.close()
        raise
    metadata = {"model_identity": backend.identity.model_dump(mode="json"),
                "model_key": model_config["backend"], "model_config": model_config,
                "modality": modality, "validation_level": (
                    "cpu_fixture" if model_config["backend"] == "deterministic" else "real_model_attempt")}
    # Ownership transfers here. The runner always closes adapters and deduplicates
    # the same object used as both generator and judge.
    return run_episode(episode, config, environment, backend, backend,
                       output_dir=output, metadata=metadata)


def versions() -> dict:
    packages = {}
    for name in ("pydantic", "httpx", "torch", "transformers", "semif-phase1", "vllm", "minigrid"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"jevgo": __version__, "python": sys.version, "platform": platform.platform(), "packages": packages}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_model(model_config: Mapping, *, assets_root: str | Path, output_dir: str | Path,
                image_path: str | Path | None = None) -> dict:
    """Actually call generation and three judge roles, retaining failure artifacts.

    Passing this check demonstrates those protocol calls only; it is not a
    benchmark or a statement about model effectiveness.
    """
    config = _model_config(model_config)
    output = _new_output(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    image_paths = (str(Path(image_path).expanduser().resolve()),) if image_path is not None else ()
    state = {"task": FIXTURE_TASK, "actions": [], "observation": {"text": "A visible empty cell ahead."}}
    context = "Protocol check using synthetic public observations.\nPublic state:\n" + json.dumps(state)
    generation = GenerationRequest(prompt="Propose an action as JSON.\n" + context, max_new_tokens=128,
                                   temperature=0, image_paths=image_paths)
    requests = [generation,
                JudgeRequest(role="candidate", context=context, image_paths=image_paths,
                             items=('{"action":{"name":"left"}}', '{"action":{"name":"forward"}}')),
                JudgeRequest(role="evidence", context=context, image_paths=image_paths,
                             items=("The visible cell ahead is empty.",)),
                JudgeRequest(role="stop", context=context, image_paths=image_paths,
                             items=("The available evidence supports submitting a result.",))]
    _write_json(output / "config.json", {"model": config, "image_path": image_paths[0] if image_paths else None})
    _write_json(output / "versions.json", versions())
    _write_json(output / "requests.json", [request.model_dump(mode="json") for request in requests])
    (output / "responses.jsonl").touch(exist_ok=False)
    ledger, backend, completed, error = BudgetLedger(), None, 0, None
    identity = None
    try:
        backend = build_backend(config, assets_root=assets_root)
        identity = backend.identity.model_dump(mode="json")
        with (output / "responses.jsonl").open("a", encoding="utf-8") as stream:
            for index, request in enumerate(requests):
                method = backend.generate if isinstance(request, GenerationRequest) else backend.score
                response = method(request, ledger=ledger, phase=UsagePhase.COMMIT)
                stream.write(json.dumps({"request_index": index, "response": response.model_dump(mode="json")},
                                        ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                completed += 1
    except Exception as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as caught:
                if error is None:
                    error = {"type": type(caught).__name__, "message": str(caught)}
        _write_json(output / "ledger.json", ledger.summary())
    result = {"ok": error is None, "completed_requests": completed, "error": error,
              "model_identity": identity, "model_key": config["backend"],
              "modality": "vl" if image_paths else "text",
              "validation_level": "cpu_fixture" if config["backend"] == "deterministic" else "real_model_attempt",
              "benchmark_effectiveness_verified": False, "usage": ledger.used.model_dump(mode="json"),
              "usage_complete": all(entry.metadata.get("usage_complete", True) for entry in ledger.entries)}
    _write_json(output / "result.json", result)
    return result
