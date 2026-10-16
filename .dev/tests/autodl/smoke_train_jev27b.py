#!/usr/bin/env python3
"""Manual, offline GPU acceptance: a new LoRA on a JEV-9B/27B BF16 backbone.

The released System 1 adapter/head are neither loaded nor modified. This checks
real optimizer steps and a checkpoint round trip, not convergence or JEV quality.
--output must already be a container symlink to an isolated NAS runs directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
from importlib.metadata import PackageNotFoundError, version
import inspect
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
GIB = 1024**3
TARGETS = ["q_proj", "v_proj", "in_proj_qkv"]
SHARED_MODELS = Path("/root/autodl-fs/JevGO/models")
FAST_STORAGE = Path("/root/autodl-tmp/jevgo")
MODEL_PROFILES = {
    "jev_27b": {"directory": "JEV-27B", "hidden_size": 5120, "num_hidden_layers": 64,
                "min_total_gib": 75, "min_free_gib": 70,
                "capacity_basis": "Conservative gate retained from the validated 80/96 GB acceptance"},
    "jev_9b": {"directory": "JEV-9B", "hidden_size": 4096, "num_hidden_layers": 32,
               "min_total_gib": 24, "min_free_gib": 22,
               "capacity_basis": "Estimated gate from 16.68 GiB BF16 weights, not a measured minimum; use 48 GB for the first acceptance"},
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    require(not temporary.is_symlink(), f"Refusing report symlink: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def report_path(path):
    path = path.absolute()
    require(".." not in path.parts, "--report must not contain parent-directory traversal")
    artifacts = ROOT / ".dev/artifacts"
    if path.is_relative_to(artifacts):
        require(path.parent.is_relative_to(artifacts), "--report must be a file within .dev/artifacts")
        # Small reports stay in real project directories, never in an output
        # symlink or a mounted NAS directory. Check before creating any parent.
        cursor = ROOT
        device = ROOT.stat().st_dev
        for part in path.parent.relative_to(ROOT).parts:
            cursor = cursor / part
            require(not cursor.is_symlink(), f"Report parent must not be a symlink: {cursor}")
            if cursor.exists():
                require(cursor.is_dir() and cursor.stat().st_dev == device,
                        f"Report parent must remain on the project filesystem: {cursor}")
    else:
        fast = FAST_STORAGE.resolve(strict=True)
        require(path.parent.resolve().is_relative_to(fast),
                "--report must be under project .dev/artifacts or AutoDL fast storage")
    require(not path.exists() and not path.is_symlink(), "--report must be a new file")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def model_directory(profile, directory):
    root = directory.resolve(strict=True)
    require(root == (SHARED_MODELS / profile["directory"]).resolve(strict=True),
            f"Use the shared, original {profile['directory']} directory")
    return root


def check_capacity(profile, total, free):
    require(total >= profile["min_total_gib"] * GIB and free >= profile["min_free_gib"] * GIB,
            f"{profile['directory']} needs at least {profile['min_total_gib']} GiB total and "
            f"{profile['min_free_gib']} GiB free; have {total/GIB:.2f} GiB total, {free/GIB:.2f} GiB free. "
            f"{profile['capacity_basis']}")


def check_architecture(profile, config):
    require(config.model_type == "qwen3_5_text" and config.hidden_size == profile["hidden_size"]
            and config.num_hidden_layers == profile["num_hidden_layers"],
            f"Unexpected {profile['directory']} architecture; refusing a smaller/replacement model")


def output_directory(output):
    nas = Path("/root/autodl-fs").resolve(strict=True)
    require(nas.is_mount(), "NAS is not mounted; refusing root-disk fallback")
    runs = (nas / "JevGO/runs").resolve(strict=True)
    require(runs.is_relative_to(nas), "NAS runs directory points outside NAS")
    require(output.is_symlink(), "--output must be a pre-created container symlink")
    target = output.resolve(strict=True)
    require(target.is_dir() and target != runs and target.is_relative_to(runs),
            "--output must point to an isolated directory under /root/autodl-fs/JevGO/runs")
    models = (nas / "JevGO/models").resolve(strict=True)
    require(not target.is_relative_to(models), "Training output must never be inside shared models")
    require(not os.path.lexists(target / "checkpoint-final"), "Checkpoint already exists; choose a new run")
    return target


def file_stats(root):
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Unexpected model-file symlink: {path}")
        if path.is_file():
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = [
                stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev]
    return result


class FastKernelProbe:
    """Count calls to the providers already selected by Transformers 5.17.

    The fallback decorator captures its chosen callable in ``implementation``.
    Wrapping that cell observes actual calls, including checkpoint recomputation,
    without changing provider selection or the decorator's argument filtering.
    """

    def __init__(self, qwen_module=None):
        import torch
        if qwen_module is None:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen_module
        self._cells = []
        self.providers = {}
        try:
            for symbol, package in (("causal_conv1d_fn", "causal_conv1d"),
                                    ("torch_chunk_gated_delta_rule", "fla")):
                wrapped = getattr(qwen_module, symbol)
                seen = set()
                while wrapped is not None and id(wrapped) not in seen:
                    seen.add(id(wrapped))
                    code = getattr(wrapped, "__code__", None)
                    cells = dict(zip(code.co_freevars, wrapped.__closure__ or ())) if code else {}
                    if "implementation" in cells:
                        break
                    wrapped = getattr(wrapped, "__wrapped__", None)
                else:
                    raise RuntimeError(f"Cannot inspect Transformers' resolved kernel for {symbol}")
                cell = cells["implementation"]
                implementation = cell.cell_contents
                module = implementation.__module__
                require(module == package or module.startswith(package + "."),
                        f"{symbol} resolved to {module}, expected an installed {package} kernel")
                require(cells.get("is_new_implementation") is not None
                        and cells["is_new_implementation"].cell_contents is True,
                        f"{symbol} is using the reference implementation")
                record = {"module": module, "qualname": implementation.__qualname__,
                          "source_file": inspect.getsourcefile(inspect.unwrap(implementation)),
                          "calls": 0, "completed_calls": 0, "grad_enabled_calls": 0}
                self.providers[symbol] = record

                def counted(*args, _implementation=implementation, _record=record, **kwargs):
                    _record["calls"] += 1
                    _record["grad_enabled_calls"] += int(torch.is_grad_enabled())
                    result = _implementation(*args, **kwargs)
                    _record["completed_calls"] += 1
                    return result

                cell.cell_contents = functools.wraps(implementation)(counted)
                self._cells.append((cell, implementation))
        except Exception:
            self.close()
            raise

    def assert_used(self):
        require(all(item["completed_calls"] > 0 and item["grad_enabled_calls"] > 0
                    for item in self.providers.values()),
                f"Both optimized providers must execute during training: {self.providers}")

    def snapshot(self):
        return {"observation": "wrapped already selected Transformers provider; selection unchanged",
                "providers": {key: dict(value) for key, value in self.providers.items()}}

    def close(self):
        for cell, original in reversed(self._cells):
            cell.cell_contents = original
        self._cells.clear()


def train_and_reload(base, batches, checkpoint, *, steps=3, rank=8, learning_rate=1e-4,
                     on_step=None, before_save=None):
    """The real train/save/reload path, also exercised with a tiny Qwen CPU model."""
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    require(not os.path.lexists(checkpoint), "Refusing to overwrite an existing checkpoint")
    counts = {target: sum(name.endswith("." + target) for name, _ in base.named_modules())
              for target in TARGETS}
    require(all(counts.values()), f"Expected Qwen full/linear-attention projections are absent: {counts}")
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(base, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=TARGETS,
        task_type=TaskType.CAUSAL_LM, bias="none"))
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    require(trainable and all("lora_" in name for name in trainable), "Only the new LoRA may be trainable")
    require(all(param.dtype == torch.float32 for param in trainable.values()),
            "Trainable LoRA parameters must remain FP32")
    initial = {name: param.detach().cpu().clone() for name, param in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate, weight_decay=0.0, foreach=False)
    history = []
    model.train()
    device = next(iter(trainable.values())).device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for step in range(steps):
        synchronize()
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batches[step % len(batches)], use_cache=False).loss
        require(bool(torch.isfinite(loss).item()), f"Non-finite loss at step {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
        require(float(norm) > 0, f"No gradient at step {step + 1}")
        require(all(param.grad is None for param in model.parameters() if not param.requires_grad),
                "Frozen backbone unexpectedly received gradients")
        optimizer.step()
        synchronize()
        record = {"step": step + 1, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                  "step_seconds": time.perf_counter() - step_started}
        history.append(record)
        if on_step:
            on_step(record)
    changed = [name for name, param in trainable.items()
               if not torch.equal(initial[name], param.detach().cpu())]
    require(changed, "Optimizer did not change any trainable tensor")
    require(all(bool(torch.isfinite(param).all().item()) for param in trainable.values()),
            "Optimizer produced non-finite adapter parameters")

    def evaluate(candidate):
        candidate.eval()
        batch = batches[0]
        inputs = {key: value for key, value in batch.items() if key != "labels"}
        with torch.inference_mode():
            logits = candidate(**inputs, use_cache=False).logits
            return logits.float().cpu()

    reference = evaluate(model)
    require(bool(torch.isfinite(reference).all()), "Non-finite evaluation output")
    if before_save:
        before_save()
    checkpoint.mkdir()
    adapter = checkpoint / "adapter"
    model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    torch.save({"optimizer": optimizer.state_dict(), "step": steps,
                "torch_rng_state": torch.get_rng_state(), "learning_rate": learning_rate},
               checkpoint / "optimizer.pt")
    trainable_parameters = sum(param.numel() for param in trainable.values())
    # unload() removes the new adapter without merging it into the frozen backbone.
    clean_base = model.unload()
    del optimizer, model, trainable, initial
    reloaded = PeftModel.from_pretrained(clean_base, adapter, is_trainable=True, local_files_only=True)
    require(all(param.dtype == torch.float32 for param in reloaded.parameters() if param.requires_grad),
            "Reloaded trainable LoRA parameters must remain FP32")
    actual = evaluate(reloaded)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    restored = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
    restored_optimizer = torch.optim.AdamW(
        [p for p in reloaded.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.0, foreach=False)
    restored_optimizer.load_state_dict(restored["optimizer"])
    require(restored["step"] == steps and restored_optimizer.state, "Optimizer checkpoint is incomplete")
    require(all(int(state["step"]) == steps for state in restored_optimizer.state.values()),
            "Restored optimizer step counters differ")
    return {"losses": history, "target_module_counts": counts,
            "trainable_dtype": "float32", "reloaded_trainable_dtype": "float32",
            "step_timing": {"cuda_synchronized": device.type == "cuda",
                            "includes": "zero_grad, forward, backward, gradient checks, optimizer step; first step may include JIT compilation",
                            "excludes": "model loading, progress/report writes, evaluation and checkpoint save/reload"},
            "trainable_parameters": trainable_parameters, "changed_trainable_tensors": len(changed),
            "adapter_reload": {"passed": True, "max_absolute_logit_difference": float((actual - reference).abs().max()),
                               "logits_shape": list(actual.shape), "all_logits_compared": True,
                               "rtol": 1e-5, "atol": 1e-5, "optimizer_reloaded": True},
            "checkpoint_files": [{"path": path.relative_to(checkpoint).as_posix(),
                                  "size": path.stat().st_size, "sha256": sha256(path)}
                                 for path in sorted(checkpoint.rglob("*")) if path.is_file()]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-profile", choices=tuple(MODEL_PROFILES), default="jev_27b")
    parser.add_argument("--model-dir", type=Path, help="Defaults to the selected profile in project models")
    parser.add_argument("--output", type=Path, required=True, help="Prepared container symlink to an isolated NAS run")
    parser.add_argument("--report", type=Path, required=True,
                        help="New small JSON report in project .dev/artifacts or fast storage")
    parser.add_argument("--steps", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--rank", type=int, choices=(4, 8), default=8)
    parser.add_argument("--sequence-length", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-fast-kernels", action="store_true",
                        help="Require actual training calls to installed causal_conv1d and FLA kernels")
    args = parser.parse_args(argv)
    profile = MODEL_PROFILES[args.model_profile]
    if args.model_dir is None:
        args.model_dir = ROOT / "models" / profile["directory"]
    path = report_path(args.report)
    started = time.monotonic()
    scope = (f"New causal-LM smoke LoRA on the {profile['directory']} BF16 backbone; "
             "released System 1 adapter/head untested and unchanged")
    report = {"status": "starting", "scope": scope, "hostname": socket.gethostname(),
              "model_profile": args.model_profile, "capacity_gate": profile,
              "started_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(__file__),
              "hyperparameters": {"steps": args.steps, "rank": args.rank, "lora_alpha": args.rank * 2,
                                  "sequence_length": args.sequence_length, "batch_size": 1,
                                  "learning_rate": args.learning_rate, "seed": 20261016,
                                  "target_modules": TARGETS, "gradient_checkpointing": True,
                                  "weight_quantization": None, "cpu_offload": False},
              "require_fast_kernels": args.require_fast_kernels, "losses": []}
    write_json(path, report)
    torch = None
    device = None
    gpu_ready = False
    kernel_probe = None
    try:
        require(math.isfinite(args.learning_rate) and 0 < args.learning_rate <= 0.01, "Invalid learning rate")
        output = output_directory(args.output)
        model_root = model_directory(profile, args.model_dir)
        original_stats = file_stats(model_root)
        report.update(model_directory=str(model_root), output_link=str(args.output), output_directory=str(output),
                      original_model_files=len(original_stats))
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
            os.environ[key] = "1"
        os.environ["USE_HUB_KERNELS"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        fast = Path("/root/autodl-tmp/jevgo")
        for key, relative in {"HF_HOME": "cache/huggingface", "HF_HUB_CACHE": "cache/huggingface/hub",
                              "HF_XET_CACHE": "cache/huggingface/xet", "XDG_CACHE_HOME": "cache",
                              "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton",
                              "CUDA_CACHE_PATH": "cache/cuda", "TMPDIR": "tmp"}.items():
            location = fast / relative
            location.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(location)
        packages = {}
        for name in ("torch", "transformers", "peft", "accelerate", "safetensors",
                     "causal-conv1d", "flash-linear-attention", "fla-core"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        report.update(versions=packages, git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
                      cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      nvidia_visible_devices=os.environ.get("NVIDIA_VISIBLE_DEVICES"),
                      attention_implementation="sdpa", hub_kernels_enabled=False)
        import torch as torch_module
        torch = torch_module
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        device = torch.device(args.device)
        require(device.type == "cuda", "This manual acceptance requires CUDA; no CPU fallback")
        require(torch.cuda.is_available(), "CUDA unavailable in this execution context; verify device access before diagnosing the host")
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        free, total = torch.cuda.mem_get_info(device)
        gpu_ready = True
        report["gpu"] = {"device": str(device), "name": torch.cuda.get_device_name(device),
                         "total_bytes": total, "free_bytes_before_loading": free,
                         "torch_cuda_build": torch.version.cuda,
                         "capability": list(torch.cuda.get_device_capability(device))}
        check_capacity(profile, total, free)
        require(torch.cuda.is_bf16_supported(), "Selected GPU does not support BF16")
        config = AutoConfig.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
        check_architecture(profile, config)
        report["model_metadata_sha256"] = {name: sha256(model_root / name)
                                           for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json")}
        torch.manual_seed(20261016)
        torch.cuda.reset_peak_memory_stats(device)
        if args.require_fast_kernels:
            kernel_probe = FastKernelProbe()
            report["fast_kernels"] = kernel_probe.snapshot()
        report["status"] = "loading_original_backbone"
        write_json(path, report)
        tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
        require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS token")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        torch.cuda.synchronize(device)
        model_load_started = time.perf_counter()
        base = AutoModelForCausalLM.from_pretrained(
            model_root, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            device_map={"": str(device)}, attn_implementation="sdpa", use_kernels=False)
        torch.cuda.synchronize(device)
        report["model_load_seconds"] = time.perf_counter() - model_load_started
        require(all(param.device == device for param in base.parameters()), "Backbone was offloaded or split across devices")
        require(not getattr(base, "is_quantized", False), "Quantized weights are not permitted")
        texts = [
            "A customer reports a damaged parcel and asks for a refund. The support team should inspect the order and explain the refund process.",
            "The experiment compares two methods using the same inputs. Record the measurements carefully and preserve the original data for later checks.",
            "The package arrived on time and the contents were intact. The customer confirms that the delivery was successful and no replacement is needed.",
        ]
        batches = []
        for text in texts:
            batch = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True,
                              max_length=args.sequence_length)
            batch = {key: value.to(device) for key, value in batch.items() if key in ("input_ids", "attention_mask")}
            batch["labels"] = batch["input_ids"].clone()
            batch["labels"].masked_fill_(batch["attention_mask"] == 0, -100)
            require(int((batch["labels"][:, 1:] != -100).sum()) > 0, "Training batch has no supervised tokens")
            batches.append(batch)

        def progress(step):
            report["status"] = "training"
            report["losses"].append(step)
            if kernel_probe:
                report["fast_kernels"] = kernel_probe.snapshot()
            write_json(path, report)
            print(json.dumps(step), flush=True)

        def before_save():
            # Loading/training can take time: recheck the mount and symlink at save time.
            require(output_directory(args.output) == output, "Output link changed during acceptance")
            if kernel_probe:
                kernel_probe.assert_used()

        result = train_and_reload(base, batches, output / "checkpoint-final", steps=args.steps,
                                  rank=args.rank, learning_rate=args.learning_rate, on_step=progress,
                                  before_save=before_save)
        if kernel_probe:
            kernel_probe.assert_used()
        require(file_stats(model_root) == original_stats, "Original model files changed during acceptance")
        report.update(result)
        report.update(status="passed", checkpoint=str(output / "checkpoint-final"),
                      original_model_files_unchanged_by_stat=True, convergence_claimed=False)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        if kernel_probe:
            report["fast_kernels"] = kernel_probe.snapshot()
            kernel_probe.close()
        if gpu_ready:
            report["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            report["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(path, report)
        print(json.dumps({"status": report["status"], "report": str(path)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
