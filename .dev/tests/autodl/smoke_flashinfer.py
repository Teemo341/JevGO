#!/usr/bin/env python3
"""Compile and execute FlashInfer sampling on an explicitly named AutoDL host.

Run with ``jevgo-autodl run python ... --expected-host HOST --report FAST/FILE``.
This loads no model, changes no installed packages, and uses no network APIs.
The timings describe this tiny compatibility check, not model throughput.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    require(socket.gethostname() == args.expected_host, "Unexpected execution host")
    require(args.expected_host.startswith("autodl-container-"), "AutoDL host required")
    fast = Path("/root/autodl-tmp").resolve(strict=True)
    require(os.path.ismount(fast), "Fast storage is not mounted")
    report_path = args.report.resolve()
    require(report_path.is_relative_to(fast), "Report must be on AutoDL fast storage")
    require(not report_path.exists(), "Report already exists; use a new report filename")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    report = {
        "ok": False,
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "script_sha256": sha256(Path(__file__)),
        "scope": "SM120 CUDA 12.9 FlashInfer sampling and vLLM sampler compatibility",
        "checks": [],
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_HOME", "CUDA_PATH", "FLASHINFER_NVCC",
                "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
                "TMPDIR", "FLASHINFER_WORKSPACE_BASE", "FLASHINFER_CUDA_ARCH_LIST",
                "VLLM_USE_FLASHINFER_SAMPLER", "MAX_JOBS",
            )
        },
    }
    torch = None
    cuda_ready = False
    try:
        for name in ("TMPDIR", "FLASHINFER_WORKSPACE_BASE"):
            value = os.environ.get(name)
            require(bool(value), f"Missing {name}; use jevgo-autodl run")
            require(Path(value).resolve().is_relative_to(fast), f"{name} is not on fast storage")

        import torch as imported_torch

        torch = imported_torch
        require(torch.cuda.is_available(), "CUDA unavailable in this execution context")
        torch.cuda.set_device(0)
        require(torch.cuda.get_device_capability(0) == (12, 0), "Expected SM120 GPU")
        torch.cuda.reset_peak_memory_stats(0)
        cuda_ready = True
        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "memory_free_before_bytes": torch.cuda.mem_get_info(0)[0],
        }
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("flashinfer-python", "vllm", "triton")
        }

        from flashinfer.compilation_context import CompilationContext
        from flashinfer.jit.cpp_ext import get_cuda_path, get_cuda_version
        from flashinfer.jit.sampling import gen_sampling_module
        from flashinfer.sampling import top_k_top_p_sampling_from_probs

        cuda_path = Path(get_cuda_path())
        nvcc = cuda_path / "bin/nvcc"
        nvcc_result = subprocess.run(
            [str(nvcc), "--version"], text=True, capture_output=True, check=True
        )
        version_match = re.search(r"release (\d+\.\d+),", nvcc_result.stdout)
        require(version_match is not None, "Could not parse nvcc version")
        require(version_match.group(1) == "12.9", "Selected nvcc is not CUDA 12.9")
        require(str(get_cuda_version()) == "12.9", "FlashInfer did not detect CUDA 12.9")
        normalized_arch = CompilationContext._normalize_cuda_arch(12, 0)
        require(normalized_arch == (12, "0f"), "Unexpected FlashInfer SM120 target")
        context = CompilationContext()
        require((12, "0f") in context.TARGET_CUDA_ARCHS, "SM120 missing from JIT targets")
        report["compiler"] = {
            "cuda_path": str(cuda_path),
            "nvcc": str(nvcc.resolve()),
            "nvcc_stdout": nvcc_result.stdout,
            "flashinfer_cuda_version": str(get_cuda_version()),
            "normalized_arch": list(normalized_arch),
            "nvcc_arch_flags": context.get_nvcc_flags_list(),
        }

        spec = gen_sampling_module()
        report["jit"] = {
            "library": str(spec.jit_library_path),
            "ninja_file": str(spec.ninja_path),
            "library_existed_before": spec.jit_library_path.exists(),
            "aot_available": spec.is_aot,
        }
        compile_started = time.perf_counter()
        # Explicit build proves the installed compiler works even if an AOT
        # package or a previous cache entry is available. Ninja may reuse an
        # unchanged build; record that fact rather than claim a cold compile.
        spec.build(need_lock=True)
        spec.load(spec.jit_library_path)
        report["jit"].update({
            "build_and_load_seconds": time.perf_counter() - compile_started,
            "library_bytes": spec.jit_library_path.stat().st_size,
            "library_sha256": sha256(spec.jit_library_path),
        })

        batch, vocab = 4, 1024
        expected = [1, 7, 18, 1023]
        one_hot = torch.zeros((batch, vocab), dtype=torch.float32, device="cuda")
        one_hot[torch.arange(batch, device="cuda"), expected] = 1.0
        restricted = torch.zeros_like(one_hot)
        restricted[:, 3] = 0.70
        restricted[:, 19] = 0.25
        restricted[:, 88] = 0.05

        def check_samples(name, fn, allowed):
            torch.cuda.synchronize()
            before = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - before
            values = result.cpu().tolist()
            require(len(values) == batch, f"{name}: unexpected sample shape")
            require(all(value in support for value, support in zip(values, allowed)),
                    f"{name}: sampled outside expected support: {values}")
            report["checks"].append({
                "name": name, "samples": values, "allowed": allowed,
                "seconds": seconds, "ok": True,
            })

        for order in ("joint", "top_k_first"):
            check_samples(
                f"flashinfer_one_hot_{order}",
                lambda: top_k_top_p_sampling_from_probs(
                    one_hot, 1, 0.9, filter_apply_order=order, seed=20261016,
                ),
                [[value] for value in expected],
            )
            check_samples(
                f"flashinfer_top_k_{order}",
                lambda: top_k_top_p_sampling_from_probs(
                    restricted, 2, 1.0, filter_apply_order=order, seed=20261016,
                ),
                [[3, 19]] * batch,
            )
            check_samples(
                f"flashinfer_top_p_{order}",
                lambda: top_k_top_p_sampling_from_probs(
                    restricted, 3, 0.6, filter_apply_order=order, seed=20261016,
                ),
                [[3]] * batch,
            )

        from vllm.v1.sample.ops.topk_topp_sampler import (
            TopKTopPSampler, flashinfer_sampler_supported,
        )

        supported = flashinfer_sampler_supported()
        require(supported, "vLLM FlashInfer sampler is unavailable")
        sampler = TopKTopPSampler()
        require(sampler.forward.__func__ is TopKTopPSampler.forward_cuda,
                "vLLM did not select FlashInfer CUDA dispatch")
        report["vllm_sampler"] = {
            "flashinfer_sampler_supported": supported,
            "forward": sampler.forward.__qualname__,
            "per_request_generators": False,
        }
        for name, probs, k_value, p_value, support in (
            ("vllm_one_hot", one_hot, 1, 0.9, [[value] for value in expected]),
            ("vllm_top_k", restricted, 2, 1.0, [[3, 19]] * batch),
            ("vllm_top_p", restricted, 3, 0.6, [[3]] * batch),
        ):
            k = torch.full((batch,), k_value, dtype=torch.int32, device="cuda")
            p = torch.full((batch,), p_value, dtype=torch.float32, device="cuda")
            logits = probs.log()
            check_samples(name, lambda: sampler(logits.clone(), {}, k, p)[0], support)
        report["ok"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        report["traceback"] = traceback.format_exc()
    finally:
        if cuda_ready:
            report["gpu"].update({
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            })
        report["elapsed_seconds"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"ok": report["ok"], "report": str(report_path),
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
