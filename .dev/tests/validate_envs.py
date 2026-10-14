"""Validate the shared JevGO environment and each selected component offline."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ("core", "minigrid", "alfworld", "scienceworld", "discoveryworld", "semif", "autotrust")


def run_stage(component, stage, arguments, output, timeout):
    invocation = ["bash", str(ROOT / "scripts/env.sh"), component, *arguments]
    environment = os.environ.copy()
    # Check dependencies once, then reuse that environment without synchronizing
    # it again or fetching remote models.
    environment.update(UV_OFFLINE="1", UV_NO_SYNC="1", PYTHONDONTWRITEBYTECODE="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if stage == "tiny-model":
        # Upstream SemIf requires one visible CUDA device per scorer process.
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        environment["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0] if visible is not None else "0"
    print(f"Checking {component}: {stage}", flush=True)
    process = subprocess.Popen(invocation, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        code = 124
        stderr += f"\nValidation timed out after {timeout} seconds."
    log = output / f"{component}-{stage}.log"
    log.write_text(stdout + "\nSTDERR:\n" + stderr)
    print(f"{component} {stage}: {'passed' if code == 0 else 'failed'}", flush=True)
    return {"stage": stage, "exit_code": code, "log": str(log), "command": invocation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("components", nargs="*", help="Component aliases, or omit to validate all")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Seconds allowed per stage; initial CUDA imports from NAS can be slow (default: 900)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".dev/artifacts/validation",
                        help="Directory for this run's logs and results.json")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    unknown = set(args.components) - set(COMPONENTS)
    if unknown:
        parser.error(f"Unknown components: {sorted(unknown)}")
    components = list(dict.fromkeys(args.components or COMPONENTS))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checked_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    dependencies = run_stage("core", "dependencies", ["--check"], output, args.timeout)
    records = []
    if dependencies["exit_code"] == 0:
        for name in components:
            command = ["python", f".dev/tests/smoke_{name}.py"]
            if name == "alfworld":
                command += ["--require-assets", "--game", str(ROOT / ".dev/fixtures/alfworld/game.tw-pddl")]
            elif name == "discoveryworld":
                command += ["--log", str(output / "discoveryworld-detail.log")]
            stages = [run_stage(name, "smoke", command, output, args.timeout)]
            if name == "semif" and stages[0]["exit_code"] == 0:
                # Keep random-model evidence create-only, as upstream requires.
                model_output = output / f"semif-tiny-{checked_at.strftime('%Y%m%dT%H%M%S%f')}.json"
                command = ["python", ".dev/tests/check_semif_torch_compat.py",
                           "--checkpoint", str(ROOT / ".dev/cache/semif-compat-qwen35"),
                           "--output", str(model_output)]
                stages.append(run_stage(name, "tiny-model", command, output, args.timeout))
            records.append({"component": name, "checks": stages})
    failed = dependencies["exit_code"] != 0 or any(
        stage["exit_code"] for entry in records for stage in entry["checks"]
    )
    report = {
        "schema_version": 2,
        "environment": "jevgo",
        "environment_path": str(ROOT / ".venv"),
        "checked_at": checked_at.isoformat(),
        "status": "failed" if failed else "passed",
        "scope": "One shared dependency check, component smoke tests, and random tiny-model scoring; no pretrained inference or full benchmarks.",
        "requested_components": components,
        "dependencies": dependencies,
        "results": records,
    }
    # Never inherit passing records from older independent environments or
    # components that were not executed during this run.
    destination = output / "results.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Evidence: {destination}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
