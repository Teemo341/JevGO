#!/usr/bin/env python3
"""Load the bundled ScienceWorld JVM simulator, reset one task, and take one step."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TMPDIR", "/tmp")

from scienceworld import ScienceWorldEnv


def main() -> None:
    java = shutil.which("java")
    if java is None:
        raise SystemExit("ScienceWorld requires Java; install Java 11+ or add it to PATH.")
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except OSError as exc:
        raise SystemExit(f"ScienceWorld requires local JVM/Py4J socket access: {exc}") from exc
    java_version = subprocess.run(
        [java, "-version"], check=True, text=True, capture_output=True, timeout=10
    ).stderr.splitlines()[0]
    env = ScienceWorldEnv("", envStepLimit=5)
    try:
        tasks = env.get_task_names()
        assert tasks, "The bundled JVM simulator returned no tasks"
        task = tasks[0]
        env.load(task, 0, "")
        observation, info = env.reset()
        assert isinstance(observation, str) and observation.strip()
        next_observation, reward, done, info = env.step("look around")
        assert isinstance(next_observation, str) and next_observation.strip()
        assert "score" in info
        print(json.dumps({
            "component": "scienceworld",
            "status": "passed",
            "python": platform.python_version(),
            "scienceworld": version("scienceworld"),
            "py4j": version("py4j"),
            "java": java_version,
            "task_count": len(tasks),
            "task": task,
            "variation": 0,
            "reset": "passed",
            "step": "look around",
            "reward": reward,
            "done": done,
            "score": info["score"],
        }, ensure_ascii=False))
    finally:
        env.close()
        env._gateway.java_process.wait(timeout=10)


if __name__ == "__main__":
    main()
