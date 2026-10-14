#!/usr/bin/env python3
"""Check ALFWorld imports/logic and TextWorld reset/step without downloading datasets.

Uses the developer fixture by default; --game selects another ALFWorld task.
Without a task asset this reports a partial result, never a benchmark pass.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / ".dev/fixtures/alfworld/game.tw-pddl"
os.environ.setdefault("ALFWORLD_DATA", str(ROOT / "data" / "alfworld"))
os.environ.setdefault("TMPDIR", "/tmp")

import alfworld
from alfworld.agents.environment import get_environment
from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH
import fast_downward
import jericho
import textworld


def check_pddl_backend() -> dict:
    """Exercise the native PDDL state transition engine used by ALFWorld."""
    domain = """(define (domain smoke)
      (:requirements :strips)
      (:predicates (ready) (finished))
      (:action finish :parameters () :precondition (ready)
        :effect (and (finished) (not (ready)))))"""
    problem = """(define (problem smoke-task) (:domain smoke)
      (:init (ready)) (:goal (finished)))"""
    argv = sys.argv[:]
    try:
        _, sas = fast_downward.pddl2sas(domain, problem)
    finally:
        sys.argv = argv
    lib = fast_downward.load_lib()
    try:
        lib.load_sas(sas.encode())
        assert not lib.check_goal()
        count = lib.get_applicable_operators_count()
        assert count == 1
        operators = (fast_downward.Operator * count)()
        lib.get_applicable_operators(operators)
        operator = operators[0]
        effects = (fast_downward.Atom * operator.nb_effect_atoms)()
        lib.apply_operator(operator.id, effects)
        assert lib.check_goal()
        return {"status": "passed", "transition": "ready -> finished"}
    finally:
        fast_downward.close_lib(lib)


def check_task(game_file: Path) -> dict:
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    env = textworld.start(
        str(game_file),
        textworld.EnvInfos(won=True, admissible_commands=True),
        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
    )
    try:
        state = env.reset()
        assert state.feedback.strip()
        assert state.admissible_commands, "ALFWorld task has no admissible actions"
        action = state.admissible_commands[0]
        state, reward, done = env.step(action)
        assert state.feedback.strip()
        return {"status": "passed", "game": str(game_file), "reset": "passed", "step": action}
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, help="Existing ALFWorld game.tw-pddl; defaults to the developer fixture")
    parser.add_argument("--require-assets", action="store_true", help="Exit 2 when no real ALFWorld game was tested")
    args = parser.parse_args()
    environment_class = get_environment("AlfredTWEnv")
    for path in (ALFRED_PDDL_PATH, ALFRED_TWL2_PATH):
        assert Path(path).is_file() and Path(path).stat().st_size > 0, path
    result = {
        "component": "alfworld",
        "status": "partial",
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("alfworld", "textworld", "numpy", "jericho", "fast-downward-textworld")},
        "imports": "passed",
        "environment_class": environment_class.__name__,
        "bundled_logic_files": "passed",
        "pddl_backend": check_pddl_backend(),
    }
    game_file = args.game
    if game_file is None and FIXTURE.is_file():
        game_file = FIXTURE
    if game_file is None:
        game_file = next(Path(os.environ["ALFWORLD_DATA"]).rglob("game.tw-pddl"), None)
    if game_file is not None:
        if not game_file.is_file():
            parser.error(f"ALFWorld task file does not exist: {game_file}")
        result["alfworld_task"] = check_task(game_file.resolve())
        result["status"] = "passed"
    else:
        result["alfworld_task"] = {
            "status": "blocked_missing_assets",
            "data_dir": os.environ["ALFWORLD_DATA"],
            "reason": "No ALFWorld game.tw-pddl is cached; benchmark dataset was not downloaded.",
        }
    print(json.dumps(result, ensure_ascii=False))
    if args.require_assets and result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    # Fast Downward unlinks its temporary .so while dlopen still holds it. NFS
    # turns that into a .nfs file and cannot remove the directory. Keep only
    # these small runtime files on local storage; downloads/build caches stay
    # in the project. The temporary directory is removed at the end of the run.
    local_tmp = os.environ.get("JEVGO_RUNTIME_TMPDIR", os.environ["TMPDIR"])
    with tempfile.TemporaryDirectory(prefix="jevgo-alfworld-", dir=local_tmp) as runtime_tmp:
        previous_tmp = tempfile.tempdir
        tempfile.tempdir = runtime_tmp
        try:
            main()
        finally:
            tempfile.tempdir = previous_tmp
