"""Load a DiscoveryWorld scenario, render observations, act, and tick once."""

import argparse
import base64
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import tempfile

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".dev/cache"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pygame-package", choices=("pygame", "pygame-ce"), default="pygame-ce")
    parser.add_argument("--scenario", default="Tutorial")
    parser.add_argument("--difficulty", default="Normal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, default=ROOT / ".dev/artifacts/validation/discoveryworld-smoke.log")
    args = parser.parse_args()
    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    assert installed & {"pygame", "pygame-ce"} == {args.pygame_package}, installed
    log_path = args.log.resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        # Upstream emits detailed object/asset diagnostics. Keep these for failures.
        with log_path.open("w") as log, contextlib.redirect_stdout(log):
            import pygame
            from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

            # The API writes frame files and clears previous frames on load.
            # A unique temporary cwd confines both side effects to this smoke run.
            with tempfile.TemporaryDirectory(prefix="discoveryworld-smoke-") as run_dir:
                os.chdir(run_dir)
                api = DiscoveryWorldAPI(threadID=0)
                try:
                    loaded = api.loadScenario(args.scenario, args.difficulty, randomSeed=args.seed, numUserAgents=1)
                    assert loaded, "Scenario load failed"
                    before = api.getAgentObservation(0)
                    assert not before["errors"], before["errors"]
                    assert before["vision"]["base64_no_grid"].startswith("data:image/png;base64,")
                    initial_step = before["ui"]["world_steps"]
                    result = api.performAgentAction(0, {"action": "ROTATE_DIRECTION", "arg1": "north"})
                    assert result["success"] and not result["errors"], result
                    tick = api.tick()
                    assert tick["success"] and not tick["errors"], tick
                    after = api.getAgentObservation(0)
                    assert not after["errors"], after["errors"]
                    assert after["ui"]["world_steps"] == initial_step + 1
                    assert after["ui"]["taskProgress"], "No scenario task registered"
                    # Decode both visual observations; a base64 prefix alone is insufficient.
                    from PIL import Image

                    images = {}
                    for phase, observation in (("before", before), ("after", after)):
                        for kind in ("base64_no_grid", "base64_with_grid"):
                            uri = observation["vision"][kind]
                            assert uri.startswith("data:image/png;base64,")
                            payload = base64.b64decode(uri.split(",", 1)[1], validate=True)
                            with Image.open(io.BytesIO(payload)) as frame:
                                assert frame.format == "PNG"
                                frame.load()
                                rgb = frame.convert("RGB")
                                assert min(rgb.size) > 0
                                assert any(low != high for low, high in rgb.getextrema()), "Blank frame"
                                images[f"{phase}/{kind}"] = {
                                    "size": list(rgb.size),
                                    "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                                }
                    summary = {
                        "status": "passed",
                        "environment": "discoveryworld",
                        "python": platform.python_version(),
                        "versions": {name: importlib.metadata.version(name) for name in ("discoveryworld", args.pygame_package, "numpy", "pathfinding")},
                        "pygame_provider": args.pygame_package,
                        "pygame_module": pygame.__file__,
                        "sdl_version": list(pygame.get_sdl_version()),
                        "source_commit": "fd591323920be0d3786ef350955de1945aa571e5",
                        "scenario": args.scenario,
                        "difficulty": args.difficulty,
                        "seed": args.seed,
                        "loaded": loaded,
                        "initial_world_step": initial_step,
                        "final_world_step": after["ui"]["world_steps"],
                        "action": "ROTATE_DIRECTION north",
                        "action_success": result["success"],
                        "rendered_observations": 2,
                        "decoded_images": images,
                        "ui_sha256": {
                            name: hashlib.sha256(json.dumps(obs["ui"], sort_keys=True).encode()).hexdigest()
                            for name, obs in (("before", before), ("after", after))
                        },
                        "log": str(log_path),
                    }
                finally:
                    pygame.quit()
                    os.chdir(previous_cwd)
        print(json.dumps(summary, ensure_ascii=False))
    except Exception:
        print(f"DiscoveryWorld smoke failed; upstream diagnostics: {log_path}")
        raise
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
