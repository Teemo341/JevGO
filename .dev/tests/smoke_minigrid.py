"""Reset, take one action, and render MiniGrid without a display server."""

import importlib.metadata
import json
import os
import platform

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import minigrid  # noqa: F401: registers MiniGrid environments
import pygame


def main():
    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    assert "pygame-ce" in installed and "pygame" not in installed, installed
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=0)
        assert observation["image"].shape == (7, 7, 3)
        assert observation["mission"]
        observation, reward, terminated, truncated, _ = env.step(env.unwrapped.actions.forward)
        assert env.unwrapped.step_count == 1
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3
        summary = {
            "status": "passed",
            "environment": "minigrid",
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("minigrid", "gymnasium", "numpy", "pygame-ce")},
            "scenario": "MiniGrid-Empty-5x5-v0",
            "reset": True,
            "steps": env.unwrapped.step_count,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "frame_shape": list(frame.shape),
        }
    finally:
        env.close()
    # rgb_array rendering uses NumPy directly. Exercise the shared pygame-ce
    # provider's SDL, surfarray and freetype path as well, without a real window.
    human_env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    try:
        human_env.reset(seed=0)
        human_env.step(human_env.unwrapped.actions.forward)
        human_env.render()
        surface = pygame.display.get_surface()
        assert surface is not None, "MiniGrid human renderer did not create an SDL surface"
        expected_size = (human_env.unwrapped.screen_size,) * 2
        assert surface.get_size() == expected_size
        pixels = pygame.surfarray.array3d(surface)
        assert pixels.max() > pixels.min(), "MiniGrid human renderer produced a blank surface"
        summary["human_render"] = {
            "status": "passed",
            "surface_size": list(surface.get_size()),
            "sdl_driver": pygame.display.get_driver(),
            "pygame_module": pygame.__file__,
        }
    finally:
        human_env.close()
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
