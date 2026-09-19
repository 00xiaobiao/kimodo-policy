"""
Shared Isaac SimulationApp creation helpers.
"""

from __future__ import annotations

import os
import sys
from simple.utils import env_flag

HYDRA_WAIT_IDLE = "/app/hydraEngine/waitIdle"
HYDRA_RENDER_COMPLETE = "/app/updateOrder/checkForHydraRenderComplete"
THROTTLING_ENABLE_ASYNC = "/exts/isaacsim.core.throttling/enable_async"


def _compact_dict(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None and value != ""}


def _env_int(name: str, default: int | None = None) -> int | None:
    """Read an optional integer launcher setting without raising on bad envs."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def create_simulation_app(
    SimulationApp,
    *,
    headless: bool,
    renderer: str = "RayTracedLighting",
    width: int | None = None,
    height: int | None = None,
    anti_aliasing: int | None = None,
    hide_ui: bool | None = None,
    multi_gpu: bool = False,
):
    experience = os.getenv("SIMPLE_ISAAC_EXPERIENCE", "").strip()
    zero_delay = env_flag("SIMPLE_ISAAC_ZERO_DELAY", default=True)
    disable_throttling_async = env_flag("SIMPLE_ISAAC_DISABLE_THROTTLING_ASYNC", default=True)

    # Isaac's Vulkan device list is not affected consistently by
    # CUDA_VISIBLE_DEVICES on multi-GPU hosts.  The replay worker therefore
    # exposes an explicit launcher override so the app can select the same
    # (usually masked) device for rendering and physics.  These are optional
    # and leave the upstream SIMPLE defaults unchanged for normal users.
    active_gpu = _env_int("SIMPLE_ISAAC_ACTIVE_GPU")
    physics_gpu = _env_int("SIMPLE_ISAAC_PHYSICS_GPU")
    max_gpu_count = _env_int("SIMPLE_ISAAC_MAX_GPU_COUNT")

    settings: list[tuple[str, object, object]] = []
    if zero_delay:
        settings.append((HYDRA_WAIT_IDLE, 1, True))
        settings.append((HYDRA_RENDER_COMPLETE, 1000, 1000))
    if disable_throttling_async:
        settings.append((THROTTLING_ENABLE_ASYNC, "false", False))
    extra_args = [f"--{key}={arg_value}" for key, arg_value, _ in settings]
    portable_root = os.getenv("SIMPLE_ISAAC_PORTABLE_ROOT", "").strip()
    if portable_root:
        os.makedirs(portable_root, exist_ok=True)
        # Isaac Sim 5.1's SimulationApp checks for the *separate* token
        # ``--portable-root`` before deciding whether to append its default
        # ``--portable`` flag.  Passing ``--portable-root=/path`` therefore
        # still enables the default install-relative cache and writes into the
        # home quota.  Put the exact token in argv as well as forwarding the
        # value to Kit so that the launcher selects the data-disk root.
        if "--portable-root" not in sys.argv:
            sys.argv.extend(["--portable-root", portable_root])
        extra_args.extend(["--portable-root", portable_root])

    sim_cfg = _compact_dict({
        "headless": headless,
        "renderer": renderer,
        "multi_gpu": multi_gpu,
        "anti_aliasing": anti_aliasing,
        "hide_ui": hide_ui,
        "width": width,
        "height": height,
        "experience": experience,
        "active_gpu": active_gpu,
        "physics_gpu": physics_gpu,
        "max_gpu_count": max_gpu_count,
    })
    if extra_args:
        sim_cfg["extra_args"] = extra_args

    app = SimulationApp(sim_cfg)

    for key, _, runtime_value in settings:
        app.set_setting(key, runtime_value)

    return app
