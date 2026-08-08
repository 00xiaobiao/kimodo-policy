#!/usr/bin/env python3
"""Trace the first reset of the DoubleDesk IsaacLab environment.

This is a diagnostic script. It does not start the VLA server or run policy
inference. It reproduces the startup path up to the first reset and prints
before/after markers around the reset, manager, observation, camera overlay, and
shared-memory write boundaries.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import types

ISAACLAB_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = str(ISAACLAB_ROOT)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher


def log(message: str) -> None:
    print(f"[reset_trace] {time.time():.6f} {message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
    )
    parser.add_argument(
        "--env_config_yaml",
        default=str(ISAACLAB_ROOT / "tasks/common_test_config/base_test/doubledesk_sonic_test.yaml"),
    )
    parser.add_argument("--episode_seed", type=int, default=250972820)
    parser.add_argument(
        "--set_wait_for_textures",
        choices=("true", "false", "keep"),
        default="keep",
        help="Override env_cfg.wait_for_textures for diagnostics.",
    )
    parser.add_argument(
        "--deactivate_prim",
        action="append",
        default=[],
        help="Runtime stage prim path to deactivate before env.sim.reset. Can be repeated.",
    )
    parser.add_argument(
        "--disable_scene_attr",
        action="append",
        default=[],
        help="Set env_cfg.scene.<attr> = None before gym.make. Can be repeated.",
    )
    parser.add_argument(
        "--room_usd_path",
        default="",
        help="Override env_cfg.scene.room_walls.spawn.usd_path before gym.make.",
    )
    parser.add_argument(
        "--override_scene_usd",
        action="append",
        default=[],
        help="Override env_cfg.scene.<attr>.spawn.usd_path before gym.make, formatted as attr=/abs/path.usd.",
    )
    parser.add_argument(
        "--skip_initialize_task_scene",
        action="store_true",
        default=False,
        help="Skip env_cfg.initialize_task_scene/apply_optional_runtime_augments.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def install_camera_traces() -> None:
    try:
        import tasks.common_observations.camera_state as camera_state
    except Exception as exc:
        log(f"camera_trace import_failed error={exc}")
        return

    overlay = getattr(camera_state, "_add_recording_status_overlay", None)
    if callable(overlay):
        def traced_overlay(*args, **kwargs):
            log("camera_overlay before _add_recording_status_overlay")
            out = overlay(*args, **kwargs)
            log("camera_overlay after _add_recording_status_overlay")
            return out

        camera_state._add_recording_status_overlay = traced_overlay

    writer = getattr(camera_state, "multi_image_writer", None)
    write_images = getattr(writer, "write_images", None)
    if callable(write_images):
        def traced_write_images(images, depths=None):
            image_keys = list(images.keys()) if images else []
            depth_keys = list(depths.keys()) if depths else []
            shapes = {key: getattr(value, "shape", None) for key, value in (images or {}).items()}
            log(
                "multi_image_writer.write_images before "
                f"images={image_keys} depths={depth_keys} shapes={shapes}"
            )
            out = write_images(images, depths)
            log(f"multi_image_writer.write_images after result={out}")
            return out

        writer.write_images = traced_write_images


def install_observation_traces(env) -> None:
    obs_mgr = env.observation_manager

    def traced_compute(self):
        log("observation_manager.compute begin")
        obs_buffer = {}
        for group_name in self._group_obs_term_names:
            log(f"observation_manager.compute_group before group={group_name}")
            obs_buffer[group_name] = self.compute_group(group_name)
            log(f"observation_manager.compute_group after group={group_name}")
        self._obs_buffer = obs_buffer
        log("observation_manager.compute end")
        return obs_buffer

    def traced_compute_group(self, group_name):
        import torch

        if group_name not in self._group_obs_term_names:
            raise ValueError(f"unknown observation group: {group_name}")

        group_term_names = self._group_obs_term_names[group_name]
        group_obs = dict.fromkeys(group_term_names, None)
        obs_terms = zip(group_term_names, self._group_obs_term_cfgs[group_name])

        for term_name, term_cfg in obs_terms:
            log(f"obs_term before group={group_name} term={term_name} func={term_cfg.func}")
            obs = term_cfg.func(self._env, **term_cfg.params)
            log(
                f"obs_term after_func group={group_name} term={term_name} "
                f"type={type(obs)} shape={getattr(obs, 'shape', None)}"
            )
            obs = obs.clone()
            log(f"obs_term after_clone group={group_name} term={term_name}")

            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    log(f"obs_term modifier before group={group_name} term={term_name} modifier={modifier.func}")
                    obs = modifier.func(obs, **modifier.params)
                    log(f"obs_term modifier after group={group_name} term={term_name}")
            if term_cfg.noise:
                log(f"obs_term noise before group={group_name} term={term_name}")
                obs = term_cfg.noise.func(obs, term_cfg.noise)
                log(f"obs_term noise after group={group_name} term={term_name}")
            if term_cfg.clip:
                log(f"obs_term clip before group={group_name} term={term_name}")
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
                log(f"obs_term clip after group={group_name} term={term_name}")
            if term_cfg.scale is not None:
                log(f"obs_term scale before group={group_name} term={term_name}")
                obs = obs.mul_(term_cfg.scale)
                log(f"obs_term scale after group={group_name} term={term_name}")

            if term_cfg.history_length > 0:
                log(f"obs_term history before group={group_name} term={term_name}")
                self._group_obs_term_history_buffer[group_name][term_name].append(obs)
                if term_cfg.flatten_history_dim:
                    group_obs[term_name] = self._group_obs_term_history_buffer[group_name][term_name].buffer.reshape(
                        self._env.num_envs, -1
                    )
                else:
                    group_obs[term_name] = self._group_obs_term_history_buffer[group_name][term_name].buffer
                log(f"obs_term history after group={group_name} term={term_name}")
            else:
                group_obs[term_name] = obs

        if self._group_obs_concatenate[group_name]:
            log(f"compute_group torch.cat before group={group_name}")
            out = torch.cat(list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name])
            log(f"compute_group torch.cat after group={group_name}")
            return out
        log(f"compute_group return_dict group={group_name}")
        return group_obs

    obs_mgr.compute = types.MethodType(traced_compute, obs_mgr)
    obs_mgr.compute_group = types.MethodType(traced_compute_group, obs_mgr)


def install_reset_idx_trace(env) -> None:
    def traced_reset_idx(self, env_ids):
        log(f"_reset_idx begin env_ids={env_ids}")
        log("_reset_idx before curriculum_manager.compute")
        self.curriculum_manager.compute(env_ids=env_ids)
        log("_reset_idx after curriculum_manager.compute")

        log("_reset_idx before scene.reset")
        self.scene.reset(env_ids)
        log("_reset_idx after scene.reset")

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            log(f"_reset_idx before event_manager.apply reset env_step_count={env_step_count}")
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)
            log("_reset_idx after event_manager.apply reset")

        self.extras["log"] = {}
        for name in (
            "observation_manager",
            "action_manager",
            "reward_manager",
            "curriculum_manager",
            "command_manager",
            "event_manager",
            "termination_manager",
            "recorder_manager",
        ):
            manager = getattr(self, name)
            log(f"_reset_idx before {name}.reset")
            info = manager.reset(env_ids)
            log(f"_reset_idx after {name}.reset info_keys={list(info.keys()) if isinstance(info, dict) else type(info)}")
            self.extras["log"].update(info)

        log("_reset_idx before episode_length_buf reset")
        self.episode_length_buf[env_ids] = 0
        log("_reset_idx end")

    env._reset_idx = types.MethodType(traced_reset_idx, env)


def deactivate_prims(prim_paths: list[str]) -> None:
    if not prim_paths:
        return
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            was_active = prim.IsActive()
            prim.SetActive(False)
            log(f"deactivate_prim path={prim_path} was_active={was_active} now_active={prim.IsActive()}")
        else:
            log(f"deactivate_prim path={prim_path} missing_or_invalid")


def traced_reset(env, episode_seed: int):
    import torch

    env_ids = torch.arange(env.num_envs, dtype=torch.int64, device=env.device)
    log(f"manual_reset begin episode_seed={episode_seed} env_ids={env_ids}")

    log("manual_reset before recorder_manager.record_pre_reset")
    env.recorder_manager.record_pre_reset(env_ids)
    log("manual_reset after recorder_manager.record_pre_reset")

    log("manual_reset before _reset_idx")
    env._reset_idx(env_ids)
    log("manual_reset after _reset_idx")

    log("manual_reset before scene.write_data_to_sim")
    env.scene.write_data_to_sim()
    log("manual_reset after scene.write_data_to_sim")

    log("manual_reset before sim.forward")
    env.sim.forward()
    log("manual_reset after sim.forward")

    log(
        "manual_reset sensor_state "
        f"has_rtx_sensors={env.sim.has_rtx_sensors()} rerender_on_reset={env.cfg.rerender_on_reset}"
    )
    if env.sim.has_rtx_sensors() and env.cfg.rerender_on_reset:
        log("manual_reset before sim.render")
        env.sim.render()
        log("manual_reset after sim.render")

    log("manual_reset before recorder_manager.record_post_reset")
    env.recorder_manager.record_post_reset(env_ids)
    log("manual_reset after recorder_manager.record_post_reset")

    log("manual_reset before observation_manager.compute")
    env.obs_buf = env.observation_manager.compute()
    log("manual_reset after observation_manager.compute")

    log(
        "manual_reset texture_wait_state "
        f"wait_for_textures={env.cfg.wait_for_textures} has_rtx_sensors={env.sim.has_rtx_sensors()}"
    )
    if env.cfg.wait_for_textures and env.sim.has_rtx_sensors():
        from isaacsim.core.simulation_manager import SimulationManager

        max_loops = int(os.environ.get("RESET_TRACE_MAX_TEXTURE_WAIT_LOOPS", "50"))
        loops = 0
        while SimulationManager.assets_loading():
            loops += 1
            if loops > max_loops:
                log(
                    "manual_reset texture_wait reached diagnostic limit "
                    f"loops={loops - 1} assets_loading=True"
                )
                break
            log(f"manual_reset before texture_wait render loop={loops}")
            env.sim.render()
            log(f"manual_reset after texture_wait render loop={loops}")
        log(f"manual_reset texture_wait finished loops={loops} assets_loading={SimulationManager.assets_loading()}")

    log("manual_reset end")
    return env.obs_buf, env.extras


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.enable_cameras = True
    args.multi_gpu = False
    disable_multi_gpu_arg = "--/renderer/multiGpu/enabled=False"
    existing_kit_args = (getattr(args, "kit_args", "") or "").split()
    if disable_multi_gpu_arg not in existing_kit_args:
        args.kit_args = " ".join([*existing_kit_args, disable_multi_gpu_arg]).strip()

    log("before AppLauncher")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    log("after AppLauncher")

    import gymnasium as gym
    import tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from tasks.common_env_config import apply_env_config_yaml
    from tasks.common_runtime import apply_optional_runtime_augments

    log("before parse_env_cfg")
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.env_name = args.task
    apply_env_config_yaml(env_cfg, args.env_config_yaml, task_name=args.task, route_name="sonic")
    if args.set_wait_for_textures != "keep":
        env_cfg.wait_for_textures = args.set_wait_for_textures == "true"
        log(f"override env_cfg.wait_for_textures={env_cfg.wait_for_textures}")
    for scene_attr in args.disable_scene_attr:
        if hasattr(env_cfg.scene, scene_attr):
            setattr(env_cfg.scene, scene_attr, None)
            log(f"override env_cfg.scene.{scene_attr}=None")
        else:
            log(f"override env_cfg.scene.{scene_attr}=missing")
    if args.room_usd_path:
        room_walls = getattr(env_cfg.scene, "room_walls", None)
        room_spawn = getattr(room_walls, "spawn", None)
        if room_spawn is None or not hasattr(room_spawn, "usd_path"):
            log("override room_usd_path failed: env_cfg.scene.room_walls.spawn.usd_path missing")
        else:
            room_spawn.usd_path = args.room_usd_path
            log(f"override env_cfg.scene.room_walls.spawn.usd_path={room_spawn.usd_path}")
    for override in args.override_scene_usd:
        if "=" not in override:
            log(f"override_scene_usd invalid={override}")
            continue
        scene_attr, usd_path = override.split("=", 1)
        scene_obj = getattr(env_cfg.scene, scene_attr, None)
        scene_spawn = getattr(scene_obj, "spawn", None)
        if scene_spawn is None or not hasattr(scene_spawn, "usd_path"):
            log(f"override_scene_usd failed attr={scene_attr}: spawn.usd_path missing")
            continue
        scene_spawn.usd_path = usd_path
        log(f"override env_cfg.scene.{scene_attr}.spawn.usd_path={scene_spawn.usd_path}")
    log("after parse_env_cfg/apply_yaml")

    log("before gym.make")
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    log("after gym.make")

    log("before initialize_task_scene")
    if args.skip_initialize_task_scene:
        log("skip initialize_task_scene")
    elif hasattr(env_cfg, "initialize_task_scene"):
        env_cfg.initialize_task_scene(env, args)
    else:
        apply_optional_runtime_augments(args)
    log("after initialize_task_scene")
    deactivate_prims(args.deactivate_prim)

    install_camera_traces()
    install_observation_traces(env)
    install_reset_idx_trace(env)

    log("before env.sim.reset")
    env.sim.reset()
    log("after env.sim.reset")

    traced_reset(env, args.episode_seed)

    log("closing env/app")
    env.close()
    simulation_app.close()
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
