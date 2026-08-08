from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


THIS_DIR = Path(__file__).resolve().parent


def _load_loader_module():
    loader_path = THIS_DIR / "tasks" / "common_env_config" / "loader.py"
    spec = importlib.util.spec_from_file_location("common_env_config_loader", loader_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_env_config_yaml_rejects_task_name_mismatch() -> None:
    loader = _load_loader_module()
    env_cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.001), decimation=10)

    with pytest.raises(ValueError, match="task_name mismatch"):
        loader.apply_env_config_yaml(
            env_cfg,
            "football_single_twist2.yaml",
            task_name="Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
            route_name="twist2",
        )


def test_apply_env_config_yaml_loads_open_door_deterministic_reset_config() -> None:
    loader = _load_loader_module()
    env_cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=0.005, render_interval=4),
        decimation=4,
        object_reset_seed_source="env_seed",
        deterministic_object_resets=[],
        scene=SimpleNamespace(contact_forces=SimpleNamespace(update_period=0.005)),
    )

    loader.apply_env_config_yaml(
        env_cfg,
        "opendoor_sonic.yaml",
        task_name="Isaac-Move-Open-Door-G129-Dex3-Wholebody",
        route_name="sonic",
    )

    assert env_cfg.object_reset_seed_source == "time"
    assert env_cfg.decimation == 4
    assert env_cfg.sim.dt == 0.005
    assert env_cfg.scene.contact_forces.update_period == 0.005
    assert env_cfg.deterministic_object_resets == [
        {
            "record_name": "door",
            "prim_paths": ["/World/envs/env_{env_idx}/Door"],
            "prim_pose_write_mode": "local_matrix",
            "pose_range": {
                "x": [-0.754, 0.55],
                "y": [-0.8, 0.1],
                "z": [0, 0],
            },
            "zero_velocity_on_reset": True,
        }
    ]


def test_resolve_env_config_yaml_path_finds_vision_test_config() -> None:
    loader = _load_loader_module()

    resolved = loader.resolve_env_config_yaml_path("vision/vision_navi_sonic_test.yaml")

    assert resolved.name == "vision_navi_sonic_test.yaml"
    assert resolved.parent.name == "vision"
    assert resolved.parent.parent.name == "common_test_config"


def test_apply_env_config_yaml_loads_vision_randomization_from_test_defaults(tmp_path) -> None:
    loader = _load_loader_module()
    cfg_path = tmp_path / "vision_test.yaml"
    cfg_path.write_text(
        """
task_name: ExampleTask
backend: sonic
test_defaults:
  vision_randomization:
    enabled: true
    prim_path: "/World/light"
    rotation:
      yaw_deg: [-180, 180]
      pitch_deg: [-20, 20]
      roll_deg: [-10, 10]
    intensity_range: [3000, 7000]
    room_rect_lights:
      enabled: true
      root_keywords: ["Room", "cell_light_bars"]
      expected_count_per_env: 8
      disable_count: 4
overrides:
  sim:
    dt: 0.005
  decimation: 4
""",
        encoding="utf-8",
    )
    env_cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.001), decimation=20)

    loader.apply_env_config_yaml(
        env_cfg,
        str(cfg_path),
        task_name="ExampleTask",
        route_name="sonic",
    )

    assert env_cfg.sim.dt == 0.005
    assert env_cfg.decimation == 4
    assert env_cfg.vision_randomization["enabled"] is True
    assert env_cfg.vision_randomization["prim_path"] == "/World/light"
    assert env_cfg.vision_randomization["rotation"]["yaw_deg"] == [-180, 180]
    assert env_cfg.vision_randomization["room_rect_lights"]["disable_count"] == 4
