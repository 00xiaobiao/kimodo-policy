from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent
project_root_str = str(_PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import torch

import common_env_objects as common_env_objects_module
from common_env_objects import (
    add_env_object_frame_arrays,
    apply_explicit_env_object_states,
    apply_deterministic_object_resets_with_seed,
    collect_recordable_env_object_states,
    ensure_current_episode_object_seed,
)


def _football_state(position: list[float]) -> dict[str, np.ndarray]:
    return {
        "position": np.asarray(position, dtype=np.float32),
        "orientation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "linear_velocity": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "angular_velocity": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
    }


def test_add_env_object_frame_arrays_accepts_legacy_env_obj_layout() -> None:
    organized: dict[str, np.ndarray] = {}
    data_buffer = [
        {"env_obj": {"football": _football_state([1.0, 2.0, 3.0])}},
        {"env_obj": {"football": _football_state([4.0, 5.0, 6.0])}},
    ]

    add_env_object_frame_arrays(organized, data_buffer)

    np.testing.assert_allclose(
        organized["env_obj_football_position"],
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        organized["env_obj_football_linear_velocity"],
        np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=np.float32),
    )


def test_add_env_object_frame_arrays_accepts_sonic_env_layout() -> None:
    organized: dict[str, np.ndarray] = {}
    data_buffer = [
        {
            "env": {
                "football": _football_state([0.0, 1.0, 2.0]),
                "vision": {"rgb": None, "depth": None},
            }
        },
        {
            "env": {
                "football": _football_state([3.0, 4.0, 5.0]),
                "vision": {"rgb": None, "depth": None},
            }
        },
    ]

    add_env_object_frame_arrays(organized, data_buffer)

    np.testing.assert_allclose(
        organized["env_obj_football_position"],
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        organized["env_obj_football_angular_velocity"],
        np.asarray([[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]], dtype=np.float32),
    )


def test_collect_recordable_env_object_states_uses_prim_paths_when_scene_key_missing(monkeypatch) -> None:
    env_cfg = SimpleNamespace(
        deterministic_object_resets=[
            {
                "record_name": "basket",
                "prim_paths": ["/World/envs/env_{env_idx}/Basket"],
            }
        ]
    )

    class _Scene:
        def keys(self):
            return []

    env = SimpleNamespace(scene=_Scene())

    def _fake_read_prim_world_pose(path: str):
        assert path in {
            "/World/envs/env_0/Basket",
            "/World/envs/env_0/Basket/PRootNode",
        }
        return (
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    monkeypatch.setattr(common_env_objects_module, "_read_prim_world_pose", _fake_read_prim_world_pose)

    states = collect_recordable_env_object_states(env, env_cfg)

    assert "basket" in states
    np.testing.assert_allclose(states["basket"]["position"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(states["basket"]["orientation"], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(states["basket"]["linear_velocity"], np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(states["basket"]["angular_velocity"], np.zeros(3, dtype=np.float32))


def test_collect_recordable_env_object_states_prefers_prim_pose_write_mode(monkeypatch) -> None:
    class _AssetWithRigidState:
        def __init__(self):
            self.data = SimpleNamespace(
                root_state_w=torch.tensor(
                    [[9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                )
            )

    class _Scene(dict):
        def __init__(self):
            super().__init__({"basket": _AssetWithRigidState()})

        def keys(self):
            return super().keys()

    env = SimpleNamespace(scene=_Scene())
    env_cfg = SimpleNamespace(
        deterministic_object_resets=[
            {
                "record_name": "basket",
                "scene_keys": ["basket"],
                "prim_paths": ["/World/envs/env_{env_idx}/Basket"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
            }
        ]
    )

    def _fake_read_prim_world_pose(path: str):
        assert path == "/World/envs/env_0/Basket"
        return (
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )

    monkeypatch.setattr(common_env_objects_module, "_read_prim_world_pose", _fake_read_prim_world_pose)

    states = collect_recordable_env_object_states(env, env_cfg)

    np.testing.assert_allclose(states["basket"]["position"], np.asarray([1.0, 2.0, 3.0], dtype=np.float32))


def test_apply_deterministic_object_resets_with_seed_is_repeatable_for_scene_objects() -> None:
    class _DummyObject:
        def __init__(self):
            self.data = SimpleNamespace(
                default_root_state=torch.tensor(
                    [[-2.8, -0.1, 0.745, 1.0, 0.0, 0.0, 0.0, 0.3, 0.2, 0.1, 0.4, 0.5, 0.6]],
                    dtype=torch.float32,
                )
            )
            self.last_written = None

        def write_root_state_to_sim(self, root_state, env_ids=None):
            self.last_written = root_state.clone()

    class _Scene(dict):
        def __init__(self, basket_obj):
            super().__init__({"basket": basket_obj})

        def keys(self):
            return super().keys()

        def write_data_to_sim(self):
            return None

    basket = _DummyObject()
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_Scene(basket),
    )
    env_cfg = SimpleNamespace(
        deterministic_object_resets=[
            {
                "record_name": "basket",
                "scene_keys": ["basket", "Basket"],
                "pose_range": {
                    "x": [-3.82, -3.58],
                    "y": [-3.34, -3.06],
                    "z": [0.80, 0.80],
                    "yaw": [-0.785398, 0.785398],
                },
                "zero_velocity_on_reset": True,
            }
        ]
    )

    first = apply_deterministic_object_resets_with_seed(
        env_cfg,
        env,
        episode_seed=123456,
        seed_source="recorded",
        selected_record_names={"basket"},
    )
    first_state = basket.last_written.clone()
    second = apply_deterministic_object_resets_with_seed(
        env_cfg,
        env,
        episode_seed=123456,
        seed_source="recorded",
        selected_record_names={"basket"},
    )
    second_state = basket.last_written.clone()

    assert first and second
    torch.testing.assert_close(first_state, second_state)
    torch.testing.assert_close(second_state[0, 7:13], torch.zeros(6, dtype=torch.float32))


def test_apply_deterministic_object_resets_with_seed_falls_back_to_prim_paths(monkeypatch) -> None:
    class _AssetWithoutRigidState:
        data = SimpleNamespace()

    class _Scene(dict):
        def __init__(self):
            super().__init__({"basket": _AssetWithoutRigidState()})

        def keys(self):
            return super().keys()

        def write_data_to_sim(self):
            return None

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_Scene(),
    )
    env_cfg = SimpleNamespace(
        deterministic_object_resets=[
            {
                "record_name": "basket",
                "prim_paths": ["/World/envs/env_{env_idx}/Basket"],
                "pose_range": {
                    "x": [-3.82, -3.58],
                    "y": [-3.34, -3.06],
                    "z": [0.80, 0.80],
                    "yaw": [-0.785398, 0.785398],
                },
                "zero_velocity_on_reset": True,
            }
        ]
    )

    monkeypatch.setattr(
        common_env_objects_module,
        "_load_prim_default_states",
        lambda *_args, **_kwargs: {
            0: {
                "position": np.asarray([-2.8, -0.1, 0.745], dtype=np.float32),
                "orientation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            }
        },
    )

    writes: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _fake_write_prim_world_pose(path: str, position: np.ndarray, orientation: np.ndarray) -> bool:
        writes.append((path, position.copy(), orientation.copy()))
        return True

    monkeypatch.setattr(common_env_objects_module, "_write_prim_world_pose", _fake_write_prim_world_pose)

    first = apply_deterministic_object_resets_with_seed(
        env_cfg,
        env,
        episode_seed=123456,
        seed_source="recorded",
        selected_record_names={"basket"},
    )
    first_pos = writes[-1][1].copy()
    first_ori = writes[-1][2].copy()

    second = apply_deterministic_object_resets_with_seed(
        env_cfg,
        env,
        episode_seed=123456,
        seed_source="recorded",
        selected_record_names={"basket"},
    )
    second_pos = writes[-1][1].copy()
    second_ori = writes[-1][2].copy()

    assert first and second
    np.testing.assert_allclose(first_pos, second_pos)
    np.testing.assert_allclose(first_ori, second_ori)


def test_apply_explicit_env_object_states_falls_back_to_prim_paths(monkeypatch) -> None:
    class _AssetWithoutRigidState:
        data = SimpleNamespace()

    class _Scene(dict):
        def __init__(self):
            super().__init__({"basket": _AssetWithoutRigidState()})

        def keys(self):
            return super().keys()

        def write_data_to_sim(self):
            return None

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_Scene(),
    )
    env_cfg = SimpleNamespace(
        deterministic_object_resets=[
            {
                "record_name": "basket",
                "prim_paths": ["/World/envs/env_{env_idx}/Basket"],
            }
        ]
    )

    writes: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _fake_write_prim_world_pose(path: str, position: np.ndarray, orientation: np.ndarray) -> bool:
        writes.append((path, position.copy(), orientation.copy()))
        return True

    monkeypatch.setattr(common_env_objects_module, "_write_prim_world_pose", _fake_write_prim_world_pose)

    applied = apply_explicit_env_object_states(
        env,
        env_cfg,
        {
            "basket": {
                "position": np.asarray([-2.7, -0.2, 0.8], dtype=np.float32),
                "orientation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            }
        },
    )

    assert applied is True
    assert writes
    np.testing.assert_allclose(writes[-1][1], np.asarray([-2.7, -0.2, 0.8], dtype=np.float32))


def test_apply_explicit_env_object_states_prefers_prim_pose_write_mode(monkeypatch) -> None:
    class _AssetWithRigidState:
        def __init__(self):
            self.data = SimpleNamespace(
                default_root_state=torch.tensor(
                    [[9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                )
            )
            self.write_calls = 0

        def write_root_state_to_sim(self, root_state, env_ids=None):
            self.write_calls += 1

    class _Scene(dict):
        def __init__(self, basket_obj):
            super().__init__({"basket": basket_obj})

        def keys(self):
            return super().keys()

        def write_data_to_sim(self):
            return None

    basket = _AssetWithRigidState()
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_Scene(basket),
    )
    env_cfg = SimpleNamespace(
        deterministic_object_resets=[
            {
                "record_name": "basket",
                "scene_keys": ["basket"],
                "prim_paths": ["/World/envs/env_{env_idx}/Basket"],
                "allow_prroot_fallback": False,
                "prim_pose_write_mode": "local_matrix",
            }
        ]
    )

    writes: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _fake_write_prim_world_pose(path: str, position: np.ndarray, orientation: np.ndarray) -> bool:
        writes.append((path, position.copy(), orientation.copy()))
        return True

    monkeypatch.setattr(common_env_objects_module, "_write_prim_world_pose", _fake_write_prim_world_pose)
    monkeypatch.setattr(
        common_env_objects_module,
        "_read_prim_world_pose",
        lambda *_args, **_kwargs: (
            np.asarray([-2.7, -0.2, 0.8], dtype=np.float32),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
    )

    applied = apply_explicit_env_object_states(
        env,
        env_cfg,
        {
            "basket": {
                "position": np.asarray([-2.7, -0.2, 0.8], dtype=np.float32),
                "orientation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            }
        },
    )

    assert applied is True
    assert basket.write_calls == 0
    assert writes
    assert writes[-1][0] == "/World/envs/env_0/Basket"
    np.testing.assert_allclose(writes[-1][1], np.asarray([-2.7, -0.2, 0.8], dtype=np.float32))


def test_ensure_current_episode_object_seed_reuses_existing_seed() -> None:
    env_cfg = SimpleNamespace(
        _current_episode_object_seed=1234,
        _current_episode_object_seed_source="recorded",
        object_reset_seed_source="time",
    )

    seed, source = ensure_current_episode_object_seed(env_cfg)

    assert seed == 1234
    assert source == "recorded"


def test_ensure_current_episode_object_seed_sets_time_seed_once(monkeypatch) -> None:
    env_cfg = SimpleNamespace(object_reset_seed_source="time")
    time_values = iter([111, 222])

    monkeypatch.setattr(common_env_objects_module.time, "time_ns", lambda: next(time_values))

    first_seed, first_source = ensure_current_episode_object_seed(env_cfg)
    second_seed, second_source = ensure_current_episode_object_seed(env_cfg)

    assert first_seed == 111
    assert first_source == "time"
    assert second_seed == 111
    assert second_source == "time"


def test_ensure_current_episode_object_seed_sets_env_seed_once() -> None:
    env_cfg = SimpleNamespace(object_reset_seed_source="env_seed", seed=42)

    first_seed, first_source = ensure_current_episode_object_seed(env_cfg)
    second_seed, second_source = ensure_current_episode_object_seed(env_cfg)

    assert first_source == "env_seed"
    assert second_source == "env_seed"
    assert first_seed == second_seed
    assert getattr(env_cfg, "_episode_object_seed_counter", 0) == 1
