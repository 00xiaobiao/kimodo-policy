from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

from action_provider.vla_smpl_runtime import (
    CANONICAL_G1_JOINT_NAMES_29,
    VLA_SMPL_ACTION_DIM,
    VLA_SMPL_HAND_DIM,
    VLA_SMPL_STATE_DIM,
    compute_yaw_rate_from_quats,
    quat_angle_between_wxyz,
    quat_conjugate_wxyz,
    quat_from_roll_pitch_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_slerp_shortest_wxyz,
    quat_to_roll_pitch_wxyz,
    quat_to_rot6d_wxyz,
    reorder_canonical_to_twist2_29,
    rot6d_to_quat_wxyz_with_layout,
    yaw_from_quat_wxyz,
)


UNITREE_G1_REFPOSE_V31_SCHEMA_VERSION = "unitree_g1_gmt_refpose_v3_1"
ROBOT_CURRENT_LOCAL_V3_SCHEMA_VERSION = UNITREE_G1_REFPOSE_V31_SCHEMA_VERSION
VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM = VLA_SMPL_STATE_DIM
VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM = VLA_SMPL_ACTION_DIM
VLA_ROBOT_CURRENT_LOCAL_V3_HAND_DIM = VLA_SMPL_HAND_DIM
VLA_UNITREE_G1_REFPOSE_V31_STATE_DIM = VLA_SMPL_STATE_DIM
VLA_UNITREE_G1_REFPOSE_V31_ACTION_DIM = VLA_SMPL_ACTION_DIM
VLA_UNITREE_G1_REFPOSE_V31_HAND_DIM = VLA_SMPL_HAND_DIM

CANONICAL_JOINT_NAMES_29 = CANONICAL_G1_JOINT_NAMES_29

STATE_NAMES = [
    *[f"state.root_heading_canonical_rot6d.{idx}" for idx in range(6)],
    *[f"state.dof_pos.{name}" for name in CANONICAL_JOINT_NAMES_29],
    *[f"state.dof_vel.{name}" for name in CANONICAL_JOINT_NAMES_29],
]

ACTION_NAMES = [
    "action.root_ref_base_local_xy_delta.x",
    "action.root_ref_base_local_xy_delta.y",
    "action.root_z",
    *[f"action.root_ref_rot6d.{idx}" for idx in range(6)],
    *[f"action.joint_pos.{name}" for name in CANONICAL_JOINT_NAMES_29],
    "action.hand_binary.left",
    "action.hand_binary.right",
]


def _rot_from_quat_wxyz(quat_wxyz: np.ndarray) -> R:
    quat_xyzw = quat_normalize_wxyz(np.asarray(quat_wxyz, dtype=np.float32).reshape(4))[[1, 2, 3, 0]]
    return R.from_quat(quat_xyzw)


def quat_heading_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_from_roll_pitch_yaw_wxyz(0.0, 0.0, yaw_from_quat_wxyz(quat_wxyz))


def heading_canonical_robot_quat_wxyz(
    initial_robot_quat_wxyz: np.ndarray,
    current_robot_quat_wxyz: np.ndarray,
) -> np.ndarray:
    initial_heading_inv = quat_conjugate_wxyz(quat_heading_wxyz(initial_robot_quat_wxyz))
    return quat_mul_wxyz(initial_heading_inv, current_robot_quat_wxyz)


def robot_current_local_target_quat_wxyz(
    current_robot_quat_wxyz: np.ndarray,
    target_root_quat_wxyz: np.ndarray,
) -> np.ndarray:
    return quat_mul_wxyz(
        quat_conjugate_wxyz(np.asarray(current_robot_quat_wxyz, dtype=np.float32).reshape(4)),
        np.asarray(target_root_quat_wxyz, dtype=np.float32).reshape(4),
    )


def rotate_world_delta_to_target_heading_local(
    target_root_quat_wxyz: np.ndarray,
    world_xy_delta: np.ndarray,
) -> np.ndarray:
    world_delta_3d = np.zeros((3,), dtype=np.float32)
    world_delta_3d[:2] = np.asarray(world_xy_delta, dtype=np.float32).reshape(2)
    return _rot_from_quat_wxyz(quat_heading_wxyz(target_root_quat_wxyz)).inv().apply(world_delta_3d)[:2].astype(np.float32)


def rotate_target_heading_local_delta_to_world(
    target_root_quat_wxyz: np.ndarray,
    target_heading_local_xy_delta: np.ndarray,
) -> np.ndarray:
    local_delta_3d = np.zeros((3,), dtype=np.float32)
    local_delta_3d[:2] = np.asarray(target_heading_local_xy_delta, dtype=np.float32).reshape(2)
    return _rot_from_quat_wxyz(quat_heading_wxyz(target_root_quat_wxyz)).apply(local_delta_3d)[:2].astype(np.float32)


def rotate_ref_base_local_delta_to_world(
    *,
    root_ref_quat_wxyz: np.ndarray,
    root_ref_base_local_xy_delta: np.ndarray,
    root_z: float | None = None,
    prev_root_z: float | None = None,
) -> np.ndarray:
    local_delta_3d = np.zeros((3,), dtype=np.float32)
    local_delta_3d[:2] = np.asarray(root_ref_base_local_xy_delta, dtype=np.float32).reshape(2)
    rot = _rot_from_quat_wxyz(root_ref_quat_wxyz).as_matrix().astype(np.float32)
    if root_z is not None and prev_root_z is not None:
        dz_world = float(root_z) - float(prev_root_z)
        denom = float(rot[2, 2])
        if abs(denom) > 1e-6:
            local_delta_3d[2] = (
                dz_world
                - float(rot[2, 0]) * local_delta_3d[0]
                - float(rot[2, 1]) * local_delta_3d[1]
            ) / denom
    return rot.dot(local_delta_3d)[:2].astype(np.float32)


def build_vla_rotlocal_v3_observation_state(
    *,
    initial_robot_orientation_wxyz: np.ndarray,
    root_orientation_wxyz: np.ndarray,
    joint_pos_canonical_29: np.ndarray,
    joint_vel_canonical_29: np.ndarray,
) -> np.ndarray:
    heading_canonical_root = heading_canonical_robot_quat_wxyz(
        initial_robot_orientation_wxyz,
        root_orientation_wxyz,
    )
    state = np.concatenate(
        [
            quat_to_rot6d_wxyz(heading_canonical_root).reshape(6),
            np.asarray(joint_pos_canonical_29, dtype=np.float32).reshape(29),
            np.asarray(joint_vel_canonical_29, dtype=np.float32).reshape(29),
        ],
        axis=0,
    ).astype(np.float32)
    if state.shape != (VLA_ROBOT_CURRENT_LOCAL_V3_STATE_DIM,):
        raise ValueError(f"Unexpected VLA v3.1 observation.state shape: {state.shape}")
    return state


def build_vla_rotlocal_v3_action(
    *,
    root_target_heading_local_xy_delta: np.ndarray,
    root_z: np.ndarray | float,
    root_current_local_target_rot6d: np.ndarray,
    joint_pos_canonical_29: np.ndarray,
    hand_binary: np.ndarray,
) -> np.ndarray:
    action = np.concatenate(
        [
            np.asarray(root_target_heading_local_xy_delta, dtype=np.float32).reshape(2),
            np.asarray(root_z, dtype=np.float32).reshape(1),
            np.asarray(root_current_local_target_rot6d, dtype=np.float32).reshape(6),
            np.asarray(joint_pos_canonical_29, dtype=np.float32).reshape(29),
            np.asarray(hand_binary, dtype=np.float32).reshape(2),
        ],
        axis=0,
    ).astype(np.float32)
    if action.shape != (VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,):
        raise ValueError(f"Unexpected VLA v3.1 action shape: {action.shape}")
    return action


@dataclass
class UnifiedRobotCurrentLocalActionFrameV3:
    root_target_heading_local_xy_delta: np.ndarray
    root_xy_delta_world: np.ndarray
    root_z: float
    body_pos_world: np.ndarray
    root_current_local_target_rot6d: np.ndarray
    root_current_local_target_quat_wxyz: np.ndarray
    target_root_quat_wxyz: np.ndarray
    current_robot_quat_wxyz: np.ndarray
    joint_pos_canonical_29: np.ndarray
    hand_binary: np.ndarray
    prev_target_root_quat_wxyz: np.ndarray | None
    prev_joint_pos_canonical_29: np.ndarray | None

    @property
    def root_local_xy_delta(self) -> np.ndarray:
        return self.root_target_heading_local_xy_delta

    @property
    def root_ref_base_local_xy_delta(self) -> np.ndarray:
        return self.root_target_heading_local_xy_delta

    @property
    def root_target_local_xy_delta(self) -> np.ndarray:
        return self.root_target_heading_local_xy_delta

    @property
    def root_orient_rot6d(self) -> np.ndarray:
        return self.root_current_local_target_rot6d

    @property
    def root_ref_rot6d(self) -> np.ndarray:
        return self.root_current_local_target_rot6d

    @property
    def root_quat_wxyz(self) -> np.ndarray:
        return self.target_root_quat_wxyz

    @property
    def root_ref_quat_wxyz(self) -> np.ndarray:
        return self.target_root_quat_wxyz

    @property
    def prev_root_quat_wxyz(self) -> np.ndarray | None:
        return self.prev_target_root_quat_wxyz


class UnifiedRobotCurrentLocalActionRuntimeV3:
    """V3.1 refpose runtime kept under the old import name during migration."""

    def __init__(self, root_rot6d_layout: str = "row", max_root_delta_deg: float | None = None) -> None:
        self._root_rot6d_layout = str(root_rot6d_layout).strip().lower()
        if self._root_rot6d_layout not in {"row", "col", "auto"}:
            raise ValueError(
                f"Unsupported root_rot6d_layout={root_rot6d_layout}. "
                "Expected one of ['row', 'col', 'auto']."
            )
        self._max_root_delta_rad: float | None = None
        self.set_max_root_delta_deg(max_root_delta_deg)
        self.reset()

    def reset(
        self,
        *,
        body_xy_world: np.ndarray | None = None,
        target_root_quat_wxyz: np.ndarray | None = None,
    ) -> None:
        self._body_xy_world = None if body_xy_world is None else np.asarray(body_xy_world, dtype=np.float32).reshape(2).copy()
        self._body_z_world: float | None = None
        self._prev_target_root_quat_wxyz = (
            None
            if target_root_quat_wxyz is None
            else quat_normalize_wxyz(np.asarray(target_root_quat_wxyz, dtype=np.float32).reshape(4))
        )
        self._episode_ref_to_world_heading_quat_wxyz = (
            None
            if self._prev_target_root_quat_wxyz is None
            else quat_heading_wxyz(self._prev_target_root_quat_wxyz)
        )
        self._prev_root_quat_wxyz = None if self._prev_target_root_quat_wxyz is None else self._prev_target_root_quat_wxyz.copy()
        self._prev_action_rel_quat_wxyz: np.ndarray | None = None
        self._prev_joint_pos_canonical_29: np.ndarray | None = None
        self._last_selected_root_rot6d_layout: str = "row"

    def prime_root_quat(self, root_quat_wxyz: np.ndarray) -> None:
        root_quat = quat_normalize_wxyz(np.asarray(root_quat_wxyz, dtype=np.float32).reshape(4))
        self._episode_ref_to_world_heading_quat_wxyz = quat_heading_wxyz(root_quat)
        self._prev_target_root_quat_wxyz = root_quat.copy()
        self._prev_root_quat_wxyz = root_quat.copy()

    def set_max_root_delta_deg(self, max_root_delta_deg: float | None) -> None:
        if max_root_delta_deg is None:
            self._max_root_delta_rad = None
            return
        value = float(max_root_delta_deg)
        if not np.isfinite(value) or value <= 0.0:
            self._max_root_delta_rad = None
            return
        self._max_root_delta_rad = float(np.deg2rad(value))

    def _decode_root_ref_quat(self, root_ref_rot6d: np.ndarray) -> np.ndarray:
        if self._root_rot6d_layout in {"row", "col"}:
            root_ref_quat_wxyz = rot6d_to_quat_wxyz_with_layout(
                root_ref_rot6d,
                layout=self._root_rot6d_layout,
            ).reshape(4).astype(np.float32)
            self._last_selected_root_rot6d_layout = self._root_rot6d_layout
            return root_ref_quat_wxyz

        quat_row = rot6d_to_quat_wxyz_with_layout(
            root_ref_rot6d,
            layout="row",
        ).reshape(4).astype(np.float32)
        quat_col = rot6d_to_quat_wxyz_with_layout(
            root_ref_rot6d,
            layout="col",
        ).reshape(4).astype(np.float32)
        if self._prev_action_rel_quat_wxyz is None:
            self._last_selected_root_rot6d_layout = "row"
            return quat_row
        row_delta = quat_angle_between_wxyz(self._prev_action_rel_quat_wxyz, quat_row)
        col_delta = quat_angle_between_wxyz(self._prev_action_rel_quat_wxyz, quat_col)
        if col_delta < row_delta:
            self._last_selected_root_rot6d_layout = "col"
            return quat_col
        self._last_selected_root_rot6d_layout = "row"
        return quat_row

    def _decode_action_rel_quat(self, root_current_local_target_rot6d: np.ndarray) -> np.ndarray:
        return self._decode_root_ref_quat(root_current_local_target_rot6d)

    def step(
        self,
        action: np.ndarray,
        *,
        current_robot_quat_wxyz: np.ndarray,
        current_robot_xy_world: np.ndarray,
    ) -> UnifiedRobotCurrentLocalActionFrameV3:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM,):
            raise ValueError(
                f"Expected VLA v3.1 action dim {VLA_ROBOT_CURRENT_LOCAL_V3_ACTION_DIM}, got {action.shape}"
            )

        current_robot_quat_wxyz = quat_normalize_wxyz(
            np.asarray(current_robot_quat_wxyz, dtype=np.float32).reshape(4)
        )
        current_robot_xy_world = np.asarray(current_robot_xy_world, dtype=np.float32).reshape(2)
        if self._body_xy_world is None:
            self._body_xy_world = current_robot_xy_world.copy()
        if self._episode_ref_to_world_heading_quat_wxyz is None:
            self._episode_ref_to_world_heading_quat_wxyz = quat_heading_wxyz(current_robot_quat_wxyz)

        root_ref_base_local_xy_delta = action[0:2].astype(np.float32, copy=True)
        root_z = float(action[2])
        root_ref_rot6d = action[3:9].astype(np.float32, copy=True)
        joint_pos_canonical_29 = action[9:38].astype(np.float32, copy=True)
        hand_binary = action[38:40].astype(np.float32, copy=True)

        root_ref_quat_episode_wxyz = self._decode_root_ref_quat(root_ref_rot6d)
        target_root_quat_wxyz = quat_mul_wxyz(
            self._episode_ref_to_world_heading_quat_wxyz,
            root_ref_quat_episode_wxyz,
        )
        if self._max_root_delta_rad is not None and self._prev_target_root_quat_wxyz is not None:
            root_delta = quat_angle_between_wxyz(self._prev_target_root_quat_wxyz, target_root_quat_wxyz)
            if root_delta > self._max_root_delta_rad:
                blend = self._max_root_delta_rad / max(root_delta, 1e-8)
                target_root_quat_wxyz = quat_slerp_shortest_wxyz(
                    self._prev_target_root_quat_wxyz,
                    target_root_quat_wxyz,
                    blend,
                )

        prev_root_z = self._body_z_world if self._body_z_world is not None else root_z
        root_xy_delta_world = rotate_ref_base_local_delta_to_world(
            root_ref_quat_wxyz=target_root_quat_wxyz,
            root_ref_base_local_xy_delta=root_ref_base_local_xy_delta,
            root_z=root_z,
            prev_root_z=prev_root_z,
        )
        prev_target_root_quat = (
            None if self._prev_target_root_quat_wxyz is None else self._prev_target_root_quat_wxyz.copy()
        )
        prev_joint_pos = (
            None if self._prev_joint_pos_canonical_29 is None else self._prev_joint_pos_canonical_29.copy()
        )

        self._body_xy_world = self._body_xy_world + root_xy_delta_world
        self._body_z_world = root_z
        body_pos_world = np.array(
            [self._body_xy_world[0], self._body_xy_world[1], root_z],
            dtype=np.float32,
        )
        self._prev_target_root_quat_wxyz = target_root_quat_wxyz.copy()
        self._prev_root_quat_wxyz = target_root_quat_wxyz.copy()
        self._prev_action_rel_quat_wxyz = root_ref_quat_episode_wxyz.copy()
        self._prev_joint_pos_canonical_29 = joint_pos_canonical_29.copy()

        return UnifiedRobotCurrentLocalActionFrameV3(
            root_target_heading_local_xy_delta=root_ref_base_local_xy_delta,
            root_xy_delta_world=root_xy_delta_world,
            root_z=root_z,
            body_pos_world=body_pos_world,
            root_current_local_target_rot6d=root_ref_rot6d,
            root_current_local_target_quat_wxyz=target_root_quat_wxyz,
            target_root_quat_wxyz=target_root_quat_wxyz,
            current_robot_quat_wxyz=current_robot_quat_wxyz,
            joint_pos_canonical_29=joint_pos_canonical_29,
            hand_binary=hand_binary,
            prev_target_root_quat_wxyz=prev_target_root_quat,
            prev_joint_pos_canonical_29=prev_joint_pos,
        )


def build_twist2_mimic_obs_v3(
    *,
    runtime_frame: UnifiedRobotCurrentLocalActionFrameV3,
    control_dt: float,
) -> np.ndarray:
    dt = max(float(control_dt), 1e-6)
    target_root_quat_wxyz = quat_normalize_wxyz(runtime_frame.target_root_quat_wxyz.astype(np.float32, copy=False))
    xy_vel_ref_base_local = (runtime_frame.root_ref_base_local_xy_delta / dt).astype(np.float32)
    roll_pitch = quat_to_roll_pitch_wxyz(target_root_quat_wxyz)
    yaw_vel = compute_yaw_rate_from_quats(
        runtime_frame.prev_target_root_quat_wxyz,
        target_root_quat_wxyz,
        control_dt,
    )
    joint_pos_twist2 = reorder_canonical_to_twist2_29(runtime_frame.joint_pos_canonical_29)
    mimic_obs = np.concatenate(
        [
            xy_vel_ref_base_local,
            np.array([runtime_frame.root_z], dtype=np.float32),
            roll_pitch,
            yaw_vel,
            joint_pos_twist2,
        ],
        axis=0,
    ).astype(np.float32)
    if mimic_obs.shape != (35,):
        raise ValueError(f"Unexpected TWIST2 mimic_obs v3.1 shape: {mimic_obs.shape}")
    return mimic_obs


def build_sonic_joint29_payload_v3(
    *,
    runtime_frame: UnifiedRobotCurrentLocalActionFrameV3,
    control_dt: float,
) -> dict[str, np.ndarray]:
    if runtime_frame.prev_joint_pos_canonical_29 is None:
        joint_vel = np.zeros((29,), dtype=np.float32)
    else:
        joint_vel = (
            (runtime_frame.joint_pos_canonical_29 - runtime_frame.prev_joint_pos_canonical_29)
            / max(float(control_dt), 1e-6)
        ).astype(np.float32)
    return {
        "body_pos": runtime_frame.body_pos_world.astype(np.float32, copy=True),
        "body_quat_w": runtime_frame.target_root_quat_wxyz.astype(np.float32, copy=True),
        "joint_pos": runtime_frame.joint_pos_canonical_29.astype(np.float32, copy=True),
        "joint_vel": joint_vel,
    }


def build_vla_refpose_v31_action(
    *,
    root_ref_base_local_xy_delta: np.ndarray,
    root_z: np.ndarray | float,
    root_ref_rot6d: np.ndarray,
    joint_pos_canonical_29: np.ndarray,
    hand_binary: np.ndarray,
) -> np.ndarray:
    return build_vla_rotlocal_v3_action(
        root_target_heading_local_xy_delta=root_ref_base_local_xy_delta,
        root_z=root_z,
        root_current_local_target_rot6d=root_ref_rot6d,
        joint_pos_canonical_29=joint_pos_canonical_29,
        hand_binary=hand_binary,
    )


UnifiedRefPoseActionFrameV31 = UnifiedRobotCurrentLocalActionFrameV3
UnifiedRefPoseActionRuntimeV31 = UnifiedRobotCurrentLocalActionRuntimeV3
build_vla_refpose_v31_observation_state = build_vla_rotlocal_v3_observation_state
build_twist2_mimic_obs_v31 = build_twist2_mimic_obs_v3
build_sonic_joint29_payload_v31 = build_sonic_joint29_payload_v3
