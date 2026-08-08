"""
SMPL utility functions for OpenPI integration.

This module provides utility functions for SMPL data processing, including:
- Angle wrapping
- SMPL action accumulation
- Quaternion operations
- Coordinate transformations
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def angle_wrap(angle):
    """
    Wrap angles to (-pi, pi]. Works elementwise on numpy arrays.

    Args:
        angle: numpy array of angles in radians

    Returns:
        Wrapped angles in range (-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def cumsum_smpl_actions(action_diffs: np.ndarray, last_state: np.ndarray) -> np.ndarray:
    """
    Accumulate SMPL action differences to get state trajectory.

    Args:
        action_diffs: (T, 75) array of action differences
                     [:, :72] are joint angle differences (rotation vectors)
                     [:, 72:] are root translation differences
        last_state: (75,) last SMPL state used as starting point

    Returns:
        smpl_trajectory: (T, 75) accumulated SMPL state trajectory
    """
    # Split into angle and translation components
    angle_diffs = action_diffs[:, :72]   # (T, 72)
    transl_diffs = action_diffs[:, 72:]  # (T, 3)

    # Get starting state
    start_angles = last_state[:72]
    start_transl = last_state[72:75]

    # Accumulate with angle wrapping for rotations
    angles_traj = angle_wrap(
        start_angles[None, :] + np.cumsum(angle_diffs, axis=0)
    )  # (T, 72)

    # Simple linear accumulation for translation
    transl_traj = start_transl[None, :] + np.cumsum(transl_diffs, axis=0)  # (T, 3)

    # Concatenate back to 75 dims
    smpl_traj = np.concatenate([angles_traj, transl_traj], axis=-1)  # (T, 75)

    return smpl_traj


def quat_rotate_inverse_np(q, v, scalar_first=True):
    """
    Rotate vector v by the inverse of quaternion q.

    Args:
        q: quaternion (w,x,y,z) if scalar_first=True, else (x,y,z,w)
        v: vector to rotate
        scalar_first: if True, q is (w,x,y,z); else (x,y,z,w)

    Returns:
        Rotated vector
    """
    q = np.asarray(q)
    v = np.asarray(v)

    if scalar_first:
        q = q[..., [1, 2, 3, 0]]  # Convert to (x,y,z,w)
    else:
        q = q[..., [0, 1, 2, 3]]

    q_w = q[..., -1]
    q_vec = q[..., :3]

    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0

    return a - b + c


def euler_from_quaternion_np(quat, scalar_first=True):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: quaternion array (N, 4)
               (w,x,y,z) if scalar_first=True, else (x,y,z,w)
        scalar_first: if True, quat is (w,x,y,z); else (x,y,z,w)

    Returns:
        roll, pitch, yaw: Euler angles in radians
    """
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]  # Convert to (x,y,z,w)
    else:
        quat = quat[..., [0, 1, 2, 3]]

    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def quat_diff_np(q1, q2, scalar_first=True):
    """
    Compute rotation vector representing rotation from q1 to q2.

    Args:
        q1: first quaternion
        q2: second quaternion
        scalar_first: if True, quaternions are (w,x,y,z); else (x,y,z,w)

    Returns:
        Rotation vector (axis * angle)
    """
    q1 = np.array(q1)
    q2 = np.array(q2)

    # Convert to scipy Rotation object
    r1 = R.from_quat(q1, scalar_first=scalar_first)
    r2 = R.from_quat(q2, scalar_first=scalar_first)

    # Relative rotation
    r_rel = r2 * r1.inv()

    # Rotation vector (axis * angle)
    rotvec = r_rel.as_rotvec()

    return rotvec


def get_default_smpl_state():
    """
    Get default SMPL state (T-pose with default height).

    Returns:
        smpl_state: (75,) array
                    [:72] = zero rotation vectors (T-pose)
                    [72:75] = [0, 0, 0.9] (default root position, 0.9m height)
    """
    smpl_state = np.zeros(75, dtype=np.float32)
    smpl_state[74] = 0.9  # Set z height to 0.9m
    return smpl_state


def extract_mimic_obs_whole_body(qpos, last_qpos, dt=1/30):
    """
    Extract whole body mimic observations from robot joint positions (35 dims).

    This follows the format used by TWIST2 for motion tracking.

    Args:
        qpos: (36,) current robot state [root_pos(3), root_quat(4), dof_pos(29)]
        last_qpos: (36,) previous robot state
        dt: timestep for velocity calculation

    Returns:
        mimic_obs: (35,) observation vector
                   [xy_vel(2), z_pos(1), roll(1), pitch(1), yaw_vel(1), dof_pos(29), mask(1)]
    """
    # Extract components
    root_pos = qpos[0:3]
    last_root_pos = last_qpos[0:3]
    root_quat = qpos[3:7]  # (w,x,y,z) scalar-first format
    last_root_quat = last_qpos[3:7]
    robot_joints = qpos[7:].copy()  # 29 DOFs

    # Compute velocities
    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt

    # Convert to Euler angles
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)

    # Convert root velocities to local frame
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)

    # Construct mimic observation (35 dims)
    mimic_obs = np.concatenate([
        base_vel_local[:2],        # xy velocity in local frame (2)
        root_pos[2:3],             # z position (height) (1)
        roll, pitch,               # roll, pitch angles (2)
        base_ang_vel_local[2:3],   # yaw angular velocity (1)
        robot_joints               # joint positions (29)
    ])

    return mimic_obs


def parse_smpl_state(smpl_state):
    """
    Parse 75-dim SMPL state into components.

    Args:
        smpl_state: (75,) SMPL state array

    Returns:
        dict with keys:
            'body_pose': (63,) body joint rotation vectors (21 joints × 3)
            'root_orient': (3,) root orientation rotation vector
            'hand_pose': (6,) simplified hand pose (2 × 3)
            'trans': (3,) root translation
    """
    return {
        'body_pose': smpl_state[:63],
        'root_orient': smpl_state[63:66],
        'hand_pose': smpl_state[66:72],
        'trans': smpl_state[72:75]
    }


def compose_smpl_state(body_pose, root_orient, hand_pose, trans):
    """
    Compose SMPL state from components.

    Args:
        body_pose: (63,) body joint rotation vectors
        root_orient: (3,) root orientation rotation vector
        hand_pose: (6,) hand pose
        trans: (3,) root translation

    Returns:
        smpl_state: (75,) full SMPL state
    """
    return np.concatenate([body_pose, root_orient, hand_pose, trans], axis=0)


if __name__ == "__main__":
    # Quick tests
    print("Testing SMPL utils...")

    # Test angle_wrap
    angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    wrapped = angle_wrap(angles)
    print(f"angle_wrap test: {wrapped}")
    assert np.allclose(wrapped[-1], 0.0), "2*pi should wrap to 0"

    # Test cumsum_smpl_actions
    last_state = get_default_smpl_state()
    action_diffs = np.random.randn(16, 75) * 0.01  # Small random diffs
    traj = cumsum_smpl_actions(action_diffs, last_state)
    print(f"cumsum_smpl_actions test: traj shape = {traj.shape}")
    assert traj.shape == (16, 75), "Trajectory should be (16, 75)"

    # Test parse/compose
    parsed = parse_smpl_state(last_state)
    recomposed = compose_smpl_state(**parsed)
    assert np.allclose(recomposed, last_state), "Parse/compose should be invertible"

    print("All tests passed!")
