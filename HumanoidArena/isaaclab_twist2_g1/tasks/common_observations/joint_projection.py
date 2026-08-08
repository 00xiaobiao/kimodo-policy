"""
Efficient joint projection utilities for IsaacLab
Projects 3D joint positions to 2D pixel coordinates
"""

import torch
from isaaclab.utils.math import matrix_from_quat


def compute_camera_extrinsics(pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
    """
    Compute camera extrinsic matrix (World-to-Camera transform).

    Args:
        pos_w: Camera position in world frame, shape (N, 3)
        quat_w: Camera orientation quaternion (w, x, y, z), shape (N, 4)

    Returns:
        extrinsics: World-to-Camera transformation matrix, shape (N, 4, 4)
    """
    # Get rotation matrix from quaternion
    R = matrix_from_quat(quat_w)  # (N, 3, 3)

    N = pos_w.shape[0]
    device = pos_w.device

    # Build Camera-to-World matrix
    T_c2w = torch.eye(4, device=device).repeat(N, 1, 1)  # (N, 4, 4)
    T_c2w[:, :3, :3] = R
    T_c2w[:, :3, 3] = pos_w

    # Apply coordinate frame fix (Isaac Lab camera convention)
    # Rotate 90° around Y axis
    R_fix_y = torch.tensor([
        [ 0.0, 0.0, 1.0, 0.0],
        [ 0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0]
    ], device=device)

    # Then rotate -90° around Z axis (clockwise)
    R_fix_z_cw = torch.tensor([
        [ 0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [ 0.0, 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 0.0, 1.0]
    ], device=device)

    T_c2w_corrected = T_c2w @ R_fix_y @ R_fix_z_cw

    # Compute World-to-Camera (extrinsics)
    extrinsics = torch.linalg.inv(T_c2w_corrected)

    return extrinsics


def project_points_to_camera(
    points_3d_world: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    img_width: int = 640,
    img_height: int = 360,
    debug: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points in world coordinates to 2D pixel coordinates.

    Args:
        points_3d_world: 3D points in world frame, shape (N_points, 3)
        intrinsics: Camera intrinsic matrix, shape (3, 3)
        extrinsics: Camera extrinsic matrix (World-to-Camera), shape (4, 4)
        img_width: Image width in pixels
        img_height: Image height in pixels
        debug: Enable debug output

    Returns:
        pixels_2d: 2D pixel coordinates (u, v), shape (N_points, 2)
        valid_mask: Boolean mask indicating which points are in view, shape (N_points,)
    """
    N_points = points_3d_world.shape[0]
    device = points_3d_world.device

    # Convert to homogeneous coordinates
    points_3d_h = torch.cat([
        points_3d_world,
        torch.ones((N_points, 1), device=device)
    ], dim=-1)  # (N_points, 4)

    # Transform to camera frame
    points_cam = (extrinsics @ points_3d_h.T).T  # (N_points, 4)

    # Project to image plane using intrinsics
    # pixels_h = K * [x, y, z]^T
    pixels_h = points_cam[:, :3] @ intrinsics.T  # (N_points, 3)

    # Perspective division: u = x/z, v = y/z
    z_depth = pixels_h[:, 2:3]  # (N_points, 1)
    pixels_2d = pixels_h[:, :2] / z_depth  # (N_points, 2)

    # Check which points are valid (in front of camera and within image bounds)
    valid_depth = z_depth[:, 0] > 0
    valid_u = (pixels_2d[:, 0] >= 0) & (pixels_2d[:, 0] < img_width)
    valid_v = (pixels_2d[:, 1] >= 0) & (pixels_2d[:, 1] < img_height)
    valid_mask = valid_depth & valid_u & valid_v

    # Debug: print validation statistics
    if debug:
        num_positive_depth = valid_depth.sum().item()
        num_valid_u = valid_u.sum().item()
        num_valid_v = valid_v.sum().item()
        print(f"[PROJECT] Positive depth: {num_positive_depth}/{N_points}")
        print(f"[PROJECT] Valid U coords: {num_valid_u}/{N_points}")
        print(f"[PROJECT] Valid V coords: {num_valid_v}/{N_points}")
        print(f"[PROJECT] Z depth range: [{z_depth.min().item():.2f}, {z_depth.max().item():.2f}]")
        print(f"[PROJECT] U coords range: [{pixels_2d[:, 0].min().item():.2f}, {pixels_2d[:, 0].max().item():.2f}]")
        print(f"[PROJECT] V coords range: [{pixels_2d[:, 1].min().item():.2f}, {pixels_2d[:, 1].max().item():.2f}]")

    return pixels_2d, valid_mask


def get_joint_keypoints_2d(
    env,
    camera_name: str = "world_camera",
    img_width: int = 640,
    img_height: int = 360,
    debug: bool = False
) -> dict:
    """
    Get 2D keypoints of robot joints in camera view.

    Args:
        env: IsaacLab environment
        camera_name: Name of the camera in scene
        img_width: Image width
        img_height: Image height
        debug: Enable debug output

    Returns:
        dict with:
            - keypoints_2d: (N_joints, 2) tensor of pixel coordinates
            - valid_mask: (N_joints,) boolean tensor
            - keypoints_2d_list: List of [u, v] for JSON serialization
    """
    # Get camera
    camera = env.scene._sensors.get(camera_name)
    if camera is None:
        if debug:
            print(f"[JOINT_PROJECTION] ERROR: Camera '{camera_name}' not found")
        return None

    # Get robot articulation (assumes first articulation is the robot)
    robot = None
    for key in env.scene.keys():
        if "robot" in key.lower() or "character" in key.lower():
            robot = env.scene[key]
            if debug:
                print(f"[JOINT_PROJECTION] Found robot: {key}")
            break

    if robot is None:
        if debug:
            print(f"[JOINT_PROJECTION] ERROR: No robot found in scene")
        return None

    # Get joint positions in world frame
    all_body_pos = robot.data.body_pos_w[0]  # (N_bodies, 3)
    all_body_names = robot.data.body_names  # List of body names

    # Debug: print all available body names to find correct naming convention
    if debug:
        print(f"[JOINT_PROJECTION] Total bodies: {len(all_body_names)}")
        print(f"[JOINT_PROJECTION] All body names:")
        for i, name in enumerate(all_body_names):
            print(f"  [{i}] {name}")

        # Check if joint_names attribute exists
        if hasattr(robot.data, 'joint_names'):
            print(f"\n[JOINT_PROJECTION] Total joints: {len(robot.data.joint_names)}")
            print(f"[JOINT_PROJECTION] All joint names:")
            for i, name in enumerate(robot.data.joint_names):
                print(f"  [{i}] {name}")

    # G1 29-DOF joint names (from TWIST2 action provider)
    target_joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # Find indices of target joints in body_names
    # Try multiple naming patterns since joint names might differ from body names
    joint_indices = []
    joints_3d_world_list = []
    found_joint_names = []

    # Special mapping for joints without corresponding link names
    special_mappings = {
        "waist_pitch_joint": "torso_link",  # waist_pitch_joint connects waist_roll_link -> torso_link
    }

    for joint_name in target_joint_names:
        found = False

        # Check special mappings first
        if joint_name in special_mappings:
            mapped_name = special_mappings[joint_name]
            if mapped_name in all_body_names:
                idx = all_body_names.index(mapped_name)
                joint_indices.append(idx)
                joints_3d_world_list.append(all_body_pos[idx])
                found_joint_names.append(joint_name)
                found = True
                if debug:
                    print(f"[JOINT_PROJECTION] Matched '{joint_name}' -> '{mapped_name}' (special mapping)")

        # Try exact match
        if not found and joint_name in all_body_names:
            idx = all_body_names.index(joint_name)
            joint_indices.append(idx)
            joints_3d_world_list.append(all_body_pos[idx])
            found_joint_names.append(joint_name)
            found = True

        if not found:
            # Try alternative naming patterns
            # Pattern 1: Replace "_joint" with "_link"
            alt_name1 = joint_name.replace("_joint", "_link")
            if alt_name1 in all_body_names:
                idx = all_body_names.index(alt_name1)
                joint_indices.append(idx)
                joints_3d_world_list.append(all_body_pos[idx])
                found_joint_names.append(joint_name)
                found = True
                if debug:
                    print(f"[JOINT_PROJECTION] Matched '{joint_name}' -> '{alt_name1}'")
            else:
                # Pattern 2: Remove "_joint" suffix
                alt_name2 = joint_name.replace("_joint", "")
                if alt_name2 in all_body_names:
                    idx = all_body_names.index(alt_name2)
                    joint_indices.append(idx)
                    joints_3d_world_list.append(all_body_pos[idx])
                    found_joint_names.append(joint_name)
                    found = True
                    if debug:
                        print(f"[JOINT_PROJECTION] Matched '{joint_name}' -> '{alt_name2}'")

        if not found and debug:
            print(f"[JOINT_PROJECTION] Warning: Joint '{joint_name}' not found in body_names")
            # Also try case-insensitive search to help debugging
            for i, bname in enumerate(all_body_names):
                if joint_name.lower().replace("_joint", "") in bname.lower():
                    print(f"  -> Possible match: body_names[{i}] = '{bname}'")
                    break

    if len(joints_3d_world_list) == 0:
        if debug:
            print(f"[JOINT_PROJECTION] ERROR: No matching joints found")
        return None

    joints_3d_world = torch.stack(joints_3d_world_list)  # (N_joints, 3)

    if debug:
        print(f"[JOINT_PROJECTION] Found {len(joint_indices)}/{len(target_joint_names)} joints")
        print(f"[JOINT_PROJECTION] Joint indices: {joint_indices[:5]}... (first 5)")
        print(f"[JOINT_PROJECTION] First 3 joint positions:\n{joints_3d_world[:3]}")

    # Get camera intrinsics
    intrinsics = camera.data.intrinsic_matrices[0]  # (3, 3)
    if debug:
        print(f"[JOINT_PROJECTION] Camera intrinsics:\n{intrinsics}")

    # Compute camera extrinsics
    pos_w = camera.data.pos_w[0:1]  # (1, 3)
    quat_w = camera.data.quat_w_world[0:1]  # (1, 4)
    if debug:
        print(f"[JOINT_PROJECTION] Camera position: {pos_w}")
        print(f"[JOINT_PROJECTION] Camera quaternion: {quat_w}")
    extrinsics = compute_camera_extrinsics(pos_w, quat_w)[0]  # (4, 4)
    if debug:
        print(f"[JOINT_PROJECTION] Camera extrinsics:\n{extrinsics}")

    # Project joints to 2D
    keypoints_2d, valid_mask = project_points_to_camera(
        joints_3d_world,
        intrinsics,
        extrinsics,
        img_width,
        img_height,
        debug=debug
    )

    # Debug: print some projection results
    if debug:
        num_valid = valid_mask.sum().item()
        print(f"[JOINT_PROJECTION] Valid keypoints: {num_valid}/{len(valid_mask)}")
        if num_valid > 0:
            valid_indices = torch.where(valid_mask)[0][:5]  # First 5 valid points
            print(f"[JOINT_PROJECTION] First valid keypoints:")
            for idx in valid_indices:
                print(f"  Body {idx}: 3D={joints_3d_world[idx]} -> 2D={keypoints_2d[idx]}")
        else:
            print(f"[JOINT_PROJECTION] WARNING: No valid keypoints! Checking first 3 projections:")
            for i in range(min(3, len(keypoints_2d))):
                print(f"  Body {i}: 3D={joints_3d_world[i]} -> 2D={keypoints_2d[i]} valid={valid_mask[i]}")

    # Convert to list for JSON serialization
    keypoints_2d_list = []
    for i in range(keypoints_2d.shape[0]):
        if valid_mask[i]:
            keypoints_2d_list.append([
                float(keypoints_2d[i, 0]),
                float(keypoints_2d[i, 1])
            ])
        else:
            keypoints_2d_list.append(None)  # Out of view

    return {
        "keypoints_2d": keypoints_2d,
        "valid_mask": valid_mask,
        "keypoints_2d_list": keypoints_2d_list,
        "joint_names": found_joint_names,  # Include joint names for reference
        "joint_indices": joint_indices,  # Include body indices for debugging
    }
