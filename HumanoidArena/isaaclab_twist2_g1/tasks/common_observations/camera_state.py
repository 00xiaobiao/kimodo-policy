# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
camera state
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import numpy as np
import sys
import os
import atexit

# add the project root directory to the path, so that the shared memory tool can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from image_server.shared_memory_utils import MultiImageWriter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# create the global multi-image shared memory writer
print("=" * 80)
print("[CAMERA_STATE DIAGNOSTIC] Module is being imported!")
print("[CAMERA_STATE DIAGNOSTIC] About to create MultiImageWriter...")
print("=" * 80)

try:
    multi_image_writer = MultiImageWriter()
    print("=" * 80)
    print("[CAMERA_STATE DIAGNOSTIC] ✅ MultiImageWriter created successfully!")
    print("=" * 80)
except Exception as e:
    print("=" * 80)
    print(f"[CAMERA_STATE DIAGNOSTIC] ❌ Failed to create MultiImageWriter: {e}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
    raise

# Register cleanup function to ensure shared memory is properly released on exit
def _cleanup_shared_memory():
    """Cleanup function to properly release shared memory resources"""
    try:
        print("[camera_state] Cleaning up global multi_image_writer...")
        multi_image_writer.cleanup()
    except Exception as e:
        print(f"[camera_state] Error during shared memory cleanup: {e}")

atexit.register(_cleanup_shared_memory)

# Global counter for tracking camera updates
_camera_update_count = 0
_camera_last_report_time = None

def _add_recording_status_overlay(rgb_image, recording_display_state, pending_save_jobs: int = 0):
    """Add recording status overlay to RGB image for VR display.

    Args:
        rgb_image: numpy array [H, W, 3] in range [0, 1] (float) or [0, 255] (uint8)
        recording_display_state: str, display state ("idle", "recording", "saved", "discard")
        pending_save_jobs: int, number of active + queued save jobs

    Returns:
        numpy array with recording status overlay
    """
    import cv2
    import numpy as np

    # Convert to uint8 if needed
    if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
        img = (rgb_image * 255).astype(np.uint8)
    else:
        img = rgb_image.copy()

    # Define overlay parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    padding = 10
    circle_radius = 15

    # Determine status based on recording display state
    if recording_display_state == "saving":
        # Currently saving - orange dot
        text = "SAVING..."
        color = (0, 165, 255)  # Orange in BGR (B=0, G=165, R=255)
    elif recording_display_state == "saved":
        # Just saved - green dot
        text = "SAVED"
        color = (0, 255, 0)  # Green in BGR
    elif recording_display_state == "discard":
        # Just cancelled - red dot
        text = "DISCARD"
        color = (0, 0, 255)  # Red in BGR
    elif recording_display_state == "recording":
        # Currently recording - yellow dot
        text = "RECORDING"
        color = (0, 255, 255)  # Yellow in BGR (B=0, G=255, R=255)
    else:  # idle
        # Idle - green dot
        text = "IDLE"
        color = (0, 255, 0)  # Green in BGR

    # Draw filled circle (status indicator)
    circle_center = (padding + circle_radius, padding + circle_radius)
    cv2.circle(img, circle_center, circle_radius, color, -1)

    # Draw text next to circle
    text_pos = (padding + circle_radius * 2 + 10, padding + circle_radius + 8)
    cv2.putText(img, text, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

    # Draw pending save jobs on a second line for operator pacing decisions.
    pending_text = f"SAVE JOBS: {max(0, int(pending_save_jobs))}"
    pending_pos = (padding + circle_radius * 2 + 10, padding + circle_radius + 38)
    cv2.putText(img, pending_text, pending_pos, font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Convert back to float if original was float
    if rgb_image.dtype == np.float32 or rgb_image.dtype == np.float64:
        img = img.astype(np.float32) / 255.0

    return img

def get_camera_image(
    env: ManagerBasedRLEnv,
) -> dict:
    # pass
    """get multiple camera images and depth maps, write them to shared memory

    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance

    Returns:
        dict: dictionary containing multiple camera images
    """
    global _camera_update_count, _camera_last_report_time
    import time

    _camera_update_count += 1
    current_time = time.time()

    # Report camera update frequency every 5 seconds
    if _camera_last_report_time is None:
        _camera_last_report_time = current_time
    elif current_time - _camera_last_report_time >= 5.0:
        elapsed = current_time - _camera_last_report_time
        update_fps = _camera_update_count / elapsed
        print(f"[CAMERA_STATE] 📸 Camera update frequency: {update_fps:.2f} Hz (count={_camera_update_count})")
        _camera_update_count = 0
        _camera_last_report_time = current_time

    # Get recording status from action provider if available
    recording_active = False
    recording_command = "none"

    # Debug: check if action_provider exists
    if not hasattr(get_camera_image, '_debug_checked_provider'):
        get_camera_image._debug_checked_provider = True
        print(f"[CAMERA_STATE DEBUG] Checking action_provider availability:")
        print(f"  - hasattr(env, 'action_provider'): {hasattr(env, 'action_provider')}")
        if hasattr(env, 'action_provider'):
            action_provider = env.action_provider
            print(f"  - action_provider type: {type(action_provider)}")
            print(f"  - hasattr(action_provider, 'is_recording_active'): {hasattr(action_provider, 'is_recording_active')}")
            print(f"  - hasattr(action_provider, 'get_recording_command'): {hasattr(action_provider, 'get_recording_command')}")

    if hasattr(env, 'action_provider'):
        action_provider = env.action_provider
        if hasattr(action_provider, 'is_recording_active'):
            recording_active = action_provider.is_recording_active()
        if hasattr(action_provider, 'get_recording_command'):
            recording_command = action_provider.get_recording_command()

        # Debug: print recording state every 100 frames
        if not hasattr(get_camera_image, '_debug_counter'):
            get_camera_image._debug_counter = 0
        get_camera_image._debug_counter += 1
        if get_camera_image._debug_counter % 100 == 0:
            print(f"[CAMERA_STATE DEBUG] Recording state: active={recording_active}, command={recording_command}")
    else:
        # Debug: warn if action_provider not found
        if not hasattr(get_camera_image, '_debug_warned_no_provider'):
            get_camera_image._debug_warned_no_provider = True
            print(f"[CAMERA_STATE WARNING] env.action_provider not found! Status indicator will always show IDLE.")

    # get the camera images and depth maps
    images = {}
    depths = {}
    # env.sim.render()

    # Head camera (front camera)
    if "front_camera" in env.scene.keys():
        head_image = env.scene["front_camera"].data.output["rgb"][0]  # [batch, height, width, 3]
        images["head"] = head_image.cpu().numpy()

        # Get depth data if available
        if "distance_to_image_plane" in env.scene["front_camera"].data.output:
            head_depth = env.scene["front_camera"].data.output["distance_to_image_plane"][0]
            depths["head"] = head_depth.cpu().numpy()
        else:
            print(f"[CAMERA_STATE] ⚠️ WARNING: 'distance_to_image_plane' NOT found in front camera output!")

    if getattr(env, "_enable_world_camera_stream", False) and "world_camera" in env.scene.keys():
        world_image = env.scene["world_camera"].data.output["rgb"][0]
        images["world"] = world_image.cpu().numpy()

        if "distance_to_image_plane" in env.scene["world_camera"].data.output:
            world_depth = env.scene["world_camera"].data.output["distance_to_image_plane"][0]
            depths["world"] = world_depth.cpu().numpy()

    # Left camera (left wrist camera)
    if "left_wrist_camera" in env.scene.keys():
        left_image = env.scene["left_wrist_camera"].data.output["rgb"][0]
        images["left"] = left_image.cpu().numpy()

        # Get depth data if available
        if "distance_to_image_plane" in env.scene["left_wrist_camera"].data.output:
            left_depth = env.scene["left_wrist_camera"].data.output["distance_to_image_plane"][0]
            depths["left"] = left_depth.cpu().numpy()

    # Right camera (right wrist camera)
    if "right_wrist_camera" in env.scene.keys():
        right_image = env.scene["right_wrist_camera"].data.output["rgb"][0]
        images["right"] = right_image.cpu().numpy()

        # Get depth data if available
        if "distance_to_image_plane" in env.scene["right_wrist_camera"].data.output:
            right_depth = env.scene["right_wrist_camera"].data.output["distance_to_image_plane"][0]
            depths["right"] = right_depth.cpu().numpy()

    # if no camera with the specified name is found, try other common camera names
    if not images:
        # try to find other possible camera names
        available_cameras = [name for name in env.scene.keys() if "camera" in name.lower()]
        print(f"[camera_state] No standard cameras found. Available cameras: {available_cameras}")

        # if there are available cameras, use the first four as head, world, left, right
        for i, camera_name in enumerate(available_cameras[:4]):
            camera_image = env.scene[camera_name].data.output["rgb"][0]

            if i == 0:
                images["head"] = camera_image.cpu().numpy()
            elif i == 1:
                images["world"] = camera_image.cpu().numpy()
            elif i == 2:
                images["left"] = camera_image.cpu().numpy()
            elif i == 3:
                images["right"] = camera_image.cpu().numpy()

    # Add recording status overlay to head camera image (for VR display only)
    # This does NOT affect the original image data stored in collect_recording_data
    if "head" in images:
        # Try to get recording display state from action provider
        recording_display_state = "idle"  # Default
        pending_save_jobs = 0
        try:
            # Access action provider through env if available
            if hasattr(env, 'action_provider') and hasattr(env.action_provider, '_recording_display_state'):
                recording_display_state = env.action_provider._recording_display_state
            if hasattr(env, 'action_provider') and hasattr(env.action_provider, 'get_pending_save_jobs'):
                pending_save_jobs = int(env.action_provider.get_pending_save_jobs())
        except Exception as e:
            pass  # Silently fall back to default

        images["head"] = _add_recording_status_overlay(
            images["head"],
            recording_display_state,
            pending_save_jobs=pending_save_jobs,
        )

    # write the multi-image data (RGB + depth) to shared memory
    if images:
        # print(f"[CAMERA_STATE] Writing {len(images)} images to shared memory: {list(images.keys())}")
        # if depths:
            # print(f"[CAMERA_STATE] Writing {len(depths)} depth maps to shared memory: {list(depths.keys())}")
        success = multi_image_writer.write_images(images, depths if depths else None)
        # if success:
        #     print(f"[CAMERA_STATE] ✅ Successfully wrote images and depth maps to shared memory!")
        # else:
        #     print(f"[CAMERA_STATE] ❌ Failed to write images to shared memory!")
    # else:
    #     print("[camera_state] No camera images found in the environment")

    return torch.zeros((1, 480, 640, 3))
