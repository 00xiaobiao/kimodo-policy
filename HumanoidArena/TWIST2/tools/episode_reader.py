#!/usr/bin/env python3
"""
Episode data reader and visualizer for TWIST2 demonstrations.

This class provides utilities to:
1. Load episode data from recorded demonstrations
2. Create videos from camera streams (front/world)
3. Visualize joint keypoints on world camera view
4. Access all recorded data (SMPLX, states, actions)

Author: TWIST2 Team
Date: 2026-01-14
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
from tqdm import tqdm


class EpisodeReader:
    """Reader for TWIST2 episode data with visualization capabilities."""

    def __init__(self, episode_path: str):
        """
        Initialize episode reader.

        Args:
            episode_path: Path to episode directory (e.g., 'data/demo_task/episode_0001')
        """
        self.episode_path = Path(episode_path)

        if not self.episode_path.exists():
            raise ValueError(f"Episode path does not exist: {episode_path}")

        # Load data.json
        json_path = self.episode_path / "data.json"
        if not json_path.exists():
            raise ValueError(f"data.json not found in {episode_path}")

        with open(json_path, 'r') as f:
            self.metadata = json.load(f)

        # Extract info
        self.info = self.metadata.get("info", {})
        self.text = self.metadata.get("text", {})
        self.frames = self.metadata.get("data", [])

        # Image parameters
        self.image_width = self.info.get("image", {}).get("width", 640)
        self.image_height = self.info.get("image", {}).get("height", 360)
        self.fps = self.info.get("image", {}).get("fps", 30)

        # Camera paths
        self.front_rgb_dir = self.episode_path / "front_rgb"
        self.world_rgb_dir = self.episode_path / "world_rgb"

        # Check camera availability
        self.has_front_cam = self.front_rgb_dir.exists()
        self.has_world_cam = self.world_rgb_dir.exists()

        print(f"📂 Loaded episode: {self.episode_path}")
        print(f"   Total frames: {len(self.frames)}")
        print(f"   Resolution: {self.image_height}x{self.image_width}")
        print(f"   FPS: {self.fps}")
        print(f"   Front camera: {'✓' if self.has_front_cam else '✗'}")
        print(f"   World camera: {'✓' if self.has_world_cam else '✗'}")

    def __len__(self) -> int:
        """Return number of frames in episode."""
        return len(self.frames)

    def get_frame(self, idx: int) -> Dict[str, Any]:
        """
        Get frame data by index.

        Args:
            idx: Frame index (0 to len-1)

        Returns:
            Dictionary containing frame data (states, actions, keypoints, etc.)
        """
        if idx < 0 or idx >= len(self.frames):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self.frames)-1}]")
        return self.frames[idx]

    def get_image(self, idx: int, camera: str = "front") -> Optional[np.ndarray]:
        """
        Load image from specified camera.

        Args:
            idx: Frame index
            camera: Camera name ('front' or 'world')

        Returns:
            RGB image as numpy array (H, W, 3), or None if not available
        """
        if camera not in ["front", "world"]:
            raise ValueError(f"Invalid camera: {camera}. Must be 'front' or 'world'")

        frame_data = self.get_frame(idx)
        image_path_key = f"{camera}_rgb"

        if image_path_key not in frame_data:
            return None

        image_path = self.episode_path / frame_data[image_path_key]

        if not image_path.exists():
            print(f"⚠️  Image not found: {image_path}")
            return None

        # Load image (OpenCV loads as BGR, convert to RGB)
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_depth(self, idx: int, camera: str = "front") -> Optional[np.ndarray]:
        """
        Load depth map from specified camera.

        Args:
            idx: Frame index
            camera: Camera name ('front' or 'world')

        Returns:
            Depth map as numpy array (H, W) float32, or None if not available
        """
        if camera not in ["front", "world"]:
            raise ValueError(f"Invalid camera: {camera}. Must be 'front' or 'world'")

        frame_data = self.get_frame(idx)
        depth_path_key = f"{camera}_depth"

        if depth_path_key not in frame_data:
            return None

        depth_path = self.episode_path / frame_data[depth_path_key]

        if not depth_path.exists():
            print(f"⚠️  Depth map not found: {depth_path}")
            return None

        # Load depth map (.npy file)
        try:
            depth = np.load(str(depth_path))
            return depth
        except Exception as e:
            print(f"⚠️  Error loading depth map: {e}")
            return None

    def get_keypoints(self, idx: int) -> Optional[List[Optional[List[float]]]]:
        """
        Get joint keypoints for world camera.

        Args:
            idx: Frame index

        Returns:
            List of keypoints, where each keypoint is [u, v] or None if out of view
        """
        frame_data = self.get_frame(idx)
        return frame_data.get("world_camera_joint_keypoints", None)

    def create_video(
        self,
        output_path: str,
        camera: str = "front",
        fps: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        codec: str = "mp4v"
    ) -> bool:
        """
        Create video from camera images.

        Args:
            output_path: Output video file path (e.g., 'output.mp4')
            camera: Camera name ('front' or 'world')
            fps: Frame rate (default: use episode fps)
            start_frame: Starting frame index
            end_frame: Ending frame index (default: last frame)
            codec: Video codec (default: 'mp4v')

        Returns:
            True if successful, False otherwise
        """
        if camera not in ["front", "world"]:
            raise ValueError(f"Invalid camera: {camera}. Must be 'front' or 'world'")

        if camera == "front" and not self.has_front_cam:
            print(f"❌ Front camera not available in this episode")
            return False

        if camera == "world" and not self.has_world_cam:
            print(f"❌ World camera not available in this episode")
            return False

        # Set frame range
        if end_frame is None:
            end_frame = len(self.frames)

        if start_frame < 0 or end_frame > len(self.frames):
            raise ValueError(f"Invalid frame range: [{start_frame}, {end_frame})")

        # Set FPS
        if fps is None:
            fps = self.fps

        # Get first image to determine size
        first_img = self.get_image(start_frame, camera)
        if first_img is None:
            print(f"❌ Failed to load first image")
            return False

        height, width = first_img.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_path = str(output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"❌ Failed to create video writer")
            return False

        print(f"🎬 Creating video: {output_path}")
        print(f"   Camera: {camera}")
        print(f"   Frames: {start_frame} to {end_frame-1} ({end_frame - start_frame} frames)")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {height}x{width}")

        # Write frames
        for idx in tqdm(range(start_frame, end_frame), desc="Writing frames"):
            img = self.get_image(idx, camera)
            if img is None:
                print(f"⚠️  Skipping frame {idx}: image not found")
                continue

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)

        out.release()
        print(f"✅ Video saved: {output_path}")
        return True

    def visualize_keypoints_on_world_cam(
        self,
        output_path: str,
        fps: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        keypoint_radius: int = 2,
        keypoint_color: Tuple[int, int, int] = (255, 0, 0),
        show_frame_number: bool = True,
        codec: str = "mp4v"
    ) -> bool:
        """
        Create video with joint keypoints visualized on world camera view.

        Args:
            output_path: Output video file path (e.g., 'output_with_keypoints.mp4')
            fps: Frame rate (default: use episode fps)
            start_frame: Starting frame index
            end_frame: Ending frame index (default: last frame)
            keypoint_radius: Radius of keypoint circles in pixels
            keypoint_color: RGB color for keypoints (default: red)
            show_frame_number: Whether to show frame number on video
            codec: Video codec (default: 'mp4v')

        Returns:
            True if successful, False otherwise
        """
        if not self.has_world_cam:
            print(f"❌ World camera not available in this episode")
            return False

        # Set frame range
        if end_frame is None:
            end_frame = len(self.frames)

        if start_frame < 0 or end_frame > len(self.frames):
            raise ValueError(f"Invalid frame range: [{start_frame}, {end_frame})")

        # Set FPS
        if fps is None:
            fps = self.fps

        # Get first image to determine size
        first_img = self.get_image(start_frame, "world")
        if first_img is None:
            print(f"❌ Failed to load first image")
            return False

        height, width = first_img.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        output_path = str(output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print(f"❌ Failed to create video writer")
            return False

        print(f"🎬 Creating video with keypoints: {output_path}")
        print(f"   Frames: {start_frame} to {end_frame-1} ({end_frame - start_frame} frames)")
        print(f"   FPS: {fps}")
        print(f"   Resolution: {height}x{width}")
        print(f"   Keypoint color: RGB{keypoint_color}")

        # Write frames with keypoints
        for idx in tqdm(range(start_frame, end_frame), desc="Drawing keypoints"):
            # Load image
            img = self.get_image(idx, "world")
            if img is None:
                print(f"⚠️  Skipping frame {idx}: image not found")
                continue

            # Make a copy for drawing
            img_draw = img.copy()

            # Get and draw keypoints
            keypoints = self.get_keypoints(idx)
            if keypoints is not None:
                num_visible = 0
                for kp in keypoints:
                    if kp is not None:  # Skip out-of-view joints
                        u, v = int(kp[0]), int(kp[1])
                        # Check bounds
                        if 0 <= u < width and 0 <= v < height:
                            cv2.circle(
                                img_draw,
                                (u, v),
                                radius=keypoint_radius,
                                color=keypoint_color,
                                thickness=-1
                            )
                            num_visible += 1

            # Add frame number if requested
            if show_frame_number:
                cv2.putText(
                    img_draw,
                    f"Frame: {idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)

        out.release()
        print(f"✅ Video with keypoints saved: {output_path}")
        return True

    def get_smplx_data(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get SMPLX motion data for a frame."""
        frame_data = self.get_frame(idx)
        return frame_data.get("smplx_data", None)

    def get_state_body(self, idx: int) -> Optional[List[float]]:
        """Get body state for a frame."""
        frame_data = self.get_frame(idx)
        return frame_data.get("state_body", None)

    def get_action_body(self, idx: int) -> Optional[List[float]]:
        """Get body action for a frame."""
        frame_data = self.get_frame(idx)
        return frame_data.get("action_body", None)

    def get_state_hand(self, idx: int, hand: str = "left") -> Optional[List[float]]:
        """Get hand state for a frame (left or right)."""
        frame_data = self.get_frame(idx)
        return frame_data.get(f"state_hand_{hand}", None)

    def get_action_hand(self, idx: int, hand: str = "left") -> Optional[List[float]]:
        """Get hand action for a frame (left or right)."""
        frame_data = self.get_frame(idx)
        return frame_data.get(f"action_hand_{hand}", None)

    def visualize_smplx(self, idx: int, save_path: Optional[str] = None, show: bool = True):
        """
        Visualize SMPLX skeleton in 3D.

        Args:
            idx: Frame index
            save_path: Optional path to save the figure
            show: Whether to display the figure
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Get SMPLX data
        smplx_data = self.get_smplx_data(idx)
        if smplx_data is None:
            print(f"❌ No SMPLX data available for frame {idx}")
            return

        # Extract joint positions
        joint_positions = {}
        for joint_name, joint_data in smplx_data.items():
            if joint_data and len(joint_data) >= 1:
                pos = joint_data[0]  # [x, y, z]
                joint_positions[joint_name] = pos

        # Define skeleton connections (parent-child relationships)
        skeleton_connections = [
            ('Pelvis', 'Spine1'),
            ('Spine1', 'Spine2'),
            ('Spine2', 'Spine3'),
            ('Spine3', 'Neck'),
            ('Neck', 'Head'),
            ('Spine3', 'Left_Collar'),
            ('Left_Collar', 'Left_Shoulder'),
            ('Left_Shoulder', 'Left_Elbow'),
            ('Left_Elbow', 'Left_Wrist'),
            ('Left_Wrist', 'Left_Hand'),
            ('Spine3', 'Right_Collar'),
            ('Right_Collar', 'Right_Shoulder'),
            ('Right_Shoulder', 'Right_Elbow'),
            ('Right_Elbow', 'Right_Wrist'),
            ('Right_Wrist', 'Right_Hand'),
            ('Pelvis', 'Left_Hip'),
            ('Left_Hip', 'Left_Knee'),
            ('Left_Knee', 'Left_Ankle'),
            ('Left_Ankle', 'Left_Foot'),
            ('Pelvis', 'Right_Hip'),
            ('Right_Hip', 'Right_Knee'),
            ('Right_Knee', 'Right_Ankle'),
            ('Right_Ankle', 'Right_Foot'),
        ]

        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot joints
        xs, ys, zs = [], [], []
        for joint_name, pos in joint_positions.items():
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
        ax.scatter(xs, ys, zs, c='red', marker='o', s=50, label='Joints')

        # Plot skeleton connections
        for parent, child in skeleton_connections:
            if parent in joint_positions and child in joint_positions:
                p_pos = joint_positions[parent]
                c_pos = joint_positions[child]
                ax.plot([p_pos[0], c_pos[0]],
                       [p_pos[1], c_pos[1]],
                       [p_pos[2], c_pos[2]],
                       'b-', linewidth=2)

        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'SMPLX Skeleton - Frame {idx}')
        ax.legend()

        # Set equal aspect ratio
        max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
        mid_x = (max(xs) + min(xs)) / 2
        mid_y = (max(ys) + min(ys)) / 2
        mid_z = (max(zs) + min(zs)) / 2
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ SMPLX visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def visualize_qpos(self, start_frame: int = 0, end_frame: Optional[int] = None,
                      save_path: Optional[str] = None, show: bool = True):
        """
        Visualize robot joint positions (qpos) over time.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (default: last frame)
            save_path: Optional path to save the figure
            show: Whether to display the figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if end_frame is None:
            end_frame = len(self.frames)

        # Collect qpos data over time
        body_qpos_list = []
        hand_left_qpos_list = []
        hand_right_qpos_list = []
        neck_qpos_list = []
        frame_indices = []

        for idx in range(start_frame, end_frame):
            state_body = self.get_state_body(idx)
            state_hand_left = self.get_state_hand(idx, 'left')
            state_hand_right = self.get_state_hand(idx, 'right')

            # Get neck state if available
            frame_data = self.get_frame(idx)
            state_neck = frame_data.get('state_neck', None)

            if state_body:
                body_qpos_list.append(state_body)
                frame_indices.append(idx)
            if state_hand_left:
                hand_left_qpos_list.append(state_hand_left)
            if state_hand_right:
                hand_right_qpos_list.append(state_hand_right)
            if state_neck:
                neck_qpos_list.append(state_neck)

        if not body_qpos_list:
            print(f"❌ No qpos data available in frames [{start_frame}, {end_frame})")
            return

        # Convert to numpy arrays
        body_qpos = np.array(body_qpos_list)
        frame_indices = np.array(frame_indices)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Robot Joint Positions (qpos) - Frames [{start_frame}, {end_frame})', fontsize=14)

        # Plot body qpos
        ax = axes[0, 0]
        for i in range(min(body_qpos.shape[1], 10)):  # Plot first 10 joints
            ax.plot(frame_indices, body_qpos[:, i], label=f'Joint {i}', alpha=0.7)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Joint Position (rad)')
        ax.set_title(f'Body Joints (first 10 of {body_qpos.shape[1]})')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Plot body qpos heatmap
        ax = axes[0, 1]
        im = ax.imshow(body_qpos.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Joint Index')
        ax.set_title('Body Joints Heatmap')
        plt.colorbar(im, ax=ax, label='Position (rad)')

        # Plot hand qpos if available
        ax = axes[1, 0]
        if hand_left_qpos_list and hand_right_qpos_list:
            hand_left_qpos = np.array(hand_left_qpos_list)
            hand_right_qpos = np.array(hand_right_qpos_list)

            for i in range(min(hand_left_qpos.shape[1], 7)):
                ax.plot(frame_indices, hand_left_qpos[:, i],
                       label=f'Left {i}', linestyle='-', alpha=0.7)
                ax.plot(frame_indices, hand_right_qpos[:, i],
                       label=f'Right {i}', linestyle='--', alpha=0.7)

            ax.set_xlabel('Frame')
            ax.set_ylabel('Joint Position (rad)')
            ax.set_title('Hand Joints')
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hand data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Hand Joints')

        # Plot statistics
        ax = axes[1, 1]
        stats_text = f"Body Joints: {body_qpos.shape[1]}\n"
        stats_text += f"Frames: {len(frame_indices)}\n"
        stats_text += f"Duration: {len(frame_indices) / self.fps:.2f}s\n\n"
        stats_text += "Body Joint Statistics:\n"
        stats_text += f"  Mean: [{body_qpos.mean(axis=0).min():.3f}, {body_qpos.mean(axis=0).max():.3f}]\n"
        stats_text += f"  Std:  [{body_qpos.std(axis=0).min():.3f}, {body_qpos.std(axis=0).max():.3f}]\n"
        stats_text += f"  Range: [{body_qpos.min():.3f}, {body_qpos.max():.3f}]\n"

        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')
        ax.set_title('Statistics')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Qpos visualization saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def print_info(self):
        """Print detailed episode information."""
        print("\n" + "="*60)
        print("📊 EPISODE INFORMATION")
        print("="*60)
        print(f"Path: {self.episode_path}")
        print(f"\n📝 Metadata:")
        print(f"   Version: {self.info.get('version', 'N/A')}")
        print(f"   Date: {self.info.get('date', 'N/A')}")
        print(f"   Author: {self.info.get('author', 'N/A')}")

        print(f"\n🎯 Task Description:")
        goal = self.text.get("goal", "N/A")
        desc = self.text.get("desc", "N/A")
        steps = self.text.get("steps", "N/A")
        print(f"   Goal: {goal[:100]}..." if len(goal) > 100 else f"   Goal: {goal}")
        print(f"   Description: {desc}")
        print(f"   Steps: {steps}")

        print(f"\n📹 Recording Info:")
        print(f"   Total frames: {len(self.frames)}")
        print(f"   Duration: {len(self.frames)/self.fps:.2f} seconds")
        print(f"   Resolution: {self.image_height}x{self.image_width}")
        print(f"   FPS: {self.fps}")

        print(f"\n📷 Available Cameras:")
        print(f"   Front camera: {'✓ Available' if self.has_front_cam else '✗ Not available'}")
        print(f"   World camera: {'✓ Available' if self.has_world_cam else '✗ Not available'}")

        # Check data availability
        sample_frame = self.frames[0] if self.frames else {}
        print(f"\n📦 Available Data:")
        print(f"   SMPLX data: {'✓' if 'smplx_data' in sample_frame else '✗'}")
        print(f"   Body state/action: {'✓' if 'state_body' in sample_frame else '✗'}")
        print(f"   Hand state/action: {'✓' if 'state_hand_left' in sample_frame else '✗'}")
        print(f"   Joint keypoints: {'✓' if 'world_camera_joint_keypoints' in sample_frame else '✗'}")
        print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TWIST2 Episode Reader and Visualizer")
    parser.add_argument(
        "episode_path",
        type=str,
        help="Path to episode directory (e.g., data/demo_task/episode_0001)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print episode information"
    )
    parser.add_argument(
        "--create-video",
        type=str,
        metavar="OUTPUT",
        help="Create video from camera (specify output path)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="front",
        choices=["front", "world"],
        help="Camera to use (default: front)"
    )
    parser.add_argument(
        "--visualize-keypoints",
        type=str,
        metavar="OUTPUT",
        help="Create video with keypoints on world camera (specify output path)"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (default: 0)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Ending frame index (default: last frame)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output video FPS (default: use episode fps)"
    )

    args = parser.parse_args()

    # Load episode
    try:
        reader = EpisodeReader(args.episode_path)
    except Exception as e:
        print(f"❌ Error loading episode: {e}")
        exit(1)

    # Print info if requested
    if args.info:
        reader.print_info()

    # Create video if requested
    if args.create_video:
        success = reader.create_video(
            output_path=args.create_video,
            camera=args.camera,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        if not success:
            exit(1)

    # Visualize keypoints if requested
    if args.visualize_keypoints:
        success = reader.visualize_keypoints_on_world_cam(
            output_path=args.visualize_keypoints,
            fps=args.fps,
            start_frame=args.start_frame,
            end_frame=args.end_frame
        )
        if not success:
            exit(1)
