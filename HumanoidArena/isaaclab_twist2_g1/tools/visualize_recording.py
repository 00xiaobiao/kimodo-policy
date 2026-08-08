#!/usr/bin/env python3
"""
Visualization tool for recorded data from action_provider_wh_twist2.py

Usage:
    python visualize_recording.py <path_to_npz_file>
    python visualize_recording.py ./recording_data/move_football_g1_29dof_dex3_wholebody_1234567890.npz

Controls:
    - Left/Right Arrow: Navigate frames
    - Space: Play/Pause
    - Q/Esc: Quit
    - S: Save current frame as image
    - I: Show/Hide info overlay
"""

import argparse
import numpy as np
import cv2
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm


class RecordingVisualizer:
    """Visualizer for recorded data from action_provider_wh_twist2.py"""

    def __init__(self, npz_path: str):
        """Initialize the visualizer.

        Args:
            npz_path: Path to the .npz recording file
        """
        self.npz_path = npz_path
        self.data = None
        self.current_frame = 0
        self.playing = False
        self.show_info = True
        self.fps = 30  # Playback FPS

        # Performance optimization: cache processed frames
        self.frame_cache = {}  # {frame_idx: processed_image}
        self.cache_enabled = True
        self.max_cache_size = 100  # Limit cache to prevent memory overflow
        self.fast_load = False  # Skip preprocessing, load on-demand

        # Load data
        self._load_data()

    def _load_data(self):
        """Load the npz file and parse its contents."""
        print(f"Loading recording from: {self.npz_path}")

        if not os.path.exists(self.npz_path):
            raise FileNotFoundError(f"File not found: {self.npz_path}")

        self.data = np.load(self.npz_path, allow_pickle=True)

        # Print available keys
        print(f"\nAvailable keys in recording:")
        for key in sorted(self.data.keys()):
            item = self.data[key]
            if isinstance(item, np.ndarray):
                print(f"  {key:40s}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"  {key:40s}: {type(item)}")

        # Parse metadata
        self.num_frames = int(self.data['num_frames'])
        self.task_name = str(self.data['task'])

        # Parse observation semantics
        if 'observation_semantics' in self.data:
            self.obs_semantics = json.loads(str(self.data['observation_semantics']))
        else:
            self.obs_semantics = None

        # Parse human data
        if 'human_smplx_data' in self.data:
            self.human_smplx_list = json.loads(str(self.data['human_smplx_data']))
        else:
            self.human_smplx_list = None

        if 'human_info_data' in self.data:
            self.human_info_list = json.loads(str(self.data['human_info_data']))
        else:
            self.human_info_list = None

        # Check vision data availability
        self.has_vision = 'vision_rgb' in self.data and 'vision_frame_indices' in self.data
        if self.has_vision:
            self.vision_frame_indices = self.data['vision_frame_indices']
            print(f"\n✅ Vision data available for frames: {self.vision_frame_indices}")

            # CRITICAL: Load vision data into memory immediately to avoid slow lazy loading
            print(f"📦 Loading vision data into memory...")
            import time
            load_start = time.time()
            self.vision_rgb_array = np.array(self.data['vision_rgb'])
            self.vision_depth_array = np.array(self.data['vision_depth'])
            load_time = time.time() - load_start
            print(f"✅ Loaded vision data in {load_time:.2f}s")
        else:
            print(f"\n⚠️  No vision data found in recording")
            self.vision_rgb_array = None
            self.vision_depth_array = None

        print(f"\n📊 Recording Summary:")
        print(f"  Task: {self.task_name}")
        print(f"  Total frames: {self.num_frames}")
        print(f"  Control frequency: {self.data['system_control_frequency'][0]:.1f} Hz")
        print(f"  Physics dt: {self.data['system_physics_dt'][0]:.4f} s")
        print(f"  Decimation: {self.data['system_decimation'][0]}")

        if self.human_info_list and self.human_info_list[0]:
            human_info = self.human_info_list[0]
            print(f"  Human height: {human_info.get('height', 'N/A'):.3f} m")

        # Preprocess vision data for faster playback
        if not self.fast_load:
            print(f"\n⚡ Preprocessing vision data for faster playback...")
            import time
            preprocess_start = time.time()
            self._preprocess_vision_data()
            preprocess_time = time.time() - preprocess_start
            if preprocess_time > 1.0:
                print(f"  ⏱️  Preprocessing took {preprocess_time:.1f}s")
        else:
            print(f"\n⚡ Fast load mode: skipping preprocessing (will load frames on-demand)")

    def _preprocess_vision_data(self):
        """Preprocess and cache vision data for faster playback."""
        if not self.has_vision:
            return

        import time

        total_frames = len(self.vision_rgb_array)

        # Check image size
        sample_rgb = self.vision_rgb_array[0]
        h, w = sample_rgb.shape[:2]
        print(f"  Vision frame resolution: {w}x{h}")
        print(f"  Total vision frames to process: {total_frames}")

        # Determine if we need to downsample for performance
        max_dimension = max(h, w)
        downsample_factor = 1
        if max_dimension > 1080:
            downsample_factor = 2
            print(f"  ⚠️  Large resolution detected, downsampling by {downsample_factor}x for faster processing")

        self.bgr_cache = []
        self.depth_colored_cache = []

        # Measure first frame to estimate total time
        frame_start = time.time()
        rgb = self.vision_rgb_array[0]
        if downsample_factor > 1:
            rgb = cv2.resize(rgb, (w // downsample_factor, h // downsample_factor),
                            interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.bgr_cache.append(bgr)

        depth = self.vision_depth_array[0]
        if downsample_factor > 1:
            depth = cv2.resize(depth, (w // downsample_factor, h // downsample_factor),
                              interpolation=cv2.INTER_NEAREST)
        depth_colored = self._normalize_depth(depth)
        self.depth_colored_cache.append(depth_colored)

        first_frame_time = time.time() - frame_start
        estimated_total = first_frame_time * total_frames
        print(f"  First frame took {first_frame_time:.3f}s, estimated total: {estimated_total:.1f}s")

        # Use tqdm for progress bar (start from frame 1 since we already processed frame 0)
        for i in tqdm(range(1, total_frames), desc="  Processing frames", unit="frame", ncols=100):
            # Convert RGB to BGR
            rgb = self.vision_rgb_array[i]

            # Downsample if needed
            if downsample_factor > 1:
                rgb = cv2.resize(rgb, (w // downsample_factor, h // downsample_factor),
                                interpolation=cv2.INTER_AREA)

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self.bgr_cache.append(bgr)

            # Preprocess depth
            depth = self.vision_depth_array[i]

            # Downsample depth if needed
            if downsample_factor > 1:
                depth = cv2.resize(depth, (w // downsample_factor, h // downsample_factor),
                                  interpolation=cv2.INTER_NEAREST)

            depth_colored = self._normalize_depth(depth)
            self.depth_colored_cache.append(depth_colored)

        print(f"  ✅ Preprocessed {len(self.bgr_cache)} vision frames")

    def _get_vision_data_for_frame(self, frame_idx: int):
        """Get vision data for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            Tuple of (bgr, depth_colored) or (None, None) if not available
        """
        if not self.has_vision:
            return None, None

        # Find the closest available vision frame
        vision_indices = self.vision_frame_indices
        if frame_idx in vision_indices:
            # Exact match
            vision_idx = np.where(vision_indices == frame_idx)[0][0]
        else:
            # Find closest
            vision_idx = np.argmin(np.abs(vision_indices - frame_idx))

        # If fast_load mode, process on-demand
        if self.fast_load:
            rgb = self.vision_rgb_array[vision_idx]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = self.vision_depth_array[vision_idx]
            depth_colored = self._normalize_depth(depth)
            return bgr, depth_colored

        # Return preprocessed data from cache
        bgr = self.bgr_cache[vision_idx]
        depth_colored = self.depth_colored_cache[vision_idx]

        return bgr, depth_colored

    def _draw_info_overlay(self, img: np.ndarray, frame_idx: int) -> np.ndarray:
        """Draw information overlay on the image.

        Args:
            img: Input image (BGR)
            frame_idx: Current frame index

        Returns:
            Image with overlay
        """
        if not self.show_info:
            return img

        overlay = img.copy()
        h, w = img.shape[:2]

        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (w - 10, 180), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        # Text parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color = (255, 255, 255)
        y_offset = 30

        # Frame info
        texts = [
            f"Frame: {frame_idx + 1}/{self.num_frames}",
            f"Task: {self.task_name}",
            f"Timestamp: {self.data['system_timestamp'][frame_idx]:.3f}s",
            f"Control Freq: {self.data['system_control_frequency'][frame_idx]:.1f} Hz",
        ]

        # Robot state
        root_pos = self.data['robot_root_position'][frame_idx]
        texts.append(f"Robot Pos: [{root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}]")

        for i, text in enumerate(texts):
            y = y_offset + i * 25
            cv2.putText(img, text, (20, y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Controls hint at bottom
        controls = "Controls: [←→] Navigate | [Space] Play/Pause | [I] Info | [S] Save | [Q] Quit"
        cv2.putText(img, controls, (20, h - 20), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        return img

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """Normalize depth map for visualization.

        Args:
            depth: Depth map (H, W)

        Returns:
            Normalized depth map (H, W, 3) in BGR
        """
        # Clip extreme values
        depth_clipped = np.clip(depth, 0, 10.0)

        # Normalize to 0-255
        depth_norm = (depth_clipped / 10.0 * 255).astype(np.uint8)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        return depth_colored

    def _create_display_image(self, frame_idx: int) -> np.ndarray:
        """Create the display image for a frame.

        Args:
            frame_idx: Frame index

        Returns:
            Display image (BGR)
        """
        # Check cache first
        cache_key = (frame_idx, self.show_info)
        if self.cache_enabled and cache_key in self.frame_cache:
            return self.frame_cache[cache_key]

        bgr, depth_colored = self._get_vision_data_for_frame(frame_idx)

        if bgr is None:
            # No vision data, create placeholder
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No vision data for this frame",
                       (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            display = self._draw_info_overlay(placeholder, frame_idx)
        else:
            # Concatenate horizontally (already preprocessed)
            display = np.hstack([bgr, depth_colored])

            # Add info overlay
            display = self._draw_info_overlay(display, frame_idx)

        # Cache the result (with size limit)
        if self.cache_enabled and len(self.frame_cache) < self.max_cache_size:
            self.frame_cache[cache_key] = display

        return display

    def _save_current_frame(self):
        """Save the current frame as an image file."""
        output_dir = Path(self.npz_path).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)

        filename = f"{self.task_name}_frame_{self.current_frame:05d}.png"
        output_path = output_dir / filename

        display = self._create_display_image(self.current_frame)
        cv2.imwrite(str(output_path), display)

        print(f"💾 Saved frame to: {output_path}")

    def run(self):
        """Run the visualization loop."""
        print(f"\n🎬 Starting visualization...")
        print(f"Controls:")
        print(f"  Left/Right Arrow: Navigate frames")
        print(f"  Space: Play/Pause")
        print(f"  I: Toggle info overlay")
        print(f"  S: Save current frame")
        print(f"  Q/Esc: Quit")

        window_name = f"Recording Viewer - {self.task_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 480)

        while True:
            # Create display image
            display = self._create_display_image(self.current_frame)

            # Show image
            cv2.imshow(window_name, display)

            # Handle keyboard input
            wait_time = int(1000 / self.fps) if self.playing else 0
            key = cv2.waitKey(wait_time) & 0xFF

            if key == ord('q') or key == 27:  # Q or Esc
                break
            elif key == ord(' '):  # Space
                self.playing = not self.playing
                print(f"{'▶️  Playing' if self.playing else '⏸️  Paused'}")
            elif key == ord('i'):  # I
                self.show_info = not self.show_info
                # Clear cache when toggling info overlay
                self.frame_cache.clear()
            elif key == ord('s'):  # S
                self._save_current_frame()
            elif key == 81 or key == 2:  # Left arrow
                self.current_frame = max(0, self.current_frame - 1)
                self.playing = False
            elif key == 83 or key == 3:  # Right arrow
                self.current_frame = min(self.num_frames - 1, self.current_frame + 1)
                self.playing = False

            # Auto-advance if playing
            if self.playing:
                self.current_frame += 1
                if self.current_frame >= self.num_frames:
                    self.current_frame = 0  # Loop

        cv2.destroyAllWindows()
        print("\n👋 Visualization closed")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize recorded data from action_provider_wh_twist2.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_recording.py recording_data/task_1234567890.npz
  python visualize_recording.py recording_data/task_1234567890.npz --fps 60
        """
    )
    parser.add_argument("npz_file", type=str, help="Path to the .npz recording file")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--no-cache", action="store_true", help="Disable frame caching (saves memory)")
    parser.add_argument("--fast-load", action="store_true", help="Skip preprocessing, load frames on-demand (faster startup)")

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.npz_file):
        print(f"❌ Error: File not found: {args.npz_file}")
        sys.exit(1)

    # Create visualizer
    visualizer = RecordingVisualizer(args.npz_file)
    visualizer.fps = args.fps
    if args.no_cache:
        visualizer.cache_enabled = False
        print("⚠️  Frame caching disabled")
    if args.fast_load:
        visualizer.fast_load = True

    # Run visualization
    try:
        visualizer.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
