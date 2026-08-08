"""
Video Recorder for multi-view recording during OpenPI verification.

Records and combines:
- First-person view (from robot camera)
- Third-person view (world camera)
- SMPL visualization
"""

import os
import numpy as np
from pathlib import Path
import cv2
from typing import Optional


class VideoRecorder:
    """
    Multi-view video recorder that combines multiple video streams.
    """

    def __init__(self, save_dir: str, fps: int = 30, enable_smpl_vis: bool = True,
                 smpl_visualizer=None):
        """
        Initialize video recorder.

        Args:
            save_dir: directory to save videos
            fps: frames per second
            enable_smpl_vis: if True, include SMPL visualization in output
            smpl_visualizer: SMPLVisualizer instance (optional, created if None and enable_smpl_vis=True)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.enable_smpl_vis = enable_smpl_vis

        # Frame storage
        self.first_person_frames = []
        self.third_person_frames = []
        self.smpl_states = []  # Store SMPL states instead of pre-rendered frames

        # SMPL visualizer
        self.smpl_visualizer = smpl_visualizer
        if enable_smpl_vis and smpl_visualizer is None:
            from .smpl_visualizer import create_smpl_visualizer
            self.smpl_visualizer = create_smpl_visualizer(use_simple=True, resolution=(640, 480))

        # Video writer (created when first frame is added)
        self.video_writer = None
        self.canvas_size = None

    def add_frame(self, first_person_img: Optional[np.ndarray] = None,
                  third_person_img: Optional[np.ndarray] = None,
                  smpl_state: Optional[np.ndarray] = None):
        """
        Add a frame to the recording.

        Args:
            first_person_img: (H, W, 3) RGB image from robot camera
            third_person_img: (H, W, 3) RGB image from world camera
            smpl_state: (75,) SMPL state array
        """
        # Store frames
        if first_person_img is not None:
            # Ensure uint8 format
            if first_person_img.dtype != np.uint8:
                first_person_img = (first_person_img * 255).astype(np.uint8) if first_person_img.max() <= 1.0 else first_person_img.astype(np.uint8)
            self.first_person_frames.append(first_person_img.copy())
        else:
            # Create placeholder if not provided
            if len(self.first_person_frames) > 0:
                placeholder = np.zeros_like(self.first_person_frames[0])
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            self.first_person_frames.append(placeholder)

        if third_person_img is not None:
            if third_person_img.dtype != np.uint8:
                third_person_img = (third_person_img * 255).astype(np.uint8) if third_person_img.max() <= 1.0 else third_person_img.astype(np.uint8)
            self.third_person_frames.append(third_person_img.copy())
        else:
            if len(self.third_person_frames) > 0:
                placeholder = np.zeros_like(self.third_person_frames[0])
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            self.third_person_frames.append(placeholder)

        if smpl_state is not None and self.enable_smpl_vis:
            self.smpl_states.append(smpl_state.copy())
        elif self.enable_smpl_vis:
            # Placeholder SMPL state
            self.smpl_states.append(np.zeros(75, dtype=np.float32))

    def save(self, output_path: Optional[str] = None, name: str = "output"):
        """
        Render SMPL frames and save combined video.

        Args:
            output_path: full path to output video file (optional)
            name: base name for video if output_path not specified
        """
        if len(self.first_person_frames) == 0:
            print("No frames to save!")
            return

        # Determine output path
        if output_path is None:
            output_path = self.save_dir / f"{name}.mp4"
        else:
            output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Rendering and saving video to {output_path}...")

        # Get frame dimensions
        h, w = self.first_person_frames[0].shape[:2]

        # Resize all frames to same height if needed
        target_height = h
        resized_first_person = self._resize_frames(self.first_person_frames, target_height)
        resized_third_person = self._resize_frames(self.third_person_frames, target_height)

        # Render SMPL frames if enabled
        smpl_frames = []
        if self.enable_smpl_vis and self.smpl_visualizer is not None:
            print(f"Rendering {len(self.smpl_states)} SMPL frames...")
            for i, smpl_state in enumerate(self.smpl_states):
                smpl_img = self.smpl_visualizer.render(smpl_state)
                # Resize to target height
                smpl_h, smpl_w = smpl_img.shape[:2]
                if smpl_h != target_height:
                    scale = target_height / smpl_h
                    smpl_img = cv2.resize(smpl_img, (int(smpl_w * scale), target_height))
                smpl_frames.append(smpl_img)
            print("SMPL rendering complete.")

        # Calculate canvas size
        first_w = resized_first_person[0].shape[1]
        third_w = resized_third_person[0].shape[1]
        if len(smpl_frames) > 0:
            smpl_w = smpl_frames[0].shape[1]
            canvas_width = first_w + third_w + smpl_w
        else:
            canvas_width = first_w + third_w

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (canvas_width, target_height))

        # Combine and write frames
        num_frames = len(resized_first_person)
        print(f"Writing {num_frames} frames...")
        for i in range(num_frames):
            canvas = np.zeros((target_height, canvas_width, 3), dtype=np.uint8)

            # Place first-person view
            fp_img = resized_first_person[i]
            canvas[:, :first_w] = cv2.cvtColor(fp_img, cv2.COLOR_RGB2BGR)

            # Place third-person view
            tp_img = resized_third_person[i]
            canvas[:, first_w:first_w+third_w] = cv2.cvtColor(tp_img, cv2.COLOR_RGB2BGR)

            # Place SMPL visualization if enabled
            if len(smpl_frames) > 0 and i < len(smpl_frames):
                smpl_img = smpl_frames[i]
                canvas[:, first_w+third_w:] = cv2.cvtColor(smpl_img, cv2.COLOR_RGB2BGR)

            writer.write(canvas)

        writer.release()
        print(f"Video saved successfully: {output_path}")
        print(f"  - Resolution: {canvas_width}x{target_height}")
        print(f"  - FPS: {self.fps}")
        print(f"  - Frames: {num_frames}")
        print(f"  - Duration: {num_frames/self.fps:.2f}s")

    def _resize_frames(self, frames, target_height):
        """Resize frames to target height while maintaining aspect ratio."""
        resized = []
        for frame in frames:
            h, w = frame.shape[:2]
            if h != target_height:
                scale = target_height / h
                new_w = int(w * scale)
                resized_frame = cv2.resize(frame, (new_w, target_height))
                resized.append(resized_frame)
            else:
                resized.append(frame)
        return resized

    def clear(self):
        """Clear all stored frames."""
        self.first_person_frames = []
        self.third_person_frames = []
        self.smpl_states = []
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def close(self):
        """Clean up resources."""
        if self.video_writer is not None:
            self.video_writer.release()
        if self.smpl_visualizer is not None:
            self.smpl_visualizer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class SimpleVideoRecorder:
    """
    Simple single-stream recorder that writes frames incrementally to disk.
    This avoids holding an entire episode video in RAM during persistent eval.
    """

    def __init__(self, save_path: str, fps: int = 30):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frames = []  # compatibility sentinel for legacy callers
        self.writer = None
        self.frame_count = 0
        self._frame_size = None

    def _ensure_writer(self, img: np.ndarray):
        h, w = img.shape[:2]
        size = (w, h)
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(str(self.save_path), fourcc, self.fps, size)
            self._frame_size = size
        elif self._frame_size != size:
            img = cv2.resize(img, self._frame_size)
        return img

    def add_frame(self, img: np.ndarray):
        """Add a frame to the recording and write it immediately."""
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        img = self._ensure_writer(img)
        self.writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if self.frame_count == 0:
            self.frames = [True]
        self.frame_count += 1

    def save(self, output_path: Optional[str] = None):
        """Finalize the writer and optionally move the temp video to a final path."""
        if self.frame_count == 0:
            print("No frames to save!")
            return
        self.close()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path != self.save_path:
                if output_path.exists():
                    output_path.unlink()
                os.replace(self.save_path, output_path)
                self.save_path = output_path
        print(f"Video saved: {self.save_path} (frames={self.frame_count})")

    def clear(self):
        self.frames = []
        self.frame_count = 0
        self._frame_size = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


if __name__ == "__main__":
    # Quick test
    print("Testing VideoRecorder...")

    # Create test frames
    num_frames = 10
    h, w = 480, 640

    first_person_frames = []
    third_person_frames = []
    smpl_states = []

    for i in range(num_frames):
        # Create colored test frames
        fp_frame = np.ones((h, w, 3), dtype=np.uint8) * (255 * i // num_frames)
        fp_frame[:, :, 0] = 255  # Red channel
        first_person_frames.append(fp_frame)

        tp_frame = np.ones((h, w, 3), dtype=np.uint8) * (255 * i // num_frames)
        tp_frame[:, :, 1] = 255  # Green channel
        third_person_frames.append(tp_frame)

        # Create test SMPL state
        smpl_state = np.zeros(75, dtype=np.float32)
        smpl_state[74] = 0.9 + 0.1 * np.sin(2 * np.pi * i / num_frames)  # Varying height
        smpl_states.append(smpl_state)

    # Test recorder
    recorder = VideoRecorder(save_dir="./test_videos", fps=10, enable_smpl_vis=True)

    for i in range(num_frames):
        recorder.add_frame(
            first_person_img=first_person_frames[i],
            third_person_img=third_person_frames[i],
            smpl_state=smpl_states[i]
        )

    recorder.save(name="test_output")
    recorder.close()

    print("VideoRecorder test complete!")
