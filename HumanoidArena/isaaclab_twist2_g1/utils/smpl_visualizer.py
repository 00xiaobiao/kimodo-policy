"""
SMPL Visualizer for rendering SMPL skeleton from state.

This module provides a simple visualizer for SMPL skeleton using matplotlib
or optionally pyrender for higher quality rendering.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Try to import smplx, but make it optional
try:
    import smplx
    import torch
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    print("Warning: smplx not available. SMPL visualization will be limited.")


# SMPL skeleton connections (parent-child relationships)
# Based on SMPL-X joint hierarchy
SMPL_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),          # pelvis to legs and spine
    (1, 4), (2, 5),                   # legs
    (4, 7), (5, 8),                   # knees to ankles
    (7, 10), (8, 11),                 # ankles to feet
    (3, 6), (6, 9),                   # spine to chest to neck
    (9, 12), (9, 13), (9, 14),       # neck to head and shoulders
    (12, 15), (13, 16), (14, 17),    # shoulders to elbows
    (15, 18), (16, 19),               # elbows to wrists
    (18, 20), (18, 21),               # left wrist connections
    (19, 22), (19, 23),               # right wrist connections
]

# Simplified connections for 22 joints (SMPL body model)
SMPL_BODY_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3),          # pelvis to legs and spine
    (1, 4), (2, 5),                   # hips to knees
    (4, 7), (5, 8),                   # knees to ankles
    (7, 10), (8, 11),                 # ankles to feet
    (3, 6), (6, 9),                   # spine through chest to neck
    (9, 12), (9, 13), (9, 14),       # neck to jaw/shoulders
    (12, 15), (13, 16), (14, 17),    # shoulders to elbows
    (15, 18), (16, 19),               # elbows to wrists
    (18, 20), (19, 21),               # wrists to hands
]


class SMPLVisualizer:
    """
    Visualizer for SMPL skeleton using matplotlib.
    """

    def __init__(self, smplx_model_path=None, resolution=(640, 480), use_smplx=True):
        """
        Initialize SMPL visualizer.

        Args:
            smplx_model_path: path to SMPL-X model files (optional)
            resolution: (width, height) for rendered images
            use_smplx: if True, use smplx library for full FK; else use simplified rendering
        """
        self.resolution = resolution
        self.use_smplx = use_smplx and SMPLX_AVAILABLE

        # Initialize SMPL-X model if available and requested
        self.smplx_model = None
        if self.use_smplx and smplx_model_path is not None:
            try:
                self.smplx_model = smplx.create(
                    model_path=smplx_model_path,
                    model_type='smplx',
                    gender='neutral',
                    use_pca=False,
                    batch_size=1
                )
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.smplx_model = self.smplx_model.to(self.device)
                print(f"SMPL-X model loaded on {self.device}")
            except Exception as e:
                print(f"Failed to load SMPL-X model: {e}")
                self.smplx_model = None
                self.use_smplx = False

        # Setup matplotlib figure
        self.fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')

    def render(self, smpl_state, return_joints=False):
        """
        Render SMPL skeleton from state.

        Args:
            smpl_state: (75,) SMPL state array
                       [:63] body_pose rotation vectors (21 joints × 3)
                       [63:66] root_orient rotation vector
                       [66:72] hand_pose (simplified)
                       [72:75] trans
            return_joints: if True, also return joint positions

        Returns:
            img: (H, W, 3) RGB image
            joints: (optional) (N, 3) joint positions if return_joints=True
        """
        if self.use_smplx and self.smplx_model is not None:
            return self._render_with_smplx(smpl_state, return_joints)
        else:
            return self._render_simple(smpl_state, return_joints)

    def _render_with_smplx(self, smpl_state, return_joints=False):
        """Render using SMPL-X forward kinematics."""
        # Parse SMPL state
        body_pose = smpl_state[:63]      # (63,)
        root_orient = smpl_state[63:66]  # (3,)
        trans = smpl_state[72:75]        # (3,)

        # Run SMPL-X forward kinematics
        with torch.no_grad():
            body_pose_tensor = torch.from_numpy(body_pose).float().reshape(1, 21, 3).to(self.device)
            root_orient_tensor = torch.from_numpy(root_orient).float().reshape(1, 3).to(self.device)
            trans_tensor = torch.from_numpy(trans).float().reshape(1, 3).to(self.device)

            smplx_output = self.smplx_model(
                global_orient=root_orient_tensor,
                body_pose=body_pose_tensor.reshape(1, 63),
                transl=trans_tensor,
                left_hand_pose=torch.zeros(1, 45).float().to(self.device),
                right_hand_pose=torch.zeros(1, 45).float().to(self.device),
                jaw_pose=torch.zeros(1, 3).float().to(self.device),
                leye_pose=torch.zeros(1, 3).float().to(self.device),
                reye_pose=torch.zeros(1, 3).float().to(self.device),
                return_full_pose=False
            )

            joints = smplx_output.joints[0].detach().cpu().numpy()  # (N, 3)

        # Render skeleton
        img = self._render_skeleton(joints)

        if return_joints:
            return img, joints
        return img

    def _render_simple(self, smpl_state, return_joints=False):
        """
        Simple rendering without full SMPL-X FK.
        Just visualizes root position.
        """
        trans = smpl_state[72:75]

        # Create a simple marker at root position
        joints = np.array([trans])  # Just root

        img = self._render_skeleton(joints, connections=[])

        if return_joints:
            return img, joints
        return img

    def _render_skeleton(self, joints, connections=None):
        """
        Render skeleton from joint positions.

        Args:
            joints: (N, 3) joint positions
            connections: list of (parent, child) tuples, or None for default

        Returns:
            img: (H, W, 3) RGB image
        """
        if connections is None:
            connections = SMPL_BODY_CONNECTIONS

        # Clear previous plot
        self.ax.cla()

        # Plot joints
        self.ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                       c='red', marker='o', s=50)

        # Plot bones
        for parent, child in connections:
            if parent < len(joints) and child < len(joints):
                xs = [joints[parent, 0], joints[child, 0]]
                ys = [joints[parent, 1], joints[child, 1]]
                zs = [joints[parent, 2], joints[child, 2]]
                self.ax.plot(xs, ys, zs, 'b-', linewidth=2)

        # Set viewing angle and limits
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Auto-scale based on joints
        if len(joints) > 0:
            center = joints.mean(axis=0)
            scale = 1.5  # meters
            self.ax.set_xlim([center[0] - scale, center[0] + scale])
            self.ax.set_ylim([center[1] - scale, center[1] + scale])
            self.ax.set_zlim([center[2] - scale, center[2] + scale])
        else:
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([-1, 1])
            self.ax.set_zlim([0, 2])

        # Set viewing angle (adjust for better view)
        self.ax.view_init(elev=20, azim=45)

        # Convert plot to image
        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.resolution[1], self.resolution[0], 3)

        return img

    def close(self):
        """Clean up resources."""
        plt.close(self.fig)


class SimpleSMPLVisualizer:
    """
    Even simpler visualizer that just shows a colored box representing the SMPL state.
    Useful as a fallback when smplx is not available.
    """

    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution

    def render(self, smpl_state, return_joints=False):
        """
        Render a simple visualization (colored box with text).

        Args:
            smpl_state: (75,) SMPL state
            return_joints: ignored

        Returns:
            img: (H, W, 3) RGB image
        """
        img = np.ones((self.resolution[1], self.resolution[0], 3), dtype=np.uint8) * 255

        # Draw colored background
        trans = smpl_state[72:75]
        # Normalize height to color (0.5m to 1.5m -> 0 to 255)
        height = np.clip((trans[2] - 0.5) * 255, 0, 255).astype(np.uint8)
        img[:, :, 2] = height  # Blue channel

        # Add text
        import cv2
        text = f"SMPL State"
        text2 = f"Height: {trans[2]:.2f}m"
        cv2.putText(img, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, text2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if return_joints:
            return img, np.array([trans])
        return img

    def close(self):
        pass


def create_smpl_visualizer(smplx_model_path=None, resolution=(640, 480), use_simple=False):
    """
    Factory function to create appropriate SMPL visualizer.

    Args:
        smplx_model_path: path to SMPL-X models (optional)
        resolution: (width, height) for images
        use_simple: if True, use SimpleSMPLVisualizer

    Returns:
        SMPLVisualizer or SimpleSMPLVisualizer instance
    """
    if use_simple or not SMPLX_AVAILABLE:
        return SimpleSMPLVisualizer(resolution=resolution)
    else:
        return SMPLVisualizer(smplx_model_path=smplx_model_path, resolution=resolution)


if __name__ == "__main__":
    # Quick test
    print("Testing SMPL visualizer...")

    # Create a simple default SMPL state
    smpl_state = np.zeros(75, dtype=np.float32)
    smpl_state[74] = 1.0  # Set height to 1.0m

    # Test simple visualizer
    visualizer = SimpleSMPLVisualizer(resolution=(640, 480))
    img = visualizer.render(smpl_state)
    print(f"Simple visualizer output shape: {img.shape}")
    assert img.shape == (480, 640, 3), "Output shape should match resolution"
    visualizer.close()

    # Test full visualizer (will use simple mode if smplx not available)
    visualizer = create_smpl_visualizer(resolution=(640, 480))
    img = visualizer.render(smpl_state)
    print(f"Full visualizer output shape: {img.shape}")
    assert img.shape == (480, 640, 3), "Output shape should match resolution"
    visualizer.close()

    print("SMPL visualizer tests passed!")
