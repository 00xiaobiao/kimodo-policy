# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
public camera configuration
include the basic configuration for different types of cameras, support scene-specific parameter customization
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

FRONT_CAMERA_WIDTH = 640
FRONT_CAMERA_HEIGHT = 480
WRIST_CAMERA_WIDTH = 640
WRIST_CAMERA_HEIGHT = 480
WORLD_CAMERA_WIDTH = 1920
WORLD_CAMERA_HEIGHT = 1080


def _depth_enabled_from_env() -> bool:
    value = str(os.environ.get("ENABLE_DEPTH", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


@configclass
class CameraBaseCfg:
    """camera base configuration class
    
    provide the default configuration for different types of cameras, support scene-specific parameter customization
    """
    
    @classmethod
    def get_camera_config(
        cls,
        prim_path: str = "/World/envs/env_.*/Robot/d435_link/front_cam",
        update_period: float = 0.01,
        height: int = 480,
        width: int = 640,
        focal_length: float = 7.6,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 20.0,
        clipping_range: tuple = (0.1, 1.0e5),
        pos_offset: tuple = (0, 0.0, 0),
        rot_offset: tuple = (0.5, -0.5, 0.5, -0.5),
        convention: str = "ros",
        data_types: list = None
    ) -> CameraCfg:
        """get the front camera configuration
        
        Args:
            prim_path: the path of the camera in the scene
            update_period: update period (seconds)
            height: image height (pixels)
            width: image width (pixels)
            focal_length: focal length
            focus_distance: focus distance
            horizontal_aperture: horizontal aperture
            clipping_range: clipping range (near clipping plane, far clipping plane)
            pos_offset: position offset (x, y, z)
            rot_offset: rotation offset quaternion in (w, x, y, z)
            convention: camera orientation convention: "ros", "opengl", or "world"
            data_types: data type list
            
        Returns:
            CameraCfg: camera configuration
        """
        if data_types is None:
            data_types = ["rgb"]
            if _depth_enabled_from_env():
                data_types.append("distance_to_image_plane")

        return CameraCfg(
            prim_path=prim_path,
            update_period=update_period,
            height=height,
            width=width,
            data_types=data_types,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal_length,
                focus_distance=focus_distance,
                horizontal_aperture=horizontal_aperture,
                clipping_range=clipping_range
            ),
            offset=CameraCfg.OffsetCfg(
                pos=pos_offset,
                rot=rot_offset,
                convention=convention
            )
        )

    @classmethod
    def get_world_camera_config(
        cls,
        update_period: float = 0.01,
        height: int = WORLD_CAMERA_HEIGHT,
        width: int = WORLD_CAMERA_WIDTH,
        focal_length: float = 12,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 27,
        clipping_range: tuple = (0.1, 1.0e5),
        pos_offset: tuple = (0, 0.0, 0),
        rot_offset: tuple = (0.5, -0.5, 0.5, -0.5),
        convention: str = "opengl",
        data_types: list = None
    ) -> CameraCfg:
        """get the third-person world camera configuration"""
        return cls.get_camera_config(
            prim_path="/World/PerspectiveCamera",
            update_period=update_period,
            height=height,
            width=width,
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=horizontal_aperture,
            clipping_range=clipping_range,
            pos_offset=pos_offset,
            rot_offset=rot_offset,
            convention=convention,
            data_types=data_types,
        )
    



@configclass
class CameraPresets:
    """camera preset configuration collection
    
    include the common camera configuration preset for different scenes
    """
    
    @classmethod
    def g1_front_camera(cls) -> CameraCfg:
        """front camera configuration"""
        return CameraBaseCfg.get_camera_config(
            height=FRONT_CAMERA_HEIGHT,
            width=FRONT_CAMERA_WIDTH,
        )
    @classmethod
    def g1_world_camera(
        cls,
        *,
        pos_offset: tuple = (-2.01826, 3.33365, 1.26749),
        rot_offset: tuple = (0.85990, 0.24942, -0.43352, -0.10206),
        height: int = WORLD_CAMERA_HEIGHT,
        width: int = WORLD_CAMERA_WIDTH,
        focal_length: float = 12,
        horizontal_aperture: float = 27,
        convention: str = "opengl",
    ) -> CameraCfg:
        """world camera configuration (third-person view, fixed in world space)"""
        return CameraBaseCfg.get_world_camera_config(
            pos_offset=pos_offset,
            rot_offset=rot_offset,
            height=height,
            width=width,
            focal_length=focal_length,
            horizontal_aperture=horizontal_aperture,
            convention=convention,
        )
    
    @classmethod
    def left_gripper_wrist_camera(cls) -> CameraCfg:
        """left wrist camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/left_hand_base_link/left_wrist_camera",
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            update_period=0.01,
            data_types=["rgb", "distance_to_image_plane"],
            focal_length=12,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
            pos_offset=(0.02541028, 0.0076, 0.135),
            rot_offset=(-0.50262, 0.86451, 0, 0),
        )
    @classmethod
    def right_gripper_wrist_camera(cls) -> CameraCfg:
        """right wrist camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/right_hand_base_link/right_wrist_camera",
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            update_period=0.01,
            data_types=["rgb", "distance_to_image_plane"],
            focal_length=12,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
            pos_offset=(-0.02541028, 0.0076, 0.135),
            rot_offset=(-0.50262, 0.86451, 0, 0),
        ) 
    @classmethod
    def left_dex3_wrist_camera(cls) -> CameraCfg:
        """left wrist camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/left_hand_camera_base_link/left_wrist_camera",
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            update_period=0.01,
            data_types=["rgb", "distance_to_image_plane"],
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
            pos_offset=(-0.04012, -0.07441 ,0.15711),
            rot_offset=(0.00539,0.86024,0.0424, 0.50809),
        )
    @classmethod
    def right_dex3_wrist_camera(cls) -> CameraCfg:
        """right wrist camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/right_hand_camera_base_link/right_wrist_camera",
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            update_period=0.01,
            data_types=["rgb", "distance_to_image_plane"],
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
            pos_offset=(-0.04012, 0.07441 ,0.15711),
            rot_offset=(0.00539,0.86024,0.0424, 0.50809),
        ) 
    
    @classmethod
    def left_inspire_wrist_camera(cls) -> CameraCfg:
        """left wrist camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/left_hand_camera_base_link/left_wrist_camera",
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            update_period=0.01,
            data_types=["rgb", "distance_to_image_plane"],
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
            pos_offset=(-0.04012, -0.07441 ,0.15711),
            rot_offset=(0.00539,0.86024,0.0424, 0.50809),
        )
    @classmethod
    def right_inspire_wrist_camera(cls) -> CameraCfg:
        """right wrist camera configuration"""
        return CameraBaseCfg.get_camera_config(
            prim_path="/World/envs/env_.*/Robot/right_hand_camera_base_link/right_wrist_camera",
            height=WRIST_CAMERA_HEIGHT,
            width=WRIST_CAMERA_WIDTH,
            update_period=0.01,
            data_types=["rgb", "distance_to_image_plane"],
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
            pos_offset=(-0.04012, 0.07441 ,0.15711),
            rot_offset=(0.00539,0.86024,0.0424, 0.50809),
        ) 
