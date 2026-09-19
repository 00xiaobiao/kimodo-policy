from __future__ import annotations

import json
from pathlib import Path


EGO_VIEW_FEATURE = "observation.images.ego_view"


def _normalize_shape(ego_view_shape) -> list[int] | None:
    if ego_view_shape is None:
        return None

    shape = [int(dim) for dim in ego_view_shape]
    if len(shape) != 3:
        raise ValueError(f"Expected ego-view image shape to be HWC, got {tuple(shape)}")
    return shape


def set_ego_view_feature_shape(features: dict, ego_view_shape) -> None:
    """Match the LeRobot video metadata to the actual egocentric image shape."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    features[EGO_VIEW_FEATURE]["shape"] = shape


def validate_existing_ego_view_feature_shape(save_dir: str | Path, ego_view_shape) -> None:
    """Refuse to append 480p frames to a dataset initialized with old 360p metadata."""
    shape = _normalize_shape(ego_view_shape)
    if shape is None:
        return

    info_path = Path(save_dir) / "meta" / "info.json"
    if not info_path.exists():
        return

    with open(info_path, "r") as f:
        info = json.load(f)

    existing_shape = info.get("features", {}).get(EGO_VIEW_FEATURE, {}).get("shape")
    if existing_shape is None:
        return

    existing_shape = [int(dim) for dim in existing_shape]
    if existing_shape != shape:
        raise ValueError(
            f"Existing LeRobot dataset at {save_dir} declares {EGO_VIEW_FEATURE} "
            f"shape {existing_shape}, but the current ego-view frame is {shape}. "
            "Use a clean --save-dir or remove the old dataset before recording."
        )
