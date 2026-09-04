"""SIMPLE-specific conversion between dexterous-hand targets and closure."""

from __future__ import annotations

import numpy as np


# Natural decoupled-WBC order: thumb(3), index(2), middle(2).  The teleop
# demonstrations drive each hand along this one-dimensional open/close path.
SIMPLE_LEFT_HAND_CLOSE = np.asarray(
    [-0.5, 0.7, 0.7, -1.5, -1.5, -0.6, -1.5], dtype=np.float32
)
SIMPLE_RIGHT_HAND_CLOSE = np.asarray(
    [-0.5, -0.7, -0.7, 1.5, 1.5, 0.6, 1.5], dtype=np.float32
)
SIMPLE_HAND_CLOSE_POSES = np.stack(
    (SIMPLE_LEFT_HAND_CLOSE, SIMPLE_RIGHT_HAND_CLOSE), axis=0
)


def mjcf_hand_to_wbc(hand_q: np.ndarray) -> np.ndarray:
    """Convert MuJoCo thumb/middle/index order to WBC thumb/index/middle."""
    hand_q = np.asarray(hand_q, dtype=np.float32)
    if hand_q.ndim == 1:
        hand_q = hand_q[None, :]
    if hand_q.ndim != 2 or hand_q.shape[1] != 14:
        raise ValueError(f"MuJoCo hand values must have shape (T, 14), got {hand_q.shape}")
    return np.concatenate(
        (
            hand_q[:, :3],
            hand_q[:, 5:7],
            hand_q[:, 3:5],
            hand_q[:, 7:10],
            hand_q[:, 12:14],
            hand_q[:, 10:12],
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def _hand_matrix(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"{name} must have shape (T, 14), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return values


def source_action_hand_targets(source_action: np.ndarray) -> np.ndarray:
    """Extract 14-D WBC-order hand targets from SIMPLE's 36-D policy action."""
    source_action = np.asarray(source_action, dtype=np.float32)
    if source_action.ndim == 1:
        source_action = source_action[None, :]
    if source_action.ndim != 2 or source_action.shape[1] != 36:
        raise ValueError(
            f"SIMPLE source action must have shape (T, 36), got {source_action.shape}"
        )
    if not np.isfinite(source_action).all():
        raise ValueError("SIMPLE source action contains NaN or Inf")

    # The source policy stores the left hand as thumb, middle, index.  The
    # right-hand block already uses the natural thumb, index, middle order.
    left_source = source_action[:, :7]
    left_wbc = np.concatenate(
        (left_source[:, :3], left_source[:, 5:7], left_source[:, 3:5]), axis=1
    )
    return np.concatenate((left_wbc, source_action[:, 7:14]), axis=1)


def project_hand_closure(
    hand_q: np.ndarray,
    *,
    name: str = "SIMPLE hand target",
    max_relative_residual: float | None = None,
) -> np.ndarray:
    """Project 14 hand joints onto two normalized open/close coordinates."""
    hand_q = _hand_matrix(hand_q, name).reshape(-1, 2, 7)
    denominators = np.sum(SIMPLE_HAND_CLOSE_POSES**2, axis=1)
    closure = np.sum(hand_q * SIMPLE_HAND_CLOSE_POSES[None], axis=2) / denominators
    closure = np.clip(closure, 0.0, 1.0).astype(np.float32)

    if max_relative_residual is not None:
        tolerance = float(max_relative_residual)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("max_relative_residual must be finite and non-negative")
        reconstructed = closure[:, :, None] * SIMPLE_HAND_CLOSE_POSES[None]
        residual = np.linalg.norm(hand_q - reconstructed, axis=2) / np.sqrt(
            denominators[None]
        )
        maximum = float(residual.max())
        if maximum > tolerance:
            raise ValueError(
                f"{name} is not representable by two closure scalars: "
                f"relative residual {maximum:.4f} exceeds {tolerance:.4f}"
            )
    return closure


def hand_targets_from_closure(closure: np.ndarray) -> np.ndarray:
    """Expand normalized left/right closure into 14-D WBC hand targets."""
    closure = np.asarray(closure, dtype=np.float32)
    if closure.ndim == 1:
        closure = closure[None, :]
    if closure.ndim != 2 or closure.shape[1] != 2:
        raise ValueError(f"SIMPLE hand closure must have shape (T, 2), got {closure.shape}")
    if not np.isfinite(closure).all():
        raise ValueError("SIMPLE hand closure contains NaN or Inf")
    if ((closure < -1e-6) | (closure > 1.0 + 1e-6)).any():
        raise ValueError("SIMPLE hand closure must be within [0, 1]")
    return (
        np.clip(closure, 0.0, 1.0)[:, :, None]
        * SIMPLE_HAND_CLOSE_POSES[None]
    ).reshape(-1, 14).astype(np.float32)
