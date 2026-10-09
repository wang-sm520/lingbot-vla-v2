from __future__ import annotations

import logging

import numpy as np


logger = logging.getLogger(__name__)

JOINT_LIMITS = np.asarray(
    [
        [-3.14, 2.09],
        [-2.96, 0.17],
        [-0.087, 3.14],
        [-3.01, 3.01],
        [-1.76, 1.76],
        [-3.01, 3.01],
    ],
    dtype=np.float64,
)
G2_LIMITS = (0.0, 0.072)


def validate_action(action: object, expected_dim: int = 7) -> np.ndarray:
    values = np.array(action, dtype=np.float64, copy=True)
    if values.shape != (expected_dim,):
        raise ValueError(f"Action shape must be ({expected_dim},); received {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Action contains non-finite values.")

    groups = expected_dim // 7
    if expected_dim % 7 != 0:
        raise ValueError(f"AIRBOT action dimension must be a multiple of 7; received {expected_dim}.")
    for group_index in range(groups):
        group_action = values[group_index * 7 : (group_index + 1) * 7]
        clipped_joints = np.clip(group_action[:6], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        for joint_index in np.flatnonzero(group_action[:6] != clipped_joints):
            logger.warning(
                "Clipping joint %s action for group %s from %.6f rad to %.6f rad.",
                joint_index + 1,
                group_index,
                group_action[joint_index],
                clipped_joints[joint_index],
            )
        group_action[:6] = clipped_joints
        if not G2_LIMITS[0] <= group_action[6] <= G2_LIMITS[1]:
            clipped = float(np.clip(group_action[6], *G2_LIMITS))
            logger.warning(
                "Clipping G2 action for group %s from %.6f m to %.6f m.", group_index, group_action[6], clipped
            )
            group_action[6] = clipped
    return values


def interpolate_action(step_length: object, previous: object, target: object) -> np.ndarray:
    previous_values = validate_action(previous)
    target_values = validate_action(target)
    steps = np.asarray(step_length, dtype=np.float64)
    if steps.shape != target_values.shape or np.any(steps <= 0) or not np.isfinite(steps).all():
        raise ValueError(f"step_length must contain {target_values.size} positive finite values.")
    interpolation_count = int(np.max(np.ceil(np.abs(target_values - previous_values) / steps)))
    if interpolation_count <= 1:
        return target_values[np.newaxis, :]
    values = np.linspace(previous_values, target_values, interpolation_count + 1)[1:]
    for value in values:
        validate_action(value)
    return values
