import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.airbot_play_lingbot_inference import get_frame, infer_chunk, make_observation


def test_stale_and_missing_camera_timestamps_are_rejected():
    packet = {"data": np.zeros((8, 8, 3), dtype=np.uint8), "t": time.time_ns() - 3_000_000_000}
    observation = {"base/color/image_raw": packet}
    with pytest.raises(RuntimeError, match="stale"):
        get_frame(observation, "base")
    del packet["t"]
    with pytest.raises(RuntimeError, match="timestamp"):
        get_frame(observation, "base")


def test_sensor_feedback_does_not_change_command_limits():
    image = {"data": np.zeros((8, 8, 3), dtype=np.uint8), "t": time.time_ns()}
    operator = SimpleNamespace(
        capture_observation=lambda: {"base/color/image_raw": image, "wrist/color/image_raw": image},
        get_qpos=lambda observation: [0, 0, 0, 0, 0, 0, -0.000004],
    )
    observation = make_observation(operator, ("base", "wrist"), "can task")
    assert observation["task"] == "can task"
    assert observation["observation.state"][6] < 0
    policy = SimpleNamespace(infer=lambda observation: {"action": np.tile(observation["observation.state"], (50, 1))})
    actions = infer_chunk(policy, observation, 30)
    np.testing.assert_array_equal(actions[:, 6], 0.0)
    assert observation["observation.state"][6] < 0


def test_prediction_returns_clipped_gripper_commands():
    prediction = np.zeros((50, 7), dtype=np.float64)
    prediction[:, 6] = np.linspace(-0.01, 0.08, 50)
    original = prediction.copy()
    prediction.setflags(write=False)
    policy = SimpleNamespace(infer=lambda observation: {"action": prediction})
    actions = infer_chunk(policy, {}, 30)
    assert actions.shape == (50, 7)
    np.testing.assert_array_equal(actions[:, :6], prediction[:, :6])
    np.testing.assert_array_equal(actions[:, 6], np.clip(original[:, 6], 0.0, 0.072))
    np.testing.assert_array_equal(prediction, original)


def test_entire_prediction_is_clipped_before_any_execution():
    actions = np.zeros((50, 7))
    actions[0] = [-0.07152837, 0.19720805, -0.04924746, 0.91055614, -0.01928687, -0.97190601, 0.073511]
    actions[-1, 1] = -3.1
    original = actions.copy()
    actions.setflags(write=False)
    policy = SimpleNamespace(infer=lambda observation: {"action": actions})
    actual = infer_chunk(policy, {}, 30)
    assert actual[0, 1] == 0.17
    assert actual[0, 6] == 0.072
    assert actual[-1, 1] == -2.96
    np.testing.assert_array_equal(actions, original)


def test_invalid_predictions_are_rejected_before_any_execution():
    actions = np.zeros((50, 7))
    actions[-1, 1] = np.nan
    policy = SimpleNamespace(infer=lambda observation: {"action": actions})
    with pytest.raises(ValueError, match="non-finite"):
        infer_chunk(policy, {}, 30)
    policy.infer = lambda observation: {"action": actions[:5]}
    with pytest.raises(ValueError, match="less than"):
        infer_chunk(policy, {}, 30)
