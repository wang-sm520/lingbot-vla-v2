import numpy as np
import pytest

from deploy.airbot_play_safety import interpolate_action, validate_action


def test_validate_action_accepts_airbot_play_target() -> None:
    action = validate_action([0, -1, 0.5, 0, 0, 0, 0.04])
    assert action.shape == (7,)
    assert action.dtype == np.float64


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_validate_action_rejects_non_finite_joint_values(value) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_action([0, 0, 0, 0, 0, value, 0.04])


def test_reported_joint_target_is_clipped_without_changing_input() -> None:
    original = np.asarray([-0.07152837, 0.19720805, -0.04924746, 0.91055614, -0.01928687, -0.97190601, 0.073511])
    expected = original.copy()
    expected[1] = 0.17
    expected[6] = 0.072
    original.setflags(write=False)
    action = validate_action(original)
    np.testing.assert_array_equal(action, expected)
    assert original[1] == 0.19720805
    assert original[6] == 0.073511


def test_joint_clipping_applies_to_both_limits_and_each_arm_group() -> None:
    action = validate_action([-4, -4, -1, -4, -3, -4, -0.01, 3, 1, 4, 4, 2, 4, 0.08], expected_dim=14)
    expected = [-3.14, -2.96, -0.087, -3.01, -1.76, -3.01, 0.0, 2.09, 0.17, 3.14, 3.01, 1.76, 3.01, 0.072]
    np.testing.assert_array_equal(action, expected)


@pytest.mark.parametrize("opening, expected", [(-0.01, 0.0), (0.08, 0.072), (0.0, 0.0), (0.072, 0.072), (0.04, 0.04)])
def test_gripper_clips_to_limits_without_changing_input(opening, expected) -> None:
    original = np.asarray([0, -1, 0.5, 0, 0, 0, opening], dtype=np.float64)
    original.setflags(write=False)
    action = validate_action(original)
    np.testing.assert_array_equal(action[:6], original[:6])
    assert action[6] == expected
    assert original[6] == opening


@pytest.mark.parametrize("opening", [np.nan, np.inf, -np.inf])
def test_non_finite_gripper_commands_are_rejected(opening) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_action([0, 0, 0, 0, 0, 0, opening])


def test_gripper_clipping_applies_to_each_arm_group() -> None:
    action = validate_action([0] * 6 + [-0.01] + [0] * 6 + [0.08], expected_dim=14)
    np.testing.assert_array_equal(action[[6, 13]], [0.0, 0.072])


@pytest.mark.parametrize("target, expected", [(0.19720805, 0.17), (-3.1, -2.96)])
def test_interpolation_uses_clipped_joint_target(target, expected) -> None:
    previous = np.asarray([0] * 6 + [0.04])
    step_length = np.asarray([0.01] * 6 + [0.005])
    commands = interpolate_action(step_length, previous, [0, target, 0, 0, 0, 0, 0.04])
    assert commands[-1, 1] == expected
    assert np.all((commands[:, 1] >= -2.96) & (commands[:, 1] <= 0.17))
    changes = np.abs(np.diff(np.vstack([previous, commands]), axis=0))
    assert np.all(changes <= step_length + 1e-12)


def test_interpolation_stays_within_valid_limits() -> None:
    values = interpolate_action([0.01] * 6 + [0.005], [0] * 6 + [0.04], [0.04] * 6 + [0.06])
    assert values.shape[1] == 7
    assert np.isfinite(values).all()


@pytest.mark.parametrize("opening, expected", [(-0.01, 0.0), (0.08, 0.072)])
def test_interpolation_uses_clipped_gripper_target(opening, expected) -> None:
    previous = np.asarray([0] * 6 + [0.04])
    step_length = np.asarray([0.01] * 6 + [0.005])
    commands = interpolate_action(step_length, previous, [0] * 6 + [opening])
    assert commands[-1, 6] == expected
    assert np.all((commands[:, 6] >= 0) & (commands[:, 6] <= 0.072))
    changes = np.abs(np.diff(np.vstack([previous, commands]), axis=0))
    assert np.all(changes <= step_length + 1e-12)
