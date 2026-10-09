import numpy as np

from tools.prepare_airbot_data import action_windows


def test_windows_use_current_state_and_repeat_only_the_episode_final_action():
    states = np.zeros((3, 7), dtype=np.float32)
    states[:, 0] = [1, 2, 3]
    actions = np.zeros((3, 7), dtype=np.float32)
    actions[:, 0] = [11, 12, 13]
    actions[:, 6] = [0.01, 0.02, 0.03]
    windows = action_windows(actions, states, 4)
    np.testing.assert_allclose(windows[:, :, 0], [[10, 11, 12, 12], [10, 11, 11, 11], [10, 10, 10, 10]])
    np.testing.assert_allclose(windows[:, :, 6], [[0.01, 0.02, 0.03, 0.03], [0.02, 0.03, 0.03, 0.03], [0.03] * 4])
    np.testing.assert_array_equal(actions[:, 0], [11, 12, 13])
