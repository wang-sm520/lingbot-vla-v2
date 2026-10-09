from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from deploy.airbot_play_safety import interpolate_action, validate_action


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENPI_ROOT = REPO_ROOT.parent / "pi0.5/openpi_v0.2.0"


def load_airbot_runtime(openpi_root: Path):
    airbot_examples = openpi_root / "examples" / "airbot"
    data_collection = openpi_root.parent / "airbot-data-5.1.6.7" / "data-collection"
    for path in (airbot_examples, data_collection):
        if not path.is_dir():
            raise FileNotFoundError(f"AIRBOT runtime source directory does not exist: {path}")
        sys.path.insert(0, str(path))

    from airbot_arm import SpeedProfile
    from airbot_data_collection.basis import SystemMode
    from play_operator_ah import RobotAH
    from robot_config import RobotAHConfig

    class FailClosedRobotAH(RobotAH):
        def _validate_action(self, action):
            expected_dim = 7 * len(self.robots)
            return validate_action(action, expected_dim=expected_dim)

    return FailClosedRobotAH, RobotAHConfig, SystemMode, SpeedProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a LingBot-VLA 2.0 policy on one AIRBOT Play arm.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--robot-interface", required=True)
    parser.add_argument("--control-frequency", type=int)
    parser.add_argument("--head-camera", default="rtsp://192.168.168.168:8554/front")
    parser.add_argument("--wrist-camera", default="realsense://260322275842")
    parser.add_argument("--model-repo", default=str(REPO_ROOT))
    parser.add_argument("--openpi-root", default=os.environ.get("OPENPI_AIRBOT_ROOT", str(DEFAULT_OPENPI_ROOT)))
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--step-rate", type=int, default=25)
    parser.add_argument("--chunk-size-execute", type=int, default=30)
    parser.add_argument("--step-length", nargs=7, type=float, default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005])
    parser.add_argument("--reset-action", nargs=7, type=float, required=True)
    parser.add_argument("--camera-widths", nargs=2, type=int, default=[1920, 848])
    parser.add_argument("--camera-heights", nargs=2, type=int, default=[1080, 480])
    parser.add_argument("--camera-fps", nargs=2, type=int, default=[25, 30])
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--use-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--num-inference-steps", type=int, help="Override flow-matching denoising iterations (default: 10)."
    )
    parser.add_argument("--interpolate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.step_rate <= 0 or args.chunk_size_execute <= 0:
        raise ValueError("max_steps, step_rate, and chunk_size_execute must be positive.")
    if args.num_inference_steps is not None and args.num_inference_steps <= 0:
        raise ValueError("num-inference-steps must be positive.")
    if any(value <= 0 for value in (*args.camera_widths, *args.camera_heights, *args.camera_fps)):
        raise ValueError("Camera widths, heights, and FPS must be positive.")
    validate_action(args.reset_action)
    step_length = np.asarray(args.step_length, dtype=np.float64)
    if step_length.shape != (7,) or not np.isfinite(step_length).all() or np.any(step_length <= 0):
        raise ValueError("step-length must contain seven positive finite values.")


def get_frame(raw_observation: dict, camera_name: str) -> np.ndarray:
    key = f"{camera_name}/color/image_raw"
    if key not in raw_observation or "data" not in raw_observation[key]:
        raise RuntimeError(f"Missing camera frame: {key}")
    frame = np.asarray(raw_observation[key]["data"])
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.size == 0:
        raise RuntimeError(f"Invalid RGB frame from {camera_name}: shape={frame.shape}")
    if frame.dtype != np.uint8:
        raise RuntimeError(f"Camera frame from {camera_name} must be uint8; received {frame.dtype}.")
    timestamp = raw_observation[key].get("t")
    if not isinstance(timestamp, (int, np.integer)):
        raise RuntimeError(f"Missing nanosecond reception timestamp from {camera_name}.")
    age_seconds = (time.time_ns() - timestamp) / 1e9
    if not 0 <= age_seconds <= 2.0:
        raise RuntimeError(
            f"Camera frame from {camera_name} is stale or has an invalid timestamp: {age_seconds:.3f}s."
        )
    return frame


def make_observation(operator, camera_names: tuple[str, str], prompt: str) -> dict:
    raw_observation = operator.capture_observation()
    qpos = np.asarray(operator.get_qpos(raw_observation), dtype=np.float32)
    if qpos.shape != (7,) or not np.isfinite(qpos).all():
        raise RuntimeError("AIRBOT observation must contain seven finite joint/gripper values.")
    return {
        "observation.state": qpos,
        "observation.images.head": get_frame(raw_observation, camera_names[0]),
        "observation.images.wrist": get_frame(raw_observation, camera_names[1]),
        "task": prompt,
    }


def infer_chunk(policy, observation: dict, execute_length: int) -> np.ndarray:
    result = policy.infer(observation)
    action_chunk = result.get("action")
    if action_chunk is None:
        raise RuntimeError(f"LingBot policy did not return an action: {result.keys()}")
    action_chunk = np.asarray(action_chunk, dtype=np.float64)
    if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
        raise ValueError(f"LingBot action shape must be (chunk, 7); received {action_chunk.shape}.")
    if action_chunk.shape[0] < execute_length:
        raise ValueError(
            f"LingBot predicted {action_chunk.shape[0]} actions, less than chunk-size-execute={execute_length}."
        )
    return np.stack([validate_action(action) for action in action_chunk])


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    import torch

    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server

    os.chdir(Path(args.model_repo).resolve())
    RobotAH, RobotAHConfig, SystemMode, SpeedProfile = load_airbot_runtime(Path(args.openpi_root).resolve())

    policy = LingbotVLAv2Server(
        args.model_path,
        use_length=-1,
        chunk_ret=True,
        use_bf16=args.precision == "bf16",
        use_fp32=args.precision == "fp32",
        use_compile=args.use_compile,
        num_inference_steps=args.num_inference_steps,
    )
    policy.reset(robo_name="airbot_play")
    if args.chunk_size_execute > policy.config.chunk_size:
        raise ValueError("chunk-size-execute exceeds the model prediction horizon.")

    robot_options = {} if args.control_frequency is None else {"control_frequency": args.control_frequency}
    robot_config = RobotAHConfig(
        robot_groups=["left"],
        robot_interface=[args.robot_interface],
        camera_names=["base_0_rgb", "left_wrist_0_rgb"],
        camera_index=[args.head_camera, args.wrist_camera],
        camera_widths=args.camera_widths,
        camera_heights=args.camera_heights,
        camera_fps=args.camera_fps,
        **robot_options,
    )
    operator = RobotAH(robot_config)
    try:
        if input("Press Enter to reset and run, or q then Enter to quit...").strip().lower() == "q":
            return
        reset_action = validate_action(args.reset_action)
        operator.switch_mode(SystemMode.RESETTING)
        operator.set_speed_profile(SpeedProfile.SLOW)
        operator.move_to_joint_pos(reset_action)
        operator.switch_mode(SystemMode.SAMPLING)
        operator.set_speed_profile(SpeedProfile.FAST)

        previous_action = reset_action.copy()
        action_chunk = None
        with torch.inference_mode():
            for step in range(args.max_steps):
                action_index = step % args.chunk_size_execute
                if action_index == 0:
                    observation = make_observation(
                        operator,
                        ("base_0_rgb", "left_wrist_0_rgb"),
                        args.prompt,
                    )
                    started_at = time.monotonic()
                    action_chunk = infer_chunk(policy, observation, args.chunk_size_execute)
                    print(f"LingBot inference: {time.monotonic() - started_at:.3f}s")

                if action_chunk is None:
                    raise RuntimeError("LingBot policy did not produce an action chunk.")
                target = validate_action(action_chunk[action_index])
                commands = (
                    interpolate_action(args.step_length, previous_action, target)
                    if args.interpolate
                    else target[np.newaxis, :]
                )
                for command in commands:
                    operator.send_action(validate_action(command))
                    time.sleep(1.0 / args.step_rate)
                previous_action = target.copy()
    finally:
        try:
            operator.set_speed_profile(SpeedProfile.SLOW)
        finally:
            operator.shutdown()


if __name__ == "__main__":
    run(parse_args())
